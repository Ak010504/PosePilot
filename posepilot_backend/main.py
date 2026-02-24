#!/usr/bin/env python3
# ============================================================================
# PosePilot FastAPI Backend - main.py (WITH VOICE FEEDBACK)
# ============================================================================
# Responsibilities:
# - REST + WebSocket API
# - MediaPipe landmark extraction
# - Call correction pipeline
# - Real-time voice feedback
# ============================================================================

import os
import cv2
import json
import pickle
import tempfile
import base64
import logging
import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import mediapipe as mp

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from feedback_rules import generate_feedback

from classify_model import ClassifyPose
from correction_predict import predict_correction_from_dataframe
from voice_feedback import voice_manager  # ✅ NEW IMPORT

# ============================================================================
# CONFIG
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PosePilot")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# ============================================================================
# FASTAPI SETUP
# ============================================================================

app = FastAPI(title="PosePilot Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL STATE
# ============================================================================

class ModelState:
    classify_model = None
    classify_scaler = None
    pose_mapping = None
    voice_enabled = True  # ✅ NEW: voice toggle

state = ModelState()

# ============================================================================
# MEDIAPIPE
# ============================================================================

mp_pose = mp.solutions.pose
pose_processor = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup():
    logger.info("🚀 Loading classification model...")

    with open(MODELS_DIR / "pose_mapping.pkl", "rb") as f:
        state.pose_mapping = pickle.load(f)

    with open(MODELS_DIR / "classify_scaler.pkl", "rb") as f:
        state.classify_scaler = pickle.load(f)

    model = ClassifyPose(
        input_size=680,
        hidden_size=32,
        num_layers=1,
        sequence_length=10,
        num_classes=len(state.pose_mapping),
    )

    model.load_state_dict(
        torch.load(MODELS_DIR / "pose_classification_model.pth", map_location=DEVICE)
    )
    model.to(DEVICE).eval()

    state.classify_model = model
    logger.info("✅ Classification model loaded")
    logger.info(f"🎤 Voice feedback: {'enabled' if voice_manager.enabled else 'disabled'}")

# ✅ NEW: Shutdown event
@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down voice feedback...")
    voice_manager.shutdown()

# ============================================================================
# HELPERS
# ============================================================================

def extract_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    rows = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = pose_processor.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.pose_landmarks:
            continue

        row = []
        for lm in result.pose_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])

        rows.append(row)

    cap.release()

    if not rows:
        return None

    cols = []
    for i in range(33):
        cols.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z", f"lm_{i}_vis"])

    return pd.DataFrame(rows, columns=cols)

# ============================================================================
# REST API
# ============================================================================

@app.post("/api/process-video")
async def process_video(video: UploadFile = File(...), pose: str = Form(...)):
    if pose not in state.pose_mapping:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown pose: {pose}"}
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(await video.read())
    tmp.close()

    try:
        logger.info("📹 Extracting landmarks...")
        df = extract_landmarks_from_video(tmp.name)

        if df is None or len(df) < 30:
            return JSONResponse(
                status_code=400,
                content={"error": "Insufficient pose data detected"}
            )

        result = predict_correction_from_dataframe(df, pose)

        if result["status"] != "success":
            return {
                "status": "ok",
                "feedback": ["Good alignment. Hold the pose."]
            }

        # ✅ NEW: Voice feedback for video processing
        if state.voice_enabled and result.get("feedback"):
            await asyncio.to_thread(
                voice_manager.speak_batch,
                result["feedback"],
                max_count=3
            )

        return {
            "status": "success",
            "pose": pose,
            "feedback": result["feedback"],
            "current_angles": result.get("current_angles", {}),
            "predicted_angles": result.get("predicted_angles", {}),
        }

    finally:
        os.remove(tmp.name)

# ============================================================================
# WEBSOCKET – REAL-TIME CORRECTION WITH VOICE
# ============================================================================

@app.websocket("/ws/realtime-correction")
async def realtime_correction(ws: WebSocket):
    await ws.accept()
    logger.info("🟢 WebSocket connected")

    buffer = []

    try:
        while True:
            payload = json.loads(await ws.receive_text())
            pose = payload.get("pose")
            frame_b64 = payload.get("frame")

            if not pose or not frame_b64:
                await ws.send_json({"status": "error", "msg": "Invalid payload"})
                continue

            frame_bytes = base64.b64decode(frame_b64)
            frame = cv2.imdecode(
                np.frombuffer(frame_bytes, np.uint8),
                cv2.IMREAD_COLOR
            )

            if frame is None:
                continue

            results = pose_processor.process(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            if not results.pose_landmarks:
                await ws.send_json({"status": "no_pose"})
                continue

            row = []
            for lm in results.pose_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z, lm.visibility])

            buffer.append(row)
            buffer = buffer[-120:]

            if len(buffer) < 15:
                await ws.send_json({"status": "warming_up"})
                continue

            cols = []
            for i in range(33):
                cols.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z", f"lm_{i}_vis"])

            df = pd.DataFrame(buffer, columns=cols)

            result = predict_correction_from_dataframe(df, pose)

            if result["status"] != "success":
                await ws.send_json({"status": "ok", "message": result.get("message", "")})
                continue

            feedback = result.get("feedback", [])
            
            if not feedback or feedback == ["Good alignment. Hold the pose."]:
                await ws.send_json({"status": "ok", "message": "Good alignment"})
                continue

            # ✅ NEW: Voice feedback (non-blocking)
            if state.voice_enabled and feedback:
                asyncio.create_task(
                    asyncio.to_thread(
                        voice_manager.speak_batch,
                        feedback,
                        max_count=3
                    )
                )

            await ws.send_json({
                "status": "feedback",
                "feedback": feedback,
                "pose": pose,
                "detected_pose": result.get("detected_pose"),  # for surya_namaskar
            })

    except WebSocketDisconnect:
        logger.info("🔴 WebSocket disconnected")
        # ✅ NEW: Clear voice queue on disconnect
        voice_manager.clear_queue()

# ============================================================================
# VOICE CONTROL API
# ============================================================================

@app.post("/api/voice/toggle")
async def toggle_voice():
    """Toggle voice feedback on/off"""
    state.voice_enabled = not state.voice_enabled
    voice_manager.enabled = state.voice_enabled
    
    if not state.voice_enabled:
        voice_manager.clear_queue()
    
    logger.info(f"🎤 Voice feedback: {'enabled' if state.voice_enabled else 'disabled'}")
    
    return {
        "voice_enabled": state.voice_enabled,
        "message": f"Voice feedback {'enabled' if state.voice_enabled else 'disabled'}"
    }

@app.get("/api/voice/status")
async def voice_status():
    """Get current voice feedback status"""
    return {
        "voice_enabled": state.voice_enabled,
        "tts_available": voice_manager.enabled,
        "is_speaking": voice_manager.is_speaking
    }

# ============================================================================
# HEALTH
# ============================================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "device": DEVICE,
        "voice_enabled": state.voice_enabled
    }

@app.get("/api/poses")
async def get_poses():
    """Return available yoga poses for frontend dropdown"""
    return {
        "poses": list(state.pose_mapping.keys()),
        "count": len(state.pose_mapping),
        "display_names": {
            pose: pose.replace("_", " ").title()
            for pose in state.pose_mapping.keys()
        }
    }

# ============================================================================
# ENTRY
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8001,
        ws_ping_interval=20,      # ✅ Prevent WebSocket timeout
        ws_ping_timeout=20,
        timeout_keep_alive=300
    )
