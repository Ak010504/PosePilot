"""
correction_predict.py — SURYA NAMASKAR AWARE (IMPROVED VERSION)
"""
import os
import torch
import pickle
import numpy as np
from collections import deque, Counter
from pathlib import Path
from correction_model import CorrModel
from feedback_rules import generate_feedback
from velocity_gate import VelocityGate
from utils import (
    structure_data,
    update_body_pose_landmarks,
    correction_angles_convert,
)

# ─────────────────────────────────────────────
# CONFIG - IMPROVED
# ─────────────────────────────────────────────
WINDOW_SIZE = 30
CLASSIFIER_WINDOW = 20  # ✅ INCREASED from 15 to 20 for better detection
SURYA_SUB_POSE_MODEL_MAP = {
    "cobra": "cobra",
    "downdog": "downdog",
    "warrior": "warrior",
    "chair": "chair",
    "tree": "tree",        # ✅ ADDED
    "goddess": "goddess",  # ✅ ADDED
}

IDEAL_ANGLES = {
    "warrior": [160,160,90,90,120,120,20,180,180],
    "tree": [170,170,80,80,60,60,15,170,170],
    "chair": [160,160,150,150,90,90,20,120,120],
    "cobra": [170,170,60,60,180,180,30,180,180],
    "downdog": [170,170,160,160,170,170,20,200,200],
    "goddess": [160,160,150,150,120,120,20,140,140],  # ✅ ADDED
}

velocity_gates = {}
model_cache = {}

# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────
def load_model_and_scalers(device, pose):
    if pose in model_cache:
        return model_cache[pose]
    
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent
    models_dir = PROJECT_ROOT / "models"
    model_path = models_dir / f"{pose}_correction_model.pth"
    scalers_path = models_dir / f"{pose}_correction_scalers.pkl"
    
    model = CorrModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)
    
    model_cache[pose] = (model, scalers)
    print(f"[ModelCache] Loaded correction model for '{pose}'")
    return model, scalers

# ─────────────────────────────────────────────
# SUB-POSE CLASSIFIER - IMPROVED
# ─────────────────────────────────────────────
class SubPoseClassifier:
    _model = None
    _scaler = None
    _mapping = None
    _disabled = False
    _vote_buffer = deque(maxlen=7)  # ✅ INCREASED from 5 to 7
    
    @classmethod
    def _load(cls):
        if cls._model is not None or cls._disabled:
            return
        try:
            from classify_predict import config_model
            BASE_DIR = Path(__file__).resolve().parent
            PROJECT_ROOT = BASE_DIR.parent
            MODEL_DIR = PROJECT_ROOT / "models"
            
            cls._model, cls._scaler, cls._mapping = config_model(
                model_path=MODEL_DIR / "pose_classification_model.pth",
                scaler_path=MODEL_DIR / "classify_scaler.pkl",
                mapping_path=MODEL_DIR / "pose_mapping.pkl",
            )
            print(f"Loaded pose mapping with {len(cls._mapping)} classes: {cls._mapping}")
            print("[SubPoseClassifier] Classifier loaded.")
        except Exception as e:
            print(f"[SubPoseClassifier] DISABLED: {e}")
            cls._disabled = True
    
    @classmethod
    def detect(cls, df):
        cls._load()
        if cls._disabled or cls._model is None:
            return None
        
        try:
            from classify_predict import predict
            label = predict(df, cls._model, cls._scaler, cls._mapping)
            
            # ✅ FIXED: Handle empty predictions
            if label is None or label == "":
                return None
                
            cls._vote_buffer.append(label)
            
            # ✅ DEBUG: Show votes
            votes = list(cls._vote_buffer)
            print(f"[Classifier Votes] {votes}")
            
            # ✅ IMPROVED: Use majority vote with lower threshold
            if len(votes) >= 3:
                most_common, count = Counter(votes).most_common(1)[0]
                # Require only 40% agreement (3/7) instead of absolute majority
                if count >= max(2, len(votes) * 0.4):
                    print(f"[Surya] Detected sub-pose: {most_common} → using model: {most_common}")
                    return most_common
            
            return None
        except Exception as e:
            print(f"[SubPoseClassifier] detection error: {e}")
            return None

# ─────────────────────────────────────────────
# MAIN FUNCTION - IMPROVED
# ─────────────────────────────────────────────
def predict_correction_from_dataframe(df, pose):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1️⃣ Extract angles
    pose_data = df.iloc[:, :132].copy()
    structured_df, body_pose_landmarks = structure_data(pose_data)
    structured_df, body_pose_landmarks = update_body_pose_landmarks(
        structured_df, body_pose_landmarks
    )
    
    angle_df = correction_angles_convert(structured_df)
    angle_df = angle_df[[f"f{i}" for i in range(1, 10)]]
    angles = angle_df.values
    
    print(f"Extracted angle features. Shape: {angles.shape}")
    
    if len(angles) < 2:
        return {"status": "warming_up", "message": "Collecting frames…"}
    
    is_flow = (pose == "surya_namaskar")
    model_pose = pose
    detected_sub_pose = None
    
    # 2️⃣ Velocity gate - RELAXED for flow sequences
    gate_key = pose
    if gate_key not in velocity_gates:
        velocity_gates[gate_key] = VelocityGate(
            window_size=WINDOW_SIZE,
            velocity_threshold=3.5 if is_flow else 2.0,  # ✅ RELAXED from 2.5 to 3.5
            min_stable_frames=4 if is_flow else 8,        # ✅ REDUCED from 6 to 4
            pose=pose,
        )
    
    gate = velocity_gates[gate_key]
    latest = angles[-1]
    triggered = gate.update(latest)
    
    if not triggered:
        return {"status": "warming_up", "message": "Hold still…"}
    
    window = gate.get_window()
    if window.shape[0] < WINDOW_SIZE:
        return {"status": "warming_up", "message": "Stabilising…"}
    
    print("[VELOCITY GATE] ✅ Triggered")
    
    # 3️⃣ Classify sub-pose (stable pose)
    if is_flow:
        if len(df) >= CLASSIFIER_WINDOW:
            df_for_classification = df.iloc[-CLASSIFIER_WINDOW:].copy()
        else:
            df_for_classification = df.copy()
        
        detected_sub_pose = SubPoseClassifier.detect(df_for_classification)
        
        if detected_sub_pose is None:
            return {"status": "detecting", "message": "Identifying pose…"}
        
        if detected_sub_pose not in SURYA_SUB_POSE_MODEL_MAP:
            print(f"[WARNING] Detected '{detected_sub_pose}' but no model mapping exists")
            return {
                "status": "ok",
                "message": f"No correction model for '{detected_sub_pose}'",
                "detected_pose": detected_sub_pose,
            }
        
        model_pose = SURYA_SUB_POSE_MODEL_MAP[detected_sub_pose]
        print(f"[Surya] Using correction model: {model_pose}")
    
    # 4️⃣ Load correction model
    try:
        model, scalers = load_model_and_scalers(device, model_pose)
    except FileNotFoundError as e:
        print(f"[ERROR] Model file not found: {e}")
        return {"status": "error", "message": f"No model for '{model_pose}'"}
    
    # 5️⃣ Normalize
    current_angles = {f"f{i+1}": float(window[-1][i]) for i in range(9)}
    window_norm = window.copy()
    
    for i, scaler in enumerate(scalers):
        window_norm[:, i] = scaler.transform(
            window_norm[:, i].reshape(-1, 1)
        ).flatten()
    
    input_tensor = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 6️⃣ Predict
    with torch.no_grad():
        raw_pred = model(input_tensor).cpu().numpy()[0]
    
    predicted_angles = {
        f"f{i+1}": float(scalers[i].inverse_transform([[raw_pred[i]]])[0][0])
        for i in range(9)
    }
    
    # 7️⃣ Ideal override - REDUCED strictness
    effective_pose = detected_sub_pose if is_flow else pose
    if effective_pose in IDEAL_ANGLES:
        ideal = IDEAL_ANGLES[effective_pose]
        for i in range(9):
            key = f"f{i+1}"
            # ✅ INCREASED threshold from 1.5 to 3.0 degrees
            if abs(predicted_angles[key] - current_angles[key]) < 3.0:
                predicted_angles[key] = float(ideal[i])
    
    # 8️⃣ Generate feedback
    feedback = generate_feedback(
        effective_pose,
        current_angles=current_angles,
        predicted_angles=predicted_angles,
    )
    
    # ✅ DEBUG: Print feedback generation details
    print(f"[FEEDBACK] Pose: {effective_pose}, Messages: {len(feedback)}")
    for msg in feedback:
        print(f"  → {msg}")
    
    if is_flow and detected_sub_pose:
        feedback = feedback
    
    return {
        "status": "success",
        "pose": pose,
        "actual_pose": effective_pose,
        "feedback": feedback,
        "detected_pose": detected_sub_pose,
        "current_angles": current_angles,
        "predicted_angles": predicted_angles,
    }
