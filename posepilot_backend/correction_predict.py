# correction_predict.py

import os
import torch
import pickle
import numpy as np

from correction_model import CorrModel
from feedback_rules import generate_feedback
from velocity_gate import VelocityGate
from utils import (
    structure_data,
    update_body_pose_landmarks,
    correction_angles_convert,
)

# ==================================================
# CONFIG
# ==================================================

WINDOW_SIZE = 30

# ==================================================
# GLOBAL STATE (IMPORTANT)
# ==================================================
# VelocityGate must persist across frames
velocity_gates = {}

# Cache model + scalers to avoid reloading every frame
model_cache = {}

# ==================================================
# MODEL LOADER
# ==================================================

def load_model_and_scalers(device, pose):
    """
    Loads correction model + scalers once and caches them.
    """
    if pose in model_cache:
        return model_cache[pose]

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

    model_path = os.path.join(
        PROJECT_ROOT, "models", f"{pose}_correction_model.pth"
    )
    scalers_path = os.path.join(
        PROJECT_ROOT, "models", f"{pose}_correction_scalers.pkl"
    )

    model = CorrModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

    model_cache[pose] = (model, scalers)
    return model, scalers

# ==================================================
# REAL-TIME CORRECTION PREDICTION
# ==================================================

def predict_correction_from_dataframe(df, pose):
    """
    REAL-TIME correction logic.
    - Called once per incoming frame batch (WebSocket)
    - Uses VelocityGate to trigger only when user is stable
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scalers = load_model_and_scalers(device, pose)

    # --------------------------------------------------
    # 1. Extract pose landmarks
    # --------------------------------------------------

    pose_data = df.iloc[:, :132].copy()

    structured_df, body_pose_landmarks = structure_data(pose_data)
    structured_df, body_pose_landmarks = update_body_pose_landmarks(
        structured_df, body_pose_landmarks
    )

    # --------------------------------------------------
    # 2. Convert to joint angles
    # --------------------------------------------------

    angle_df = correction_angles_convert(structured_df)
    angle_df = angle_df[[f"f{i}" for i in range(1, 10)]]

    angles = angle_df.values
    print(f"Extracted angle features. Shape: {angles.shape}")

    if len(angles) < 2:
        return {
            "status": "warming_up",
            "message": "Collecting frames..."
        }

    # --------------------------------------------------
    # 3. Initialize persistent VelocityGate
    # --------------------------------------------------

    if pose not in velocity_gates:
        velocity_gates[pose] = VelocityGate(
            window_size=WINDOW_SIZE,
            velocity_threshold=2.0,
            min_stable_frames=8,
        )

    gate = velocity_gates[pose]

    # --------------------------------------------------
    # 4. Update gate with ONLY latest frame
    # --------------------------------------------------

    latest_angle = angles[-1]
    triggered = gate.update(latest_angle)

    if not triggered:
        return {
            "status": "warming_up",
            "message": "Hold still to analyze posture"
        }

    print("[VELOCITY GATE] Triggered")

    window = gate.get_window()

    if window.shape[0] < WINDOW_SIZE:
        return {
            "status": "warming_up",
            "message": "Stabilizing posture..."
        }

    # --------------------------------------------------
    # 5. Prepare model input
    # --------------------------------------------------

    # Current (unscaled) angles
    current_angles = {
        f"f{i+1}": float(window[-1][i])
        for i in range(9)
    }

    # Normalize window
    window_norm = window.copy()
    for i, scaler in enumerate(scalers):
        window_norm[:, i] = scaler.transform(
            window_norm[:, i].reshape(-1, 1)
        ).flatten()

    input_tensor = (
        torch.tensor(window_norm, dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
    )

    # --------------------------------------------------
    # 6. Model inference
    # --------------------------------------------------

    with torch.no_grad():
        prediction = model(input_tensor).cpu().numpy()[0]

    # Inverse scale prediction
    predicted_angles = {
        f"f{i+1}": float(
            scalers[i].inverse_transform([[prediction[i]]])[0][0]
        )
        for i in range(9)
    }

    # --------------------------------------------------
    # 7. Generate feedback
    # --------------------------------------------------

    feedback = generate_feedback(
        pose,
        current_angles=current_angles,
        predicted_angles=predicted_angles,
    )

    return {
        "status": "success",
        "pose": pose,
        "feedback": feedback,
        "current_angles": current_angles,
        "predicted_angles": predicted_angles,
    }
