"""
Compute evaluation metrics for ALL yoga correction models.

Metrics:
1. Overall MAE (degrees)
2. Per-joint MAE
3. % joints within ±5°, ±10°, ±15°
4. Peak-frame MAE

Uses:
- *_correction_model.pth
- *_correction_scalers.pkl
"""

import os
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from correction_model import CorrModel
from train_correction import prepare_correction_data


# ---------------- CONFIG ---------------- #
DATA_DIR = "data"
MODEL_DIR = "models"
CACHE_FILE = os.path.join(DATA_DIR, "cached_correction_features_v7.pkl")

BATCH_SIZE = 1
THRESHOLDS = [5, 10, 15]  # degrees
# ---------------------------------------- #


def inverse_scale(arr, scalers):
    """Inverse MinMax scaling (B, T, 9) → degrees"""
    arr_inv = arr.copy()
    for j in range(9):
        arr_inv[..., j] = scalers[j].inverse_transform(
            arr[..., j].reshape(-1, 1)
        ).reshape(arr[..., j].shape)
    return arr_inv


def compute_metrics(model, test_loader, scalers, device):
    model.eval()
    all_errors = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            pred_np = pred.cpu().numpy()
            y_np = y.cpu().numpy()

            pred_deg = inverse_scale(pred_np, scalers)
            y_deg = inverse_scale(y_np, scalers)

            all_errors.append(np.abs(pred_deg - y_deg))

    all_errors = np.concatenate(all_errors, axis=0)  # (N, T, 9)

    # Overall MAE
    mae = all_errors.mean()

    # Per-joint MAE
    per_joint_mae = all_errors.mean(axis=(0, 1))

    # Threshold metrics
    threshold_results = {
        th: (all_errors <= th).mean() * 100 for th in THRESHOLDS
    }

    # Peak-frame MAE
    frame_error = all_errors.mean(axis=2)  # (N, T)
    peak_idx = frame_error.argmax(axis=1)
    peak_mae = np.mean([
        all_errors[i, peak_idx[i]] for i in range(len(peak_idx))
    ])

    return mae, per_joint_mae, threshold_results, peak_mae


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load cached processed data
    with open(CACHE_FILE, "rb") as f:
        cached = pickle.load(f)

    normalized_data = cached["normalized_data"]
    asana_averages = cached["asana_averages"]

    print("\n" + "=" * 70)
    print("YOGA POSE CORRECTION — METRICS REPORT")
    print("=" * 70)

    for file in os.listdir(MODEL_DIR):
        if not file.endswith("_correction_model.pth"):
            continue  #  skip classification models

        asana = file.replace("_correction_model.pth", "")
        print(f"\n Evaluating: {asana.upper()}")

        # Load scalers
        scaler_path = os.path.join(
            MODEL_DIR, f"{asana}_correction_scalers.pkl"
        )
        with open(scaler_path, "rb") as f:
            scalers = pickle.load(f)

        # Prepare test data
        _, _, test_x, test_y = prepare_correction_data(
            normalized_data, asana, asana_averages
        )

        test_loader = DataLoader(
            TensorDataset(test_x, test_y),
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        # Load model
        model = CorrModel(
            input_size=9,
            hidden_size=256,
            num_layers=1,
            num_classes=9
        )
        model.load_state_dict(
            torch.load(os.path.join(MODEL_DIR, file), map_location=device)
        )
        model.to(device)

        # Compute metrics
        mae, per_joint_mae, threshold_results, peak_mae = compute_metrics(
            model, test_loader, scalers, device
        )

        # ---------- PRINT ----------
        print(f"  Overall MAE        : {mae:.2f}°")
        print(f"  Peak-frame MAE    : {peak_mae:.2f}°")

        print("  Per-joint MAE:")
        for i, v in enumerate(per_joint_mae, 1):
            print(f"    f{i}: {v:.2f}°")

        print("  % Joints within:")
        for th, val in threshold_results.items():
            print(f"    ±{th}° : {val:.2f}%")

    print("\n" + "=" * 70)
    print("METRICS COMPUTATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
