#!/usr/bin/env python3
"""
TCN-based pose correction training (FIXED).

Key changes vs old version:
- Target = IDEAL angles (NOT last frame)
- Shorter window for correction
- Explicit joint supervision (elbows + knees)
- Per-pose training
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import pickle
import argparse
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from correction_model import CorrModel
from utils import (
    structure_data,
    update_body_pose_landmarks,
    correction_angles_convert,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

WINDOW_SIZE = 16          # SHORT window = correction, not forecasting
TRAIN_TEST_SPLIT = 0.8

FEATURE_COLS = [
    "f1", "f2",   # elbows
    "f3", "f4",   # shoulders
    "f5", "f6",   # knees
    "f7",         # neck
    "f8", "f9"    # hips
]

# ============================================================
# IDEAL TARGET ANGLES (DEGREES)
# ============================================================

IDEAL_ANGLES = {
    "warrior":  [160,160, 90,90, 120,120, 20, 180,180],
    "tree":     [170,170, 80,80,  60,60,  15, 170,170],
    "chair":    [160,160, 150,150, 90,90,  20, 120,120],
    "cobra":    [170,170, 60,60,  180,180, 30, 180,180],
    "downdog":  [170,170, 160,160, 170,170, 20, 200,200],
    # ❌ Do NOT include full surya_namaskar here
}

# ============================================================
# DATA LOADING
# ============================================================

def load_all_asana_data(data_dir):
    all_data = {}

    for pose in os.listdir(data_dir):
        pose_dir = os.path.join(data_dir, pose)
        if not os.path.isdir(pose_dir):
            continue

        sequences = []
        for file in os.listdir(pose_dir):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(pose_dir, file))
                sequences.append(df)

        if sequences:
            all_data[pose] = sequences
            print(f"Loaded {len(sequences)} sequences for pose: {pose}")

    if not all_data:
        raise ValueError("No pose data found.")

    return all_data

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_angle_features(df):
    pose_data = df.iloc[:, :132].copy()

    structured_df, body_pose_landmarks = structure_data(pose_data)
    structured_df, body_pose_landmarks = update_body_pose_landmarks(
        structured_df, body_pose_landmarks
    )

    angle_df = correction_angles_convert(structured_df)
    return angle_df[FEATURE_COLS]

# ============================================================
# NORMALIZATION
# ============================================================

def fit_global_scalers(all_angle_data):
    scalers = []
    for col in FEATURE_COLS:
        scaler = MinMaxScaler()
        scaler.fit(all_angle_data[col].values.reshape(-1, 1))
        scalers.append(scaler)
    return scalers


def apply_scalers(df, scalers):
    df = df.copy()
    for i, col in enumerate(FEATURE_COLS):
        df[col] = scalers[i].transform(
            df[col].values.reshape(-1, 1)
        ).flatten()
    return df

# ============================================================
# SLIDING WINDOW DATASET (FIXED TARGET)
# ============================================================

def build_sliding_windows(sequences, ideal_target):
    """
    sequences: list of (T, 9) arrays
    ideal_target: (9,)
    """
    X, y = [], []

    for seq in sequences:
        if len(seq) < WINDOW_SIZE:
            continue

        for i in range(len(seq) - WINDOW_SIZE + 1):
            window = seq[i : i + WINDOW_SIZE]
            X.append(window)
            y.append(ideal_target)

    if len(X) == 0:
        raise ValueError("No valid sliding windows created.")

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )

# ============================================================
# TRAINING
# ============================================================

def train_model(model, train_loader, test_loader, epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += criterion(pred, y).item()

        test_loss /= len(test_loader)

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Train: {train_loss:.6f} | "
            f"Test: {test_loss:.6f}"
        )

# ============================================================
# SAVE
# ============================================================

def save_model_and_scalers(model, scalers, pose, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    torch.save(
        model.state_dict(),
        os.path.join(output_dir, f"{pose}_correction_model.pth")
    )

    with open(
        os.path.join(output_dir, f"{pose}_correction_scalers.pkl"), "wb"
    ) as f:
        pickle.dump(scalers, f)

    print(f"Saved model + scalers for pose: {pose}")

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--output_dir", default="models", type=str)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)

    args = parser.parse_args()

    print("\n=== TRAINING TCN CORRECTION MODELS ===\n")

    asana_data = load_all_asana_data(args.data_dir)

    processed = {}
    all_angle_rows = []

    for pose, dfs in asana_data.items():
        pose_sequences = []
        for df in dfs:
            angle_df = extract_angle_features(df)
            pose_sequences.append(angle_df)
            all_angle_rows.append(angle_df)
        processed[pose] = pose_sequences

    all_angle_data = pd.concat(all_angle_rows, ignore_index=True)

    scalers = fit_global_scalers(all_angle_data)

    for pose, dfs in processed.items():

        if pose not in IDEAL_ANGLES:
            print(f"Skipping {pose} (no ideal target)")
            continue

        print(f"\n{'='*50}")
        print(f"Training correction model for pose: {pose}")
        print(f"{'='*50}")

        normalized_sequences = [
            apply_scalers(df, scalers).values for df in dfs
        ]

        ideal_target = np.array(IDEAL_ANGLES[pose], dtype=np.float32)

        X, y = build_sliding_windows(normalized_sequences, ideal_target)

        split = int(TRAIN_TEST_SPLIT * len(X))
        train_X, test_X = X[:split], X[split:]
        train_y, test_y = y[:split], y[split:]

        train_loader = DataLoader(
            TensorDataset(train_X, train_y),
            batch_size=args.batch_size,
            shuffle=True,
        )

        test_loader = DataLoader(
            TensorDataset(test_X, test_y),
            batch_size=args.batch_size,
        )

        model = CorrModel(
            input_size=9,
            channels=(32, 32),
            kernel_size=3
        )

        train_model(
            model,
            train_loader,
            test_loader,
            epochs=args.epochs,
            lr=args.lr,
        )

        save_model_and_scalers(model, scalers, pose, args.output_dir)

    print("\n✅ All correction models trained successfully.\n")


if __name__ == "__main__":
    main()
