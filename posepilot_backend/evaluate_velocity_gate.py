#!/usr/bin/env python3
"""
VelocityGate Evaluation Metrics
Computes Precision/Recall/F1, False Trigger Rate, Trigger Latency
"""

import numpy as np
import pandas as pd
from velocity_gate import VelocityGate
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict

def evaluate_velocity_gate(sequence_angles, annotations, pose_name="chair"):
    """
    Evaluate VelocityGate on a sequence.
    
    Args:
        sequence_angles: (T, 9) array of angle features over time
        annotations: dict with 'stable_segments': list of [start, end] frame indices
                     where pose was stable (ground truth)
        pose_name: str, for joint selection
    
    Returns: dict of metrics
    """
    
    # Initialize gate
    gate = VelocityGate(
        window_size=20,
        velocity_threshold=3.0,
        min_stable_frames=4,
        cooldown_frames=12,
        pose=pose_name
    )
    
    # Run gate through entire sequence
    triggers = []
    for t, frame_angles in enumerate(sequence_angles):
        fired = gate.update(frame_angles)
        if fired:
            triggers.append(t)
    
    T = len(sequence_angles)
    true_stable = np.zeros(T)
    
    # Mark ground truth stable frames
    for start, end in annotations.get('stable_segments', []):
        true_stable[start:min(end+1, T)] = 1
    
    # Predictions: gate fires = predict stable frame
    pred_stable = np.zeros(T)
    for trigger_frame in triggers:
        # Window around trigger = predicted stable
        start = max(0, trigger_frame - gate.window_size//2)
        end = min(T, trigger_frame + gate.window_size//2)
        pred_stable[start:end] = 1
    
    # 1️⃣ Precision/Recall/F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_stable, pred_stable, average='binary', zero_division=0
    )
    
    # 2️⃣ False Trigger Rate
    false_triggers = sum(1 for t in triggers if true_stable[t] == 0)
    ftr_per_min = (false_triggers / T) * 60  # triggers per minute @ 30fps
    
    # 3️⃣ Trigger Latency
    latencies = []
    for start, end in annotations.get('stable_segments', []):
        # First stable frame in ground truth
        stable_start = start
        # First gate trigger after stable_start
        first_trigger = next((t for t in triggers if t >= stable_start), None)
        if first_trigger is not None:
            latencies.append(first_trigger - stable_start)
    
    avg_latency_frames = np.mean(latencies) if latencies else 0
    avg_latency_seconds = avg_latency_frames / 30  # @ 30fps
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_triggers_per_min': ftr_per_min,
        'avg_latency_frames': avg_latency_frames,
        'avg_latency_seconds': avg_latency_seconds,
        'num_true_segments': len(annotations.get('stable_segments', [])),
        'num_triggers': len(triggers),
    }

def evaluate_on_dataset(csv_files, pose_name, annotations_file=None):
    """
    Evaluate on multiple CSV files.
    
    Args:
        csv_files: list of paths to angle CSV files
        pose_name: str
        annotations_file: optional CSV with columns: filename, stable_start, stable_end
    """
    if annotations_file:
        ann_df = pd.read_csv(annotations_file)
        ann_dict = defaultdict(list)
        for _, row in ann_df.iterrows():
            ann_dict[row['filename']].append([row['stable_start'], row['stable_end']])
    
    all_metrics = []
    
    for csv_file in csv_files:
        # Load angles (assumes CSV has f1-f9 columns)
        df = pd.read_csv(csv_file)
        angles = df[['f1','f2','f3','f4','f5','f6','f7','f8','f9']].values
        
        # Get annotations for this file
        filename = os.path.basename(csv_file)
        annotations = {'stable_segments': ann_dict.get(filename, [])}
        
        metrics = evaluate_velocity_gate(angles, annotations, pose_name)
        metrics['filename'] = filename
        all_metrics.append(metrics)
    
    # Aggregate
    df_metrics = pd.DataFrame(all_metrics)
    
    agg_metrics = {
        'mean_precision': df_metrics['precision'].mean(),
        'mean_recall': df_metrics['recall'].mean(),
        'mean_f1': df_metrics['f1'].mean(),
        'mean_false_triggers_per_min': df_metrics['false_triggers_per_min'].mean(),
        'mean_latency_seconds': df_metrics['avg_latency_seconds'].mean(),
        'total_sequences': len(df_metrics),
    }
    
    return df_metrics, agg_metrics

# Example usage
if __name__ == "__main__":
    # Example: evaluate chair pose files
    chair_files = [
        "data/chair/sequence1.csv",
        "data/chair/sequence2.csv",
        # ... more files
    ]
    
    # Optional: annotations CSV format:
    # filename,stable_start,stable_end
    # sequence1.csv,10,25
    # sequence1.csv,35,50
    # sequence2.csv,5,20
    
    df_metrics, summary = evaluate_on_dataset(
        chair_files, 
        pose_name="chair",
        annotations_file="chair_annotations.csv"  # optional
    )
    
    print("Per-sequence metrics:")
    print(df_metrics.round(4))
    
    print("\nAggregate metrics:")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")
