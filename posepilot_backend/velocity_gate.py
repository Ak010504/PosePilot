import numpy as np

# ============================================================
# POSE-SPECIFIC JOINT GROUPS FOR VELOCITY GATING
# ============================================================
# Map pose names to the indices of joints that matter most for stability
# Feature indices (0-indexed): 
#   0=f1=left_elbow, 1=f2=right_elbow, 2=f3=left_shoulder, 3=f4=right_shoulder,
#   4=f5=left_knee, 5=f6=right_knee, 6=f7=neck, 7=f8=left_hip, 8=f9=right_hip

POSE_JOINT_GROUPS = {
    "warrior": [2, 3, 4, 5],        # shoulders (f3, f4) + knees (f5, f6) - arms extended, stable legs
    "tree": [4, 5, 6],              # knees (f5, f6) + neck (f7) - balance pose, core stability
    "chair": [4, 5, 7, 8],          # knees (f5, f6) + hips (f8, f9) - lower body focus
    "cobra": [0, 1, 2, 3, 6],       # elbows (f1, f2) + shoulders (f3, f4) + neck (f7) - upper body
    "downdog": [2, 3, 7, 8],        # shoulders (f3, f4) + hips (f8, f9) - inverted V stability
    "goddess": [4, 5, 7, 8],        # knees (f5, f6) + hips (f8, f9) - wide stance, lower body
    "surya_namaskar": [2, 3, 4, 5], # shoulders (f3, f4) + knees (f5, f6) - flow sequence, general
}

class VelocityGate:
    def __init__(
        self,
        window_size=20,          # smaller window
        velocity_threshold=3.0,  # slightly more tolerant
        min_stable_frames=4,     # SHORT holds
        cooldown_frames=12,      # prevents spam triggers
        pose=None                # NEW: pose name for joint-specific gating
    ):
        self.window_size = window_size
        self.velocity_threshold = velocity_threshold
        self.min_stable_frames = min_stable_frames
        self.cooldown_frames = cooldown_frames
        self.pose = pose

        self.buffer = []
        self.stable_count = 0
        self.cooldown = 0

        self.prev_velocity = None  # for smoothing
        
        # Get joint indices to monitor for this pose
        self.joint_indices = POSE_JOINT_GROUPS.get(pose, None)
        if self.joint_indices is None and pose is not None:
            print(f"[VelocityGate] Warning: No joint group defined for pose '{pose}', using all joints")

    def update(self, angle_frame):
        angle_frame = np.array(angle_frame)

        # -----------------------------
        # Add frame
        # -----------------------------
        self.buffer.append(angle_frame)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        if len(self.buffer) < 2:
            return False

        # -----------------------------
        # Smoothed velocity (EMA) - IMPROVED
        # -----------------------------
        # Calculate velocity only on pose-critical joints
        if self.joint_indices is not None:
            # Use only the joints that matter for this pose
            raw_velocity = np.mean(np.abs(
                self.buffer[-1][self.joint_indices] - self.buffer[-2][self.joint_indices]
            ))
        else:
            # Fallback to all joints if no specific group defined
            raw_velocity = np.mean(np.abs(self.buffer[-1] - self.buffer[-2]))

        if self.prev_velocity is None:
            velocity = raw_velocity
        else:
            velocity = 0.7 * self.prev_velocity + 0.3 * raw_velocity

        self.prev_velocity = velocity

        # -----------------------------
        # Cooldown logic
        # -----------------------------
        if self.cooldown > 0:
            self.cooldown -= 1
            return False

        # -----------------------------
        # Stability detection
        # -----------------------------
        if velocity < self.velocity_threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0

        # -----------------------------
        # Trigger when stable briefly
        # -----------------------------
        if self.stable_count >= self.min_stable_frames:
            self.stable_count = 0
            self.cooldown = self.cooldown_frames
            return True

        return False

    def get_window(self):
        return np.array(self.buffer)

    def reset(self):
        self.buffer.clear()
        self.stable_count = 0
        self.cooldown = 0
        self.prev_velocity = None