import numpy as np

class VelocityGate:
    def __init__(
        self,
        window_size=20,          # smaller window
        velocity_threshold=3.0,  # slightly more tolerant
        min_stable_frames=4,     # SHORT holds
        cooldown_frames=12       # prevents spam triggers
    ):
        self.window_size = window_size
        self.velocity_threshold = velocity_threshold
        self.min_stable_frames = min_stable_frames
        self.cooldown_frames = cooldown_frames

        self.buffer = []
        self.stable_count = 0
        self.cooldown = 0

        self.prev_velocity = None  # for smoothing

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
        # Smoothed velocity (EMA)
        # -----------------------------
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
