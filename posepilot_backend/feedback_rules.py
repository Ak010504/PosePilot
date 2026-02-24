# feedback_rules.py

"""
Rule-based conversion of angle deltas into human-readable yoga feedback.
This layer sits BETWEEN the ML model and the user.
"""

# -------------------------------
# Angle → Joint mapping
# -------------------------------

ANGLE_JOINT_MAP = {
    "f1": "left_elbow",
    "f2": "right_elbow",
    "f3": "left_shoulder",
    "f4": "right_shoulder",
    "f5": "left_knee",
    "f6": "right_knee",
    "f7": "neck",
    "f8": "left_hip",
    "f9": "right_hip",
}


# -------------------------------
# Pose-specific correction rules
# -------------------------------

POSE_RULES = {

    # =========================
    # WARRIOR POSE
    # =========================
    "warrior": {
        "left_knee": {
            "threshold": 8,
            "positive": "Straighten your left knee slightly",
            "negative": "Bend your left knee a little more",
        },
        "right_knee": {
            "threshold": 8,
            "positive": "Straighten your right leg",
            "negative": "Bend your right knee slightly",
        },
        "left_shoulder": {
            "threshold": 10,
            "positive": "Lift your left arm higher",
            "negative": "Lower your left arm slightly",
        },
        "right_shoulder": {
            "threshold": 10,
            "positive": "Lift your right arm higher",
            "negative": "Lower your right arm slightly",
        },
    },

    # =========================
    # TREE POSE
    # =========================
    "tree": {
        "left_knee": {
            "threshold": 3,
            "positive": "Lift your left knee higher",
            "negative": "Lower your left knee slightly",
        },
        "right_knee": {
            "threshold": 3,
            "positive": "Lift your right knee higher",
            "negative": "Lower your right knee slightly",
        },
        "neck": {
            "threshold": 6,
            "positive": "Keep your head upright",
            "negative": "Relax your neck slightly",
        },
    },

    # =========================
    # COBRA POSE
    # =========================
    "cobra": {
        "left_shoulder": {
            "threshold": 10,
            "positive": "Open your left shoulder more",
            "negative": "Relax your left shoulder slightly",
        },
        "right_shoulder": {
            "threshold": 10,
            "positive": "Open your right shoulder more",
            "negative": "Relax your right shoulder slightly",
        },
        "neck": {
            "threshold": 6,
            "positive": "Lift your chest and gaze upward",
            "negative": "Lower your chest slightly",
        },
    },

    # =========================
    # CHAIR POSE
    # =========================
    "chair": {
        "left_knee": {
            "threshold": 6,
            "positive": "Sit deeper into your left knee",
            "negative": "Ease out of your left knee slightly",
        },
        "right_knee": {
            "threshold": 6,
            "positive": "Sit deeper into your right knee",
            "negative": "Ease out of your right knee slightly",
        },
        "left_hip": {
            "threshold": 5,
            "positive": "Lower your hips slightly",
            "negative": "Lift your hips a little",
        },
        "right_hip": {
            "threshold": 5,
            "positive": "Lower your hips slightly",
            "negative": "Lift your hips a little",
        },
    },

    # =========================
    # DOWNWARD DOG
    # =========================
    "downdog": {
        "left_shoulder": {
            "threshold": 8,
            "positive": "Push your left shoulder away from the mat",
            "negative": "Soften your left shoulder slightly",
        },
        "right_shoulder": {
            "threshold": 8,
            "positive": "Push your right shoulder away from the mat",
            "negative": "Soften your right shoulder slightly",
        },
        "left_hip": {
            "threshold": 10,
            "positive": "Lift your hips higher",
            "negative": "Lower your hips slightly",
        },
        "right_hip": {
            "threshold": 10,
            "positive": "Lift your hips higher",
            "negative": "Lower your hips slightly",
        },
    },

    # =========================
    # GODDESS POSE
    # =========================
    "goddess": {
        "left_knee": {
            "threshold": 10,
            "positive": "Bend your left knee more",
            "negative": "Straighten your left knee slightly",
        },
        "right_knee": {
            "threshold": 10,
            "positive": "Bend your right knee more",
            "negative": "Straighten your right knee slightly",
        },
        "neck": {
            "threshold": 6,
            "positive": "Lift your chest and head",
            "negative": "Relax your neck slightly",
        },
    },

    # =========================
    # SURYA NAMASKAR (FLOW)
    # =========================
    "surya_namaskar": {
        "left_shoulder": {
            "threshold": 6,
            "positive": "Raise your arms fully overhead",
            "negative": "Lower your arms slightly",
        },
        "right_shoulder": {
            "threshold": 6,
            "positive": "Raise your arms fully overhead",
            "negative": "Lower your arms slightly",
        },
        "neck": {
            "threshold": 5,
            "positive": "Lift your chest and lengthen your spine",
            "negative": "Relax your upper body slightly",
        },
        "left_elbow": {
        "threshold": 6,
        "positive": "Straighten your left arm",
        "negative": "Soften your left elbow slightly",
    },
    "right_elbow": {
        "threshold": 6,
        "positive": "Straighten your right arm",
        "negative": "Soften your right elbow slightly",
    },

    "left_knee": {
        "threshold": 8,
        "positive": "Straighten your left leg",
        "negative": "Bend your left knee slightly",
    },
    "right_knee": {
        "threshold": 8,
        "positive": "Straighten your right leg",
        "negative": "Bend your right knee slightly",
    },
    },
}


# -------------------------------
# Feedback generator
# -------------------------------

def generate_feedback(pose, current_angles, predicted_angles):
    """
    Convert angle deltas into human feedback.
    """

    feedback = []

    if pose not in POSE_RULES:
        return ["Good alignment. Hold the pose."]

    rules = POSE_RULES[pose]

    for feature, joint_name in ANGLE_JOINT_MAP.items():

        # ---- Safety checks ----
        if joint_name not in rules:
            continue

        if feature not in current_angles or feature not in predicted_angles:
            print(f"[DEBUG] Missing angle for {joint_name} ({feature})")
            continue

        rule = rules[joint_name]
        threshold = rule["threshold"]

        current_angle = current_angles[feature]
        predicted_angle = predicted_angles[feature]
        delta = predicted_angle - current_angle

        # ---- Debug log (THIS IS IMPORTANT) ----
        print(
            f"[DEBUG] {joint_name}: "
            f"current={current_angle:.2f}, "
            f"predicted={predicted_angle:.2f}, "
            f"delta={delta:.2f}, "
            f"threshold={threshold}"
        )

        # ---- Feedback logic ----
        if abs(delta) < threshold:
            continue

        if delta > 0:
            feedback.append(rule["positive"])
        else:
            feedback.append(rule["negative"])

    if not feedback:
        feedback.append("Good alignment. Hold the pose.")

    return feedback
