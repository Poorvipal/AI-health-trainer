# extract_features.py

import mediapipe as mp
import cv2
import numpy as np

mp_pose = mp.solutions.pose

def extract_pose_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found.")
        return None

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            print("No pose detected.")
            return None

        landmarks = results.pose_landmarks.landmark
        features = {}
        for idx, lm in enumerate(landmarks):
            features[f"landmark_{idx}_x"] = lm.x
            features[f"landmark_{idx}_y"] = lm.y
            features[f"landmark_{idx}_z"] = lm.z
            features[f"landmark_{idx}_visibility"] = lm.visibility

        return features
