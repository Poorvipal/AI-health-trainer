
# pose_estimation.py
import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2 +
        (p1.z - p2.z) ** 2
    )

def detect_pose(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("⚠️ Image not found. Check the path.")
        return

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            print("❌ No pose detected.")
            return

        landmarks = results.pose_landmarks.landmark

        # Drawing the landmarks on image (optional)
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow("Pose", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Extract landmark distances
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        shoulder_width = calculate_distance(left_shoulder, right_shoulder)

        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        hip_width = calculate_distance(left_hip, right_hip)

        # Leg length (hip to ankle)
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        leg_length = (calculate_distance(left_hip, left_ankle) + calculate_distance(right_hip, right_ankle)) / 2

        # Arm length (shoulder to wrist)
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        arm_length = (calculate_distance(left_shoulder, left_wrist) + calculate_distance(right_shoulder, right_wrist)) / 2

        print(f"\n📏 Measurements:")
        print(f" - Shoulder Width: {shoulder_width:.4f}")
        print(f" - Hip Width: {hip_width:.4f}")
        print(f" - Leg Length: {leg_length:.4f}")
        print(f" - Arm Length: {arm_length:.4f}")

        # Ratio & Body Type Classification
        if hip_width != 0:
            ratio = shoulder_width / hip_width
            print(f"\n📐 Shoulder-to-Hip Ratio: {ratio:.2f}")

            if ratio > 1.25:
                body_type = "Inverted Triangle"
                exercises = ["Squats", "Lunges", "Deadlifts"]
            elif ratio < 0.85:
                body_type = "Pear Shape"
                exercises = ["Push-ups", "Shoulder Press", "Planks"]
            else:
                body_type = "Rectangle"
                exercises = ["Burpees", "Mountain Climbers", "Jumping Jacks"]

            print(f"\n🧍 Body Type Detected: {body_type}")
            print("🏋️ Recommended Exercises:")
            for ex in exercises:
                print(f" - {ex}")
        else:
            print("⚠️ Hip width is zero, can't compute ratio.")



