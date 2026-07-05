import cv2
import mediapipe as mp
import joblib
import numpy as np

# ==========================
# Load Trained Model
# ==========================

model = joblib.load("gesture_model.pkl")

# ==========================
# MediaPipe Setup
# ==========================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ==========================
# Webcam
# ==========================

cap = cv2.VideoCapture(0)

print("Starting SignHealth AI...")
print("Press Q to quit")

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    prediction = "No Hand"
    confidence = 0

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # ==========================
            # Extract Normalized Features
            # ==========================

            landmark_row = []

            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y
            wrist_z = hand_landmarks.landmark[0].z

            for lm in hand_landmarks.landmark:

                landmark_row.extend([
                    lm.x - wrist_x,
                    lm.y - wrist_y,
                    lm.z - wrist_z
                ])

            # ==========================
            # Prediction
            # ==========================

            probs = model.predict_proba([landmark_row])[0]

            confidence = np.max(probs)

            prediction = model.classes_[np.argmax(probs)]

            # Reject uncertain predictions

            if confidence < 0.60:
                prediction = "Unknown Gesture"

            # ==========================
            # Color Coding
            # ==========================

            if confidence >= 0.90:
                color = (0, 255, 0)      # Green

            elif confidence >= 0.70:
                color = (0, 255, 255)    # Yellow

            else:
                color = (0, 0, 255)      # Red

            text = f"{prediction} ({confidence*100:.1f}%)"

            cv2.putText(
                frame,
                text,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

    # ==========================
    # Title
    # ==========================

    cv2.putText(
        frame,
        "SignHealth AI",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("SignHealth AI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()