import cv2
import mediapipe as mp
import csv
import os

GESTURE_NAME = "chest"   # Change later

os.makedirs("dataset", exist_ok=True)

csv_file = f"dataset/{GESTURE_NAME}.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

sample_count = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            row = []

            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y
            wrist_z = hand_landmarks.landmark[0].z

            for lm in hand_landmarks.landmark:

                row.extend([
                    lm.x - wrist_x,
                    lm.y - wrist_y,
                    lm.z - wrist_z
                ])

            key = cv2.waitKey(1)

            if key == ord("s"):

                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                sample_count += 1

                print(
                    f"{GESTURE_NAME} saved : {sample_count}"
                )

    cv2.putText(
        frame,
        f"Samples : {sample_count}",
        (10,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Dataset Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()