import cv2
import mediapipe as mp
import joblib
import numpy as np
from gtts import gTTS
from playsound import playsound
import tempfile
import threading
import pyttsx3
import csv
import os
from datetime import datetime

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("gesture_model.pkl")

# ==========================================
# SPEECH ENGINE
# ==========================================

engine = pyttsx3.init()

engine.setProperty("rate", 150)

voices = engine.getProperty("voices")

if len(voices) > 1:
    engine.setProperty("voice", voices[1].id)

# ==========================================
# SENTENCE MAPPING
# ==========================================

sentence_map = {
    "pain": "Patient is experiencing pain",
    "help": "Patient needs assistance",
    "water": "Patient needs water",
    "medicine": "Patient requires medicine",
    "emergency": "Emergency assistance required",
    "head": "Patient indicates head area",
    "chest": "Patient indicates chest area"
}

# ==========================================
# HISTORY FILE
# ==========================================

history_file = "history.csv"

if not os.path.exists(history_file):

    with open(history_file, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "gesture",
            "confidence",
            "sentence"
        ])

# ==========================================
# SAVE HISTORY
# ==========================================

def save_history(
    gesture,
    confidence,
    sentence
):

    with open(
        history_file,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            datetime.now(),
            gesture,
            round(confidence * 100, 2),
            sentence
        ])

# ==========================================
# SPEAK
# ==========================================

def speak(text):

    def run():

        try:

            tts = gTTS(
                text=text,
                lang='en'
            )

            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp3"
            )

            filename = temp_file.name

            temp_file.close()

            tts.save(filename)

            playsound(filename)

            os.remove(filename)

        except Exception as e:

            print("Speech Error:", e)

    threading.Thread(
        target=run,
        daemon=True
    ).start()


# ==========================================
# MEDIAPIPE SETUP
# ==========================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

stable_prediction = ""
stable_count = 0
last_prediction = ""


print("SignHealth AI Started")

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    prediction = "No Hand"
    confidence = 0

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

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

            probs = model.predict_proba(
                [landmark_row]
            )[0]

            confidence = np.max(probs)

            prediction = model.classes_[
                np.argmax(probs)
            ]

            if confidence < 0.85:

                prediction = "Unknown Gesture"

            sentence = sentence_map.get(
                prediction,
                "Unable to understand gesture"
            )

            # ==================================
            # SPEAK ONLY ON CHANGE
            # ==================================

            if prediction == stable_prediction:

                stable_count += 1

            else:

                stable_prediction = prediction
                stable_count = 0


            if (
                stable_count >= 10
                and prediction != last_prediction
                and prediction != "Unknown Gesture"
            ):

                print(f"Confirmed: {prediction}")

                speak(sentence)

                save_history(
                    prediction,
                    confidence,
                    sentence
                )

                last_prediction = prediction
            # ==================================
            # COLOR
            # ==================================

            if confidence >= 0.90:

                color = (0,255,0)

            elif confidence >= 0.70:

                color = (0,255,255)

            else:

                color = (0,0,255)

            # ==================================
            # DISPLAY
            # ==================================

            cv2.putText(
                frame,
                f"Gesture: {prediction}",
                (10,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence*100:.1f}%",
                (10,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            cv2.putText(
                frame,
                sentence,
                (10,120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2
            )
        

    cv2.putText(
        frame,
        "SignHealth AI",
        (10,160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    cv2.imshow(
        "SignHealth AI",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()