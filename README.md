# 🤟 SignHealth AI – Healthcare Sign Language Translator

An AI-powered Healthcare Sign Language Translator that helps deaf and speech-impaired patients communicate with doctors, nurses, and healthcare professionals through real-time hand gesture recognition.

The system uses Computer Vision and Machine Learning to detect healthcare-related hand gestures via a webcam, translate them into meaningful medical messages, and convert them into speech, improving accessibility in healthcare environments.

---

## 📌 Project Overview

Communication barriers in hospitals can make it difficult for deaf and speech-impaired patients to express their medical needs, especially during emergencies.

SignHealth AI addresses this challenge by recognizing healthcare-specific hand gestures in real time and translating them into text and speech.

The system detects hand landmarks using **MediaPipe**, classifies gestures using a **Random Forest Machine Learning model**, and generates spoken healthcare messages using **Text-to-Speech (TTS)** technology.

---

## ✨ Features

- 🖐️ Real-time hand gesture recognition
- 📷 Webcam-based gesture detection
- 🤖 Machine Learning gesture classification
- 🩺 Healthcare-specific gesture translation
- 🔊 Text-to-Speech output
- 📊 Prediction confidence score
- 📝 History logging with timestamps
- ⚡ Stable real-time predictions

---

## 🏥 Supported Healthcare Gestures

- Pain
- Help
- Water
- Medicine
- Emergency
- Head
- Chest

---

## 🛠️ Tech Stack

### Programming Language
- Python 3.11

### Computer Vision
- OpenCV
- MediaPipe

### Machine Learning
- Scikit-Learn
- Random Forest Classifier
- Joblib

### Data Processing
- NumPy
- CSV

### Text-to-Speech
- gTTS
- pyttsx3

### Development Environment
- VS Code

---

## ⚙️ How It Works

1. Capture live video through the webcam.
2. Detect hand landmarks using MediaPipe.
3. Extract 21 hand landmarks (x, y, z coordinates).
4. Generate feature vectors.
5. Predict the healthcare gesture using a trained Random Forest model.
6. Translate the gesture into a healthcare message.
7. Convert the message into speech.
8. Store prediction history with confidence score and timestamp.

---

## 📂 Project Structure

```
SignHealth-AI/
│
├── dataset/
├── model/
├── collect_data.py
├── train_model.py
├── signhealth_ai_final.py
├── gesture_model.pkl
├── history.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/meghapatel25/SignHealth-AI.git
cd SignHealth-AI
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate it

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model (Optional)

```bash
python train_model.py
```

### Run the application

```bash
python signhealth_ai_final.py
```

---

## 📈 Model Performance

- Machine Learning Algorithm: Random Forest Classifier
- Real-time Gesture Recognition
- Training Accuracy: **99.51%**

---

## 🧪 Testing

The system was tested using:

- Smoke Testing
- Integration Testing

Verified functionalities include:

- Webcam access
- Hand landmark detection
- Gesture recognition
- Speech generation
- History logging
- Real-time prediction

---

## 📸 Screenshots
<img width="500" height="500" alt="Screenshot 2026-06-21 174933" src="https://github.com/user-attachments/assets/93c569e1-f65a-4a6f-b59f-da7a8ac81550" />
<img width="500" height="500" alt="Screenshot 2026-06-21 174952" src="https://github.com/user-attachments/assets/4e521850-ddc4-4890-a6ca-a940b4685862" />
<img width="500" height="500" alt="Screenshot 2026-06-21 175005" src="https://github.com/user-attachments/assets/2257f1d6-3674-4ef4-964a-3bfd7574d435" />
<img width="500" height="500" alt="Screenshot 2026-06-21 175017" src="https://github.com/user-attachments/assets/7c4df1bd-dc05-4e29-ae11-45c64cfc471a" />
<img width="500" height="500" alt="Screenshot 2026-06-21 175028" src="https://github.com/user-attachments/assets/d836e4aa-9b19-4361-a606-f98661ed7c24" />

---

## 🔮 Future Enhancements

- Indian Sign Language (ISL) support
- LSTM-based sequence recognition
- Mobile application deployment
- Multilingual speech generation
- Hospital Information System integration

---

## 👩‍💻 Author

**Megha Subhash Patel**

MCA Student | Aspiring Software Developer

LinkedIn: www.linkedin.com/in/megha-patel-58188b326

GitHub: https://github.com/meghapatel25

---

## 📄 License

This project was developed for academic and learning purposes.
