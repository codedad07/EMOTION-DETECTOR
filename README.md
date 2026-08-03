# 😊 Emotion Detector

An AI-powered Emotion Detection web application that predicts human emotions from facial expressions using Computer Vision and Machine Learning.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-ML%20Model-yellow)

---

## 📖 Overview

Emotion Detector is a Machine Learning application that analyzes a person's facial expressions and predicts their emotion in real time.

The application utilizes **MediaPipe Face Mesh** to extract facial landmarks, **OpenCV** for image processing, and an **XGBoost classifier** to recognize different emotions through an interactive **Streamlit** web interface.

---

## ✨ Features

- 🎥 Real-time emotion detection
- 😀 Facial landmark extraction using MediaPipe
- 📷 Image processing with OpenCV
- 🤖 Emotion classification using XGBoost
- 🌐 Interactive Streamlit web interface
- ⚡ Fast and lightweight prediction

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Machine Learning | XGBoost, Scikit-learn |
| Computer Vision | OpenCV, MediaPipe |
| Web Framework | Streamlit |
| Libraries | NumPy |

---

## 📂 Project Structure

```
EMOTION-DETECTOR/
│
├── app.py
├── utils.py
├── emotion_detection_model.pkl
├── requirements.txt
├── packages.txt
└── README.md
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/dattarajdev/EMOTION-DETECTOR.git
```

### Navigate to the project

```bash
cd EMOTION-DETECTOR
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the environment

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 🔍 How It Works

1. Capture or upload a facial image.
2. OpenCV processes the image.
3. MediaPipe Face Mesh detects facial landmarks.
4. Landmark coordinates are extracted as features.
5. The trained XGBoost model predicts the emotion.
6. The predicted emotion is displayed on the Streamlit interface.

---

## 😊 Supported Emotions

- Happy
- Sad
- Angry
- Surprise
- Fear
- Neutral

*(Depending on the trained model.)*

---

## 📸 Screenshots

### Home Page

<img width="1917" height="932" alt="image" src="https://github.com/user-attachments/assets/9a0d3a55-a179-4e84-b232-3634c944a2c6" />


### Emotion Prediction

<img width="1072" height="475" alt="image" src="https://github.com/user-attachments/assets/931acc59-d244-45f7-9289-5499e9337ffc" />


---

## 🚀 Future Improvements

- Support multiple face detection
- Improve prediction accuracy
- Deploy the application online
- Add emotion analytics dashboard
- Support video file emotion detection
- Integrate speech emotion recognition

---

## 👨‍💻 Author

**Dattaraj Rane**

- GitHub: https://github.com/dattarajdev

---

## ⭐ If you found this project helpful

Give this repository a ⭐ on GitHub.
