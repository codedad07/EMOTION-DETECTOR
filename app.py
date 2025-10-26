import streamlit as st
import cv2
import numpy as np
import pickle
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Import your existing utility function
from utils import get_face_info

# --- Page Configuration ---
st.set_page_config(page_title="EMOTION DETECTOR", page_icon="😊", layout="centered")

# --- Configuration ---
MODEL_FILE = 'emotion_detection_model.pkl'
EMOTIONS = ['Angry', 'Happy', 'Sad', 'Surprise']

# --- Caching the Model (for performance) ---
@st.cache_resource
def load_model_and_scaler():
    """Load the pre-trained model and scaler only once to save time."""
    try:
        with open(MODEL_FILE, 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['scaler']
    except FileNotFoundError:
        st.error(f"Model file not found: {MODEL_FILE}. Please make sure it's in the same directory.")
        return None, None

# Load the resources
model, scaler = load_model_and_scaler()

# --- NEW: RTC Configuration for Deployment ---
# This helps with network traversal issues on different networks by providing a STUN server.
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
# -------------------------------------------

# --- Video Processing Class ---
class EmotionTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.scaler = scaler

    def transform(self, frame):
        """Processes each frame from the webcam."""
        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)
        
        face_landmarks, bbox = get_face_info(img)

        if face_landmarks and self.model is not None and bbox:
            x, y, w, h = bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            landmarks_np = np.array(face_landmarks).reshape(1, -1)
            landmarks_scaled = self.scaler.transform(landmarks_np)
            prediction = self.model.predict(landmarks_scaled)
            predicted_emotion = EMOTIONS[int(prediction[0])]

            cv2.putText(img, predicted_emotion, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
            
        return img

# --- Streamlit App Interface ---


st.title("Live Emotion Detector 😊")
st.subheader("😲 Face Emotion Detection")
st.write("This app uses your webcam to analyze facial expressions in real-time.")
st.write("Note : ")
st.write("1. Ensure your face is well-lit and clearly visible to the camera.")
st.write("2. For best results, position yourself directly in front of the camera.")
st.write("3. Minimize background distractions and noise.")
st.write("Press 'START' to begin and grant camera permissions.")



if model is None or scaler is None:
    st.warning("Model resources could not be loaded. The application cannot proceed.")
else:
    # Pass the new RTC configuration to the streamer component
    webrtc_streamer(
        key="emotion_detector",
        video_processor_factory=EmotionTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration=RTC_CONFIGURATION 
    )

st.info("The model is running. Press 'STOP' to end the stream.")

st.subheader("🎤 Voice Emotion Detection")
st.info("This feature is still under development.")
