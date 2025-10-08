import streamlit as st
import cv2
import numpy as np
import pickle
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from utils import get_face_info # Use the function that provides the bounding box

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

# --- Video Processing Class ---
class EmotionTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.scaler = scaler

    def transform(self, frame):
        """Processes each frame from the webcam."""
        img = frame.to_ndarray(format="bgr24")
        
        face_landmarks, bbox = get_face_info(img)

        if face_landmarks and self.model is not None:
            # Draw bounding box
            x, y, w, h = bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Predict emotion
            landmarks_np = np.array(face_landmarks).reshape(1, -1)
            landmarks_scaled = self.scaler.transform(landmarks_np)
            prediction = self.model.predict(landmarks_scaled)
            predicted_emotion = EMOTIONS[int(prediction[0])]

            # Display emotion text
            cv2.putText(img, predicted_emotion, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            # Display instructional message
            line1 = "Please position your face"
            line2 = "in front of the camera"
            (text_width, text_height), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            center_x = (img.shape[1] - text_width) // 2
            center_y = img.shape[0] // 2
            cv2.putText(img, line1, (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, line2, (center_x - 20, center_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        return img

# --- Streamlit App Interface ---
st.title("Live Emotion Detector 😊")
st.write("This app uses your webcam to analyze facial expressions in real-time.")
st.write("Press 'START' to begin and grant camera permissions.")

if model is None or scaler is None:
    st.warning("Model resources could not be loaded. The application cannot proceed.")
else:
    # The main component that handles the webcam stream
    webrtc_streamer(
        key="emotion_detector",
        video_processor_factory=EmotionTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.info("The model is running. Press 'STOP' to end the stream.")

# --- NEW FEATURE PLACEHOLDER ---
st.subheader("🎤 Voice Emotion Detection")
st.info("This feature is still under development.")
# -----------------------------

