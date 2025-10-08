import cv2
import mediapipe as mp
import numpy as np

# --- Initialize MediaPipe FaceMesh model ONCE at the start ---
mp_face_mesh = mp.solutions.face_mesh
# Set static_image_mode to False for video streams for better performance and tracking
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
# -----------------------------------------------------------

def get_face_info(image):
    """
    Extracts normalized landmarks for the model and calculates a bounding box
    for the detected face.

    Args:
        image: A NumPy array representing the image (in BGR format from OpenCV).

    Returns:
        A tuple containing:
        - A list of 1404 normalized landmark coordinates for the model.
        - A tuple (x, y, w, h) for the bounding box, or None if no face is found.
    """
    # Get image dimensions for calculating pixel coordinates
    height, width, _ = image.shape

    # MediaPipe requires RGB images, so we convert from BGR
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to find the face landmarks
    results = face_mesh.process(image_rgb)

    landmarks_list = []
    bbox = None

    if results.multi_face_landmarks:
        # Get landmarks for the first (and only) detected face
        face_landmarks = results.multi_face_landmarks[0].landmark

        # --- Calculate Bounding Box ---
        # Get all landmark coordinates in pixel space by multiplying by image dimensions
        all_x = [int(lm.x * width) for lm in face_landmarks]
        all_y = [int(lm.y * height) for lm in face_landmarks]

        # Find the min and max coordinates to define the box
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        # Add some padding to the box for better visuals
        padding = 10
        bbox = (min_x - padding, min_y - padding, (max_x - min_x) + padding*2, (max_y - min_y) + padding*2)

        # --- Normalize Landmarks for the Model (as done previously) ---
        coords = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks])
        # Normalize by subtracting the minimum to make it translation-invariant
        coords -= coords.min(axis=0)
        landmarks_list = coords.flatten().tolist()

    return landmarks_list, bbox

