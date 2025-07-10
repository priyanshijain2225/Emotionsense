import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from tensorflow import keras

# Constants
IMAGE_WIDTH, IMAGE_HEIGHT = 1280, 720

# Load the saved Keras model
MODEL_FILENAME = "face_detection_model.h5"
loaded_model = keras.models.load_model(MODEL_FILENAME)

# Function to extract face landmarks using Mediapipe
def extract_face_landmarks(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe Face Mesh
    results = face_mesh.process(rgb_image)
    landmarks = []
    bbox = None

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for point in face_landmarks.landmark:
                landmarks.extend([point.x, point.y, point.z])
            # Calculate bounding box
            x_coords = [point.x for point in face_landmarks.landmark]
            y_coords = [point.y for point in face_landmarks.landmark]
            bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

    face_mesh.close()
    return landmarks, bbox

# Initialize OpenCV VideoCapture with the video file path
video_file_path = "C:\First\Face_detection_images\WIN_20240617_10_28_22_Pro.mp4"
cap = cv2.VideoCapture(video_file_path)

while True:
    ret, frame = cap.read()
    if not ret:
        prnt("Failed to capture frame or end of video reached.")
        break

    # Resize the frame
    frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # Extract face landmarks and bounding box
    landmarks, bbox = extract_face_landmarks(frame)
    if not landmarks:
        print("No face landmarks detected.")
        continue

    # Process landmarks for prediction
    landmarks = np.array([landmarks])
    landmarks = landmarks.reshape(landmarks.shape[0], 478, 3)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(landmarks)

    # Get the predicted class (presence of face)
    predicted_class = np.argmax(predictions)
    if bbox:
        x_min, y_min, x_max, y_max = bbox
        x_min = int(x_min * IMAGE_WIDTH)
        y_min = int(y_min * IMAGE_HEIGHT)
        x_max = int(x_max * IMAGE_WIDTH)
        y_max = int(y_max * IMAGE_HEIGHT)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Use green for face detection
    else:
        # You can display a message here indicating no face detected
        cv2.putText(frame, 'No Face Detected', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Face Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
