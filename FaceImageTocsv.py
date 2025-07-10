import cv2
import mediapipe as mp
import os
import pandas as pd

# Constants
IMAGE_WIDTH, IMAGE_HEIGHT = 1280, 720  # Adjust based on your requirements
DATA_DIR = "C:\First\Face_detection_images"
CSV_FILENAME = "face_landmarks_datasets.csv"
print("Mediapipe version:", mp.__version__)
# Function to extract face landmarks using Mediapipe
def extract_face_landmarks(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe Face Mesh
    results = face_mesh.process(rgb_image)

    landmarks = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for point in face_landmarks.landmark:
                landmarks.extend([point.x, point.y, point.z])

    face_mesh.close()
    return landmarks

# Load images, extract face landmarks, and save data to CSV
data = {"landmarks": [], "label": []}  # 'label' will be set to 1 for presence of a face

for image_name in os.listdir(DATA_DIR):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(DATA_DIR, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to read image '{image_path}'. Skipping.")
            continue
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

        landmarks = extract_face_landmarks(image)
        print(len(landmarks))
        if landmarks:  # Check if landmarks are found
            data["landmarks"].append(landmarks)
            data["label"].append(1)  # Presence of face

# Convert the data dictionary to a Pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.dropna(inplace=True)
df.to_csv(CSV_FILENAME, index=False)
print(f"Face landmarks data saved to '{CSV_FILENAME}'.")
