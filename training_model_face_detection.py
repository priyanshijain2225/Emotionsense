import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# Constants
CSV_FILENAME = "face_landmarks_datasets.csv"
MODEL_FILENAME = "face_detection_model.h5"

# Load the dataset from CSV
print(f"Reading '{CSV_FILENAME}'...")
df = pd.read_csv(CSV_FILENAME)

# Extract features (landmarks) and create binary labels (1 for presence of face)
X = np.array([np.fromstring(x[1:-1], sep=',', dtype=float).reshape(-1, 3) for x in df['landmarks']])
y = np.ones(len(X))  # Since all samples are faces, we use 1 as the label for presence of face

# Ensure the data is in the correct shape
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)

# Define the Keras model
model = keras.models.Sequential([
    keras.layers.Input(shape=(478, 3), dtype='float32'),  # 468 is the number of face landmarks
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')  # Single output unit for binary classification
])

X_train = X_train.astype(float)
X_test = X_test.astype(float)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Save the trained model
model.save(MODEL_FILENAME)
print(f"Model saved to '{MODEL_FILENAME}'.")
