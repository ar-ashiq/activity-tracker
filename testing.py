import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

MODEL_PATH = "activity_recognition_cnn.keras"
SCALER_PATH = "activity_scaler.pkl"
NEW_DATA_PATH = "new_data.csv"

WINDOW_SIZE = 100

CHANNELS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "acc_magnitude",
    "gyro_magnitude",
    "acc_magnitude_change",
    "gyro_magnitude_change",
]

CLASS_NAMES = ["Walking", "Running", "Sitting"]

model = tf.keras.models.load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

data = pd.read_csv(NEW_DATA_PATH)

missing = [col for col in CHANNELS if col not in data.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

sensor_data = data[CHANNELS].dropna()

num_windows = len(sensor_data) // WINDOW_SIZE

if num_windows == 0:
    raise ValueError("Not enough samples for one window.")

windows = []

for i in range(num_windows):
    start = i * WINDOW_SIZE
    end = start + WINDOW_SIZE
    windows.append(sensor_data.iloc[start:end].values)

X_new = np.array(windows, dtype=np.float32)

print("New data shape:", X_new.shape)

shape = X_new.shape

X_new = X_new.reshape(-1, len(CHANNELS))
X_new = scaler.transform(X_new)
X_new = X_new.reshape(shape).astype(np.float32)

probabilities = model.predict(X_new, verbose=1)
predictions = np.argmax(probabilities, axis=1)

print("\nPredictions:")

for i, prediction in enumerate(predictions):
    activity = CLASS_NAMES[prediction]
    confidence = probabilities[i][prediction] * 100

    print(f"Window {i + 1}: " f"{activity} " f"({confidence:.2f}%)")

results = pd.DataFrame(
    {
        "window": np.arange(1, len(predictions) + 1),
        "predicted_activity": [CLASS_NAMES[p] for p in predictions],
        "confidence": [
            probabilities[i][predictions[i]] for i in range(len(predictions))
        ],
    }
)

results.to_csv("new_data_predictions.csv", index=False)

print("\nResults saved to new_data_predictions.csv")
