import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


DATA_DIR = "/Users/ashiqar/chore/ubiquitous/Codes/datasets/synchronized_outputs"

MODEL_PATH = "activity_recognition_cnn.keras"
SCALER_PATH = "activity_scaler.pkl"


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

LABEL_MAP = {"Walking": 0, "Running": 1, "Sitting": 2}

CLASS_NAMES = ["Walking", "Running", "Sitting"]

NUM_CLASSES = len(CLASS_NAMES)


users = sorted(
    [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
)

print("Users found:", len(users))

for user in users:
    print(user)


if len(users) != 15:
    print("\nWARNING:")
    print("Expected 15 users, but found", len(users))


rng = np.random.default_rng(42)

users = np.array(users)
rng.shuffle(users)

train_users = users[:10]
val_users = users[10:12]
test_users = users[12:15]


print("\n========================================")
print("USER SPLIT")
print("========================================")

print("\nTraining users:")
for user in train_users:
    print(" ", user)

print("\nValidation users:")
for user in val_users:
    print(" ", user)

print("\nTest users:")
for user in test_users:
    print(" ", user)


def load_user_data(user_list):

    X = []
    y = []

    for user in user_list:

        user_dir = os.path.join(DATA_DIR, user)

        csv_files = glob.glob(os.path.join(user_dir, "*.csv"))

        print(f"\nUser {user}: " f"{len(csv_files)} CSV files found")

        for file_path in csv_files:

            try:
                data = pd.read_csv(file_path)

            except Exception as e:
                print("Could not read:", file_path)
                print("Error:", e)
                continue

            missing_columns = [col for col in CHANNELS if col not in data.columns]

            if missing_columns:
                print("Skipping file because columns are missing:", file_path)
                continue

            if "label" not in data.columns:
                print("Skipping file because label column missing:", file_path)
                continue

            label_name = str(data["label"].iloc[0])

            if label_name not in LABEL_MAP:
                continue

            label = LABEL_MAP[label_name]

            sensor_data = data[CHANNELS].copy()

            sensor_data = sensor_data.dropna()

            if len(sensor_data) < WINDOW_SIZE:

                print(
                    f"Skipping {os.path.basename(file_path)} "
                    f"because it has only "
                    f"{len(sensor_data)} samples"
                )

                continue

            sensor_data = sensor_data.iloc[:WINDOW_SIZE]

            window = sensor_data.values.astype(np.float32)

            X.append(window)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    return X, y


print("\n========================================")
print("LOADING TRAINING DATA")
print("========================================")

X_train, y_train = load_user_data(train_users)

print("\nTraining data shape:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)


print("\n========================================")
print("LOADING VALIDATION DATA")
print("========================================")

X_val, y_val = load_user_data(val_users)

print("\nValidation data shape:")
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)


print("\n========================================")
print("LOADING TEST DATA")
print("========================================")

X_test, y_test = load_user_data(test_users)

print("\nTest data shape:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


if len(X_train) == 0:
    raise ValueError("Training dataset is empty!")

if len(X_val) == 0:
    raise ValueError("Validation dataset is empty!")

if len(X_test) == 0:
    raise ValueError("Test dataset is empty!")


print("\n========================================")
print("CLASS DISTRIBUTION")
print("========================================")

for i, class_name in enumerate(CLASS_NAMES):

    train_count = np.sum(y_train == i)
    val_count = np.sum(y_val == i)
    test_count = np.sum(y_test == i)

    print(
        f"{class_name:10s} | "
        f"Train: {train_count:4d} | "
        f"Val: {val_count:4d} | "
        f"Test: {test_count:4d}"
    )


print("\n========================================")
print("NORMALIZATION")
print("========================================")


X_train_2d = X_train.reshape(-1, len(CHANNELS))


scaler = StandardScaler()


scaler.fit(X_train_2d)


def normalize_data(X):

    original_shape = X.shape

    X = X.reshape(-1, len(CHANNELS))

    X = scaler.transform(X)

    X = X.reshape(original_shape)

    return X.astype(np.float32)


X_train = normalize_data(X_train)

X_val = normalize_data(X_val)

X_test = normalize_data(X_test)


with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

print("Scaler saved to:", SCALER_PATH)


print("\n========================================")
print("FINAL DATA SHAPES")
print("========================================")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


model = models.Sequential(
    [
        layers.Input(shape=(WINDOW_SIZE, len(CHANNELS))),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)


model_checkpoint = ModelCheckpoint(
    "activity_cnn_best.keras", monitor="val_loss", save_best_only=True, verbose=1
)

print("\n========================================")
print("TRAINING")
print("========================================")


history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1,
)


model.save(MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)


print("\n========================================")
print("TESTING")
print("========================================")


test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)


print("\nTest Loss:")
print(test_loss)

print("\nTest Accuracy:")
print(test_accuracy)


y_probability = model.predict(X_test)

y_pred = np.argmax(y_probability, axis=1)


accuracy = accuracy_score(y_test, y_pred)

print("\n========================================")
print("FINAL ACCURACY")
print("========================================")

print(f"Test Accuracy: {accuracy * 100:.2f}%")


print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))


print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

cm = confusion_matrix(y_test, y_pred)

print(cm)


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)

disp.plot()

plt.title("Activity Recognition Confusion Matrix")

plt.tight_layout()

plt.show()


plt.figure()

plt.plot(history.history["accuracy"], label="Training Accuracy")

plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Training and Validation Accuracy")

plt.legend()

plt.grid()

plt.show()


plt.figure()

plt.plot(history.history["loss"], label="Training Loss")

plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training and Validation Loss")

plt.legend()

plt.grid()

plt.show()


print("\n========================================")
print("SAMPLE PREDICTIONS")
print("========================================")

number_to_show = min(20, len(y_test))

for i in range(number_to_show):

    actual = CLASS_NAMES[y_test[i]]

    predicted = CLASS_NAMES[y_pred[i]]

    confidence = np.max(y_probability[i]) * 100

    print(
        f"{i+1:2d}. "
        f"Actual: {actual:10s} | "
        f"Predicted: {predicted:10s} | "
        f"Confidence: {confidence:.2f}%"
    )


print("\n========================================")
print("DONE")
print("========================================")

print("Model:")
print(MODEL_PATH)

print("\nScaler:")
print(SCALER_PATH)
