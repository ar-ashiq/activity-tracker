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
    balanced_accuracy_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from paths import (
    RAW_SYNC_DIR,
    SYNCHRONIZED_DIR,
    RAW_LSTM_BEST_MODEL_PATH,
    RAW_LSTM_METRICS_DIR,
    RAW_LSTM_MODEL_PATH,
    RAW_LSTM_PLOT_DIR,
    RAW_LSTM_SCALER_PATH,
)

import json

SPLIT_PATH = "split.json"  # path to your JSON file

with open(SPLIT_PATH, "r") as f:
    split = json.load(f)

train_users = split["train"]
val_users = split["validation"]
test_users = split["test"]


# ============================================================
# PATHS
# ============================================================

DATA_DIR = RAW_SYNC_DIR

MODEL_PATH = RAW_LSTM_MODEL_PATH
BEST_MODEL_PATH = RAW_LSTM_BEST_MODEL_PATH
SCALER_PATH = RAW_LSTM_SCALER_PATH

MODEL_DIR = MODEL_PATH.parent
SCALER_DIR = SCALER_PATH.parent
PLOT_DIR = RAW_LSTM_PLOT_DIR
METRICS_DIR = RAW_LSTM_METRICS_DIR

for output_dir in (
    MODEL_DIR,
    SCALER_DIR,
    PLOT_DIR,
    METRICS_DIR,
):
    output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

WINDOW_SIZE = 100

CLASS_NAMES = [
    "Walking",
    "Running",
    "Sitting",
    "Bicycling",
    "Lying",
    "Standing",
]

LABEL_MAP = {
    "Walking": 0,
    "Running": 1,
    "Sitting": 2,
    "Bicycling": 3,
    "Lying": 4,
    "Standing": 5,
}

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# RAW SENSOR FEATURES
# ============================================================

CHANNELS = [
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "acc_magnitude",
    "gyro_magnitude",
    "acc_magnitude_change",
    "gyro_magnitude_change",
]

NUM_CHANNELS = len(CHANNELS)


# ============================================================
# FIND USERS
# ============================================================

users = sorted(
    [
        d
        for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ]
)

print("Users found:", len(users))

for user in users:
    print(user)

if len(users) != 15:
    print("\nWARNING:")
    print("Expected 15 users, but found", len(users))

print("\n========================================")
print("USING FIXED USER SPLIT")
print("========================================")

print("Train:", len(train_users))
print("Val:", len(val_users))
print("Test:", len(test_users))


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


# ============================================================
# LOAD DATA
# ============================================================

def load_user_data(user_list):

    X = []
    y = []

    for user in user_list:

        user_dir = os.path.join(DATA_DIR, user)

        csv_files = glob.glob(
            os.path.join(user_dir, "**", "synchronized_25hz.csv"),
            recursive=True
        )

        print(
            f"\nUser {user}: "
            f"{len(csv_files)} CSV files found"
        )

        for file_path in csv_files:

            try:
                data = pd.read_csv(file_path)

            except Exception as e:
                print("Could not read:", file_path)
                print("Error:", e)
                continue

            # ------------------------------------------------
            # CHECK REQUIRED FEATURES
            # ------------------------------------------------

            missing_columns = [
                col
                for col in CHANNELS
                if col not in data.columns
            ]

            if missing_columns:
                print(
                    "Missing sequence columns:",
                    missing_columns,
                )
                continue

            if "label" not in data.columns:
                continue

            # ------------------------------------------------
            # GET LABEL
            # ------------------------------------------------

            label_name = str(
                data["label"].iloc[0]
            )

            if label_name not in LABEL_MAP:
                continue

            label = LABEL_MAP[label_name]

            # ------------------------------------------------
            # SENSOR DATA
            # ------------------------------------------------

            sensor_data = data[CHANNELS].copy()

            sensor_data = sensor_data.apply(
                pd.to_numeric,
                errors="coerce",
            )

            sensor_data = sensor_data.dropna()

            sensor_values = sensor_data.values.astype(
                np.float32
            )

            # ------------------------------------------------
            # CREATE 100-SAMPLE WINDOWS
            # ------------------------------------------------

            number_of_windows = (
                len(sensor_values) // WINDOW_SIZE
            )

            if number_of_windows == 0:
                continue

            for window_id in range(number_of_windows):

                start = (
                    window_id * WINDOW_SIZE
                )

                end = start + WINDOW_SIZE

                window = sensor_values[
                    start:end
                ]

                if len(window) != WINDOW_SIZE:
                    continue

                X.append(window)
                y.append(label)

    X = np.array(
        X,
        dtype=np.float32,
    )

    y = np.array(
        y,
        dtype=np.int32,
    )

    return X, y


# ============================================================
# TRAINING DATA
# ============================================================

print("\n========================================")
print("LOADING TRAINING DATA")
print("========================================")

X_train, y_train = load_user_data(
    train_users
)

print("\nTraining shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)


# ============================================================
# VALIDATION DATA
# ============================================================

print("\n========================================")
print("LOADING VALIDATION DATA")
print("========================================")

X_val, y_val = load_user_data(
    val_users
)

print("\nValidation shapes:")
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)


# ============================================================
# TEST DATA
# ============================================================

print("\n========================================")
print("LOADING TEST DATA")
print("========================================")

X_test, y_test = load_user_data(
    test_users
)

print("\nTest shapes:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# CHECK DATA
# ============================================================

if len(X_train) == 0:
    raise ValueError(
        "Training dataset is empty!"
    )

if len(X_val) == 0:
    raise ValueError(
        "Validation dataset is empty!"
    )

if len(X_test) == 0:
    raise ValueError(
        "Test dataset is empty!"
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n========================================")
print("CLASS DISTRIBUTION")
print("========================================")

for i, class_name in enumerate(CLASS_NAMES):

    train_count = np.sum(y_train == i)
    val_count = np.sum(y_val == i)
    test_count = np.sum(y_test == i)

    print(
        f"{class_name:10s} | "
        f"Train: {train_count:5d} | "
        f"Val: {val_count:5d} | "
        f"Test: {test_count:5d}"
    )


# ============================================================
# NORMALIZE SENSOR FEATURES
# ============================================================

print("\n========================================")
print("NORMALIZING SENSOR FEATURES")
print("========================================")

X_train_2d = X_train.reshape(
    -1,
    NUM_CHANNELS,
)

scaler = StandardScaler()

scaler.fit(X_train_2d)


def normalize_sequence(X):

    original_shape = X.shape

    X = X.reshape(
        -1,
        NUM_CHANNELS,
    )

    X = scaler.transform(X)

    X = X.reshape(
        original_shape
    )

    return X.astype(np.float32)


X_train = normalize_sequence(X_train)
X_val = normalize_sequence(X_val)
X_test = normalize_sequence(X_test)


with open(
    SCALER_PATH,
    "wb",
) as file:

    pickle.dump(
        scaler,
        file,
    )


print(
    "Scaler saved to:",
    SCALER_PATH,
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train,
)

class_weights = dict(
    zip(classes, weights)
)


print("\n========================================")
print("CLASS WEIGHTS")
print("========================================")

for class_id, weight in class_weights.items():

    print(
        CLASS_NAMES[class_id],
        ":",
        round(weight, 4),
    )


# ============================================================
# BUILD RAW-INPUT LSTM MODEL
# ============================================================

print("\n========================================")
print("BUILDING RAW-INPUT LSTM MODEL")
print("========================================")

sequence_input = layers.Input(
    shape=(
        WINDOW_SIZE,
        NUM_CHANNELS,
    ),
    name="sensor_sequence",
)

x = layers.LSTM(
    64,
    return_sequences=True,
)(sequence_input)

x = layers.Dropout(0.3)(x)

x = layers.LSTM(32)(x)

x = layers.Dropout(0.3)(x)

x = layers.Dense(
    32,
    activation="relu",
)(x)

x = layers.Dropout(0.3)(x)

output = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="activity",
)(x)

model = Model(
    inputs=sequence_input,
    outputs=output,
)

model.summary()


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)

model_checkpoint = ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)


# ============================================================
# TRAIN
# ============================================================

print("\n========================================")
print("TRAINING")
print("========================================")

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_val,
        y_val,
    ),
    epochs=50,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[
        early_stopping,
        model_checkpoint,
    ],
    verbose=1,
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(MODEL_PATH)

print(
    "\nModel saved to:",
    MODEL_PATH,
)


# ============================================================
# TEST
# ============================================================

print("\n========================================")
print("TESTING")
print("========================================")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1,
)

print("\nTest Loss:")
print(test_loss)

print("\nTest Accuracy:")
print(test_accuracy)


# ============================================================
# PREDICTIONS
# ============================================================

y_probability = model.predict(
    X_test,
    verbose=1,
)

y_pred = np.argmax(
    y_probability,
    axis=1,
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred,
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred,
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
)


print("\n========================================")
print("FINAL RESULTS")
print("========================================")

print(
    f"Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Balanced Accuracy: "
    f"{balanced_accuracy * 100:.2f}%"
)

print(
    f"Macro F1: "
    f"{macro_f1:.4f}"
)


# ============================================================
# SAVE METRICS
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
)

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(report)

with open(
    METRICS_DIR.joinpath(
        "classification_report.txt"
    ),
    "w",
) as file:

    file.write(report)


with open(
    METRICS_DIR.joinpath(
        "metrics.txt"
    ),
    "w",
) as file:

    file.write(
        f"test_loss={test_loss}\n"
    )

    file.write(
        f"test_accuracy={test_accuracy}\n"
    )

    file.write(
        f"accuracy={accuracy}\n"
    )

    file.write(
        f"balanced_accuracy="
        f"{balanced_accuracy}\n"
    )

    file.write(
        f"macro_f1={macro_f1}\n"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(NUM_CLASSES)
)

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES,
)

disp.plot()

plt.title(
    "Raw-Input LSTM Activity Recognition"
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR.joinpath(
        "confusion_matrix.png"
    )
)

plt.show()


# ============================================================
# ACCURACY PLOT
# ============================================================

plt.figure()

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy",
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy",
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Raw-Input LSTM Training and Validation Accuracy"
)

plt.legend()
plt.grid()

plt.savefig(
    PLOT_DIR.joinpath(
        "accuracy.png"
    )
)

plt.show()


# ============================================================
# LOSS PLOT
# ============================================================

plt.figure()

plt.plot(
    history.history["loss"],
    label="Training Loss",
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Raw-Input LSTM Training and Validation Loss"
)

plt.legend()
plt.grid()

plt.savefig(
    PLOT_DIR.joinpath(
        "loss.png"
    )
)

plt.show()


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n========================================")
print("SAMPLE PREDICTIONS")
print("========================================")

number_to_show = min(
    20,
    len(y_test),
)

for i in range(number_to_show):

    actual = CLASS_NAMES[
        y_test[i]
    ]

    predicted = CLASS_NAMES[
        y_pred[i]
    ]

    confidence = (
        np.max(y_probability[i])
        * 100
    )

    print(
        f"{i + 1:2d}. "
        f"Actual: {actual:10s} | "
        f"Predicted: {predicted:10s} | "
        f"Confidence: {confidence:.2f}%"
    )


# ============================================================
# DONE
# ============================================================

print("\n========================================")
print("DONE")
print("========================================")

print(
    "Model:",
    MODEL_PATH,
)

print(
    "Scaler:",
    SCALER_PATH,
)