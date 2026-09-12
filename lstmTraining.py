import os
import glob
import pickle
from pathlib import Path
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
    LABEL_DIR,
    LSTM_BEST_MODEL_PATH,
    LSTM_METRICS_DIR,
    LSTM_MODEL_PATH,
    LSTM_PLOT_DIR,
    LSTM_SEQUENCE_SCALER_PATH,
    LSTM_STAT_SCALER_PATH,
    SYNCHRONIZED_DIR,
)

DATA_DIR = SYNCHRONIZED_DIR
MODEL_PATH = LSTM_MODEL_PATH
SEQUENCE_SCALER_PATH = LSTM_SEQUENCE_SCALER_PATH
STAT_SCALER_PATH = LSTM_STAT_SCALER_PATH

for output_dir in (
    LSTM_MODEL_PATH.parent,
    LSTM_SEQUENCE_SCALER_PATH.parent,
    LSTM_PLOT_DIR,
    LSTM_METRICS_DIR,
):
    output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

WINDOW_SIZE = 100

CLASS_NAMES = ["Walking", "Running", "Sitting"]

LABEL_MAP = {"Walking": 0, "Running": 1, "Sitting": 2}

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# RAW SEQUENCE FEATURES
# ============================================================

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


# ============================================================
# FEATURES FROM features_labels.csv
# ============================================================

STAT_FEATURES = [
    # Accelerometer magnitude statistics
    "raw_acc:magnitude_stats:mean",
    "raw_acc:magnitude_stats:std",
    "raw_acc:magnitude_stats:moment3",
    "raw_acc:magnitude_stats:moment4",
    "raw_acc:magnitude_stats:percentile25",
    "raw_acc:magnitude_stats:percentile50",
    "raw_acc:magnitude_stats:percentile75",
    "raw_acc:magnitude_stats:value_entropy",
    "raw_acc:magnitude_stats:time_entropy",
    # Accelerometer frequency features
    "raw_acc:magnitude_spectrum:log_energy_band0",
    "raw_acc:magnitude_spectrum:log_energy_band1",
    "raw_acc:magnitude_spectrum:log_energy_band2",
    "raw_acc:magnitude_spectrum:log_energy_band3",
    "raw_acc:magnitude_spectrum:log_energy_band4",
    "raw_acc:magnitude_spectrum:spectral_entropy",
    # Accelerometer autocorrelation
    "raw_acc:magnitude_autocorrelation:period",
    "raw_acc:magnitude_autocorrelation:normalized_ac",
    # Accelerometer 3D
    "raw_acc:3d:mean_x",
    "raw_acc:3d:mean_y",
    "raw_acc:3d:mean_z",
    "raw_acc:3d:std_x",
    "raw_acc:3d:std_y",
    "raw_acc:3d:std_z",
    "raw_acc:3d:ro_xy",
    "raw_acc:3d:ro_xz",
    "raw_acc:3d:ro_yz",
    # Gyroscope magnitude statistics
    "proc_gyro:magnitude_stats:mean",
    "proc_gyro:magnitude_stats:std",
    "proc_gyro:magnitude_stats:moment3",
    "proc_gyro:magnitude_stats:moment4",
    "proc_gyro:magnitude_stats:percentile25",
    "proc_gyro:magnitude_stats:percentile50",
    "proc_gyro:magnitude_stats:percentile75",
    "proc_gyro:magnitude_stats:value_entropy",
    "proc_gyro:magnitude_stats:time_entropy",
    # Gyroscope frequency features
    "proc_gyro:magnitude_spectrum:log_energy_band0",
    "proc_gyro:magnitude_spectrum:log_energy_band1",
    "proc_gyro:magnitude_spectrum:log_energy_band2",
    "proc_gyro:magnitude_spectrum:log_energy_band3",
    "proc_gyro:magnitude_spectrum:log_energy_band4",
    "proc_gyro:magnitude_spectrum:spectral_entropy",
    # Gyroscope autocorrelation
    "proc_gyro:magnitude_autocorrelation:period",
    "proc_gyro:magnitude_autocorrelation:normalized_ac",
    # Gyroscope 3D
    "proc_gyro:3d:mean_x",
    "proc_gyro:3d:mean_y",
    "proc_gyro:3d:mean_z",
    "proc_gyro:3d:std_x",
    "proc_gyro:3d:std_y",
    "proc_gyro:3d:std_z",
    "proc_gyro:3d:ro_xy",
    "proc_gyro:3d:ro_xz",
    "proc_gyro:3d:ro_yz",
]


# ============================================================
# FIND USERS
# ============================================================

users = sorted(
    [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
)

print("Users found:", len(users))

for user in users:
    print(user)


if len(users) != 15:
    print("\nWARNING:")
    print("Expected 15 users, but found", len(users))


# ============================================================
# USER SPLIT
# ============================================================

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


# ============================================================
# LOAD STATISTICAL FEATURES
# ============================================================


def load_stat_features(user_id, timestamp_id):

    label_file = os.path.join(LABEL_DIR, f"{user_id}.features_labels.csv")

    if not os.path.exists(label_file):
        return None

    try:
        labels = pd.read_csv(label_file)
    except Exception:
        return None

    if "timestamp" not in labels.columns:
        return None

    labels["timestamp"] = pd.to_numeric(labels["timestamp"], errors="coerce")

    labels = labels.dropna(subset=["timestamp"]).reset_index(drop=True)

    if len(labels) == 0:
        return None

    timestamp_value = float(timestamp_id)

    difference = np.abs(labels["timestamp"] - timestamp_value)

    closest_index = difference.idxmin()

    row = labels.loc[closest_index]

    values = []

    for feature in STAT_FEATURES:

        if feature not in labels.columns:
            return None

        value = pd.to_numeric(row[feature], errors="coerce")

        values.append(value)

    values = np.array(values, dtype=np.float32)

    if np.any(~np.isfinite(values)):
        return None

    return values


# ============================================================
# LOAD DATA
# ============================================================


def load_user_data(user_list):

    X_sequence = []

    X_statistics = []

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

            # ------------------------------------------------
            # CHECK RAW FEATURES
            # ------------------------------------------------

            missing_columns = [col for col in CHANNELS if col not in data.columns]

            if missing_columns:

                print("Missing sequence columns:", missing_columns)

                continue

            if "label" not in data.columns:

                continue

            label_name = str(data["label"].iloc[0])

            if label_name not in LABEL_MAP:

                continue

            label = LABEL_MAP[label_name]

            # ------------------------------------------------
            # GET TIMESTAMP ID
            # ------------------------------------------------

            filename = os.path.basename(file_path)

            timestamp_id = filename.split("_")[0]

            # ------------------------------------------------
            # LOAD STATISTICAL FEATURES
            # ------------------------------------------------

            stat_features = load_stat_features(user, timestamp_id)

            if stat_features is None:

                print(
                    "Skipping because statistical " "features are unavailable:",
                    filename,
                )

                continue

            # ------------------------------------------------
            # RAW SENSOR DATA
            # ------------------------------------------------

            sensor_data = data[CHANNELS].copy()

            sensor_data = sensor_data.apply(pd.to_numeric, errors="coerce")

            sensor_data = sensor_data.dropna()

            sensor_values = sensor_data.values.astype(np.float32)

            # ------------------------------------------------
            # CREATE 100-SAMPLE WINDOWS
            # ------------------------------------------------

            number_of_windows = len(sensor_values) // WINDOW_SIZE

            if number_of_windows == 0:

                continue

            for window_id in range(number_of_windows):

                start = window_id * WINDOW_SIZE

                end = start + WINDOW_SIZE

                window = sensor_values[start:end]

                if len(window) != WINDOW_SIZE:

                    continue

                X_sequence.append(window)

                X_statistics.append(stat_features)

                y.append(label)

    X_sequence = np.array(X_sequence, dtype=np.float32)

    X_statistics = np.array(X_statistics, dtype=np.float32)

    y = np.array(y, dtype=np.int32)

    return (X_sequence, X_statistics, y)


# ============================================================
# TRAINING DATA
# ============================================================

print("\n========================================")
print("LOADING TRAINING DATA")
print("========================================")

X_train_seq, X_train_stat, y_train = load_user_data(train_users)

print("\nTraining shapes:")

print("X_train_seq:", X_train_seq.shape)

print("X_train_stat:", X_train_stat.shape)

print("y_train:", y_train.shape)


# ============================================================
# VALIDATION DATA
# ============================================================

print("\n========================================")
print("LOADING VALIDATION DATA")
print("========================================")

X_val_seq, X_val_stat, y_val = load_user_data(val_users)

print("\nValidation shapes:")

print("X_val_seq:", X_val_seq.shape)

print("X_val_stat:", X_val_stat.shape)

print("y_val:", y_val.shape)


# ============================================================
# TEST DATA
# ============================================================

print("\n========================================")
print("LOADING TEST DATA")
print("========================================")

X_test_seq, X_test_stat, y_test = load_user_data(test_users)

print("\nTest shapes:")

print("X_test_seq:", X_test_seq.shape)

print("X_test_stat:", X_test_stat.shape)

print("y_test:", y_test.shape)


# ============================================================
# CHECK DATA
# ============================================================

if len(X_train_seq) == 0:
    raise ValueError("Training dataset is empty!")

if len(X_val_seq) == 0:
    raise ValueError("Validation dataset is empty!")

if len(X_test_seq) == 0:
    raise ValueError("Test dataset is empty!")


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
# NORMALIZE RAW SEQUENCE FEATURES
# ============================================================

print("\n========================================")
print("NORMALIZING SEQUENCE FEATURES")
print("========================================")


num_channels = len(CHANNELS)

X_train_2d = X_train_seq.reshape(-1, num_channels)


sequence_scaler = StandardScaler()

sequence_scaler.fit(X_train_2d)


def normalize_sequence(X):

    original_shape = X.shape

    X = X.reshape(-1, num_channels)

    X = sequence_scaler.transform(X)

    X = X.reshape(original_shape)

    return X.astype(np.float32)


X_train_seq = normalize_sequence(X_train_seq)

X_val_seq = normalize_sequence(X_val_seq)

X_test_seq = normalize_sequence(X_test_seq)


with open(SEQUENCE_SCALER_PATH, "wb") as f:

    pickle.dump(sequence_scaler, f)


# ============================================================
# NORMALIZE STATISTICAL FEATURES
# ============================================================

print("\n========================================")
print("NORMALIZING STATISTICAL FEATURES")
print("========================================")


stat_scaler = StandardScaler()

stat_scaler.fit(X_train_stat)


X_train_stat = stat_scaler.transform(X_train_stat).astype(np.float32)


X_val_stat = stat_scaler.transform(X_val_stat).astype(np.float32)


X_test_stat = stat_scaler.transform(X_test_stat).astype(np.float32)


with open(STAT_SCALER_PATH, "wb") as f:

    pickle.dump(stat_scaler, f)


print("Sequence scaler saved to:", SEQUENCE_SCALER_PATH)

print("Statistics scaler saved to:", STAT_SCALER_PATH)


# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = np.unique(y_train)

weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)

class_weights = dict(zip(classes, weights))


print("\n========================================")
print("CLASS WEIGHTS")
print("========================================")

for class_id, weight in class_weights.items():

    print(CLASS_NAMES[class_id], ":", round(weight, 4))


# ============================================================
# BUILD TWO-INPUT LSTM MODEL
# ============================================================

print("\n========================================")
print("BUILDING LSTM MODEL")
print("========================================")


# Raw sensor sequence input
sequence_input = layers.Input(shape=(WINDOW_SIZE, num_channels), name="sensor_sequence")


x = layers.LSTM(64, return_sequences=True)(sequence_input)

x = layers.Dropout(0.3)(x)

x = layers.LSTM(32)(x)

x = layers.Dropout(0.3)(x)


# Statistical feature input
statistics_input = layers.Input(
    shape=(len(STAT_FEATURES),), name="statistical_features"
)


s = layers.Dense(32, activation="relu")(statistics_input)

s = layers.Dropout(0.2)(s)


# Combine both branches
combined = layers.Concatenate()([x, s])


combined = layers.Dense(32, activation="relu")(combined)

combined = layers.Dropout(0.3)(combined)


output = layers.Dense(NUM_CLASSES, activation="softmax", name="activity")(combined)


model = Model(inputs=[sequence_input, statistics_input], outputs=output)


model.summary()


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)


model_checkpoint = ModelCheckpoint(
    LSTM_BEST_MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\n========================================")
print("TRAINING")
print("========================================")


history = model.fit(
    {"sensor_sequence": X_train_seq, "statistical_features": X_train_stat},
    y_train,
    validation_data=(
        {"sensor_sequence": X_val_seq, "statistical_features": X_val_stat},
        y_val,
    ),
    epochs=50,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1,
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)


# ============================================================
# TEST
# ============================================================

print("\n========================================")
print("TESTING")
print("========================================")


test_loss, test_accuracy = model.evaluate(
    {"sensor_sequence": X_test_seq, "statistical_features": X_test_stat},
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
    {"sensor_sequence": X_test_seq, "statistical_features": X_test_stat}, verbose=1
)


y_pred = np.argmax(y_probability, axis=1)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

macro_f1 = f1_score(y_test, y_pred, average="macro")


print("\n========================================")
print("FINAL RESULTS")
print("========================================")

print(f"Test Accuracy: " f"{accuracy * 100:.2f}%")

print(f"Balanced Accuracy: " f"{balanced_accuracy * 100:.2f}%")

print(f"Macro F1: " f"{macro_f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
print(report)
with open(LSTM_METRICS_DIR.joinpath("classification_report.txt"), "w") as file:
    file.write(report)
with open(LSTM_METRICS_DIR.joinpath("metrics.txt"), "w") as file:
    file.write(f"test_loss={test_loss}\n")
    file.write(f"test_accuracy={test_accuracy}\n")
    file.write(f"accuracy={accuracy}\n")


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

cm = confusion_matrix(y_test, y_pred)

print(cm)


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)

disp.plot()

plt.title("Activity Recognition Confusion Matrix")

plt.tight_layout()

plt.savefig(LSTM_PLOT_DIR.joinpath("confusion_matrix.png"))

plt.show()


# ============================================================
# ACCURACY PLOT
# ============================================================

plt.figure()

plt.plot(history.history["accuracy"], label="Training Accuracy")

plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Training and Validation Accuracy")

plt.legend()

plt.grid()

plt.savefig(LSTM_PLOT_DIR.joinpath("accuracy.png"))

plt.show()


# ============================================================
# LOSS PLOT
# ============================================================

plt.figure()

plt.plot(history.history["loss"], label="Training Loss")

plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training and Validation Loss")

plt.legend()

plt.grid()

plt.savefig(LSTM_PLOT_DIR.joinpath("loss.png"))

plt.show()


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

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


# ============================================================
# DONE
# ============================================================

print("\n========================================")
print("DONE")
print("========================================")

print("Model:", MODEL_PATH)

print("Sequence scaler:", SEQUENCE_SCALER_PATH)

print("Statistics scaler:", STAT_SCALER_PATH)
