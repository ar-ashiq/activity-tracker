import os
import gc
import glob
import pickle
import json
import math
import time

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


# ============================================================
# OPTIONAL: GPU CONFIGURATION
# ============================================================

print("\n========================================")
print("TENSORFLOW DEVICES")
print("========================================")

print("TensorFlow version:", tf.__version__)
print("Physical GPUs:", tf.config.list_physical_devices("GPU"))
print("Physical CPUs:", tf.config.list_physical_devices("CPU"))

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print("GPU memory growth enabled.")

    except RuntimeError as e:
        print("Could not enable GPU memory growth:", e)

else:
    print("WARNING: No TensorFlow GPU detected.")
    print("Training will use CPU.")


# ============================================================
# SPLIT
# ============================================================

SPLIT_PATH = "split.json"

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

READ_COLUMNS = CHANNELS + ["label"]

DTYPE_MAP = {
    col: "float32"
    for col in CHANNELS
}

CHUNK_SIZE = 200_000


# ============================================================
# CACHE / MEMMAP PATHS
# ============================================================

CACHE_DIR = MODEL_DIR / "raw_lstm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_X_PATH = CACHE_DIR / "X_train.dat"
TRAIN_Y_PATH = CACHE_DIR / "y_train.dat"

VAL_X_PATH = CACHE_DIR / "X_val.dat"
VAL_Y_PATH = CACHE_DIR / "y_val.dat"

TEST_X_PATH = CACHE_DIR / "X_test.dat"
TEST_Y_PATH = CACHE_DIR / "y_test.dat"

CACHE_INFO_PATH = CACHE_DIR / "cache_info.json"


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

print("\n========================================")
print("USERS")
print("========================================")

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
# FIND CSV FILES
# ============================================================

def get_user_csv_files(user_list):

    all_files = []

    for user in user_list:

        user_dir = os.path.join(DATA_DIR, user)

        csv_files = glob.glob(
            os.path.join(
                user_dir,
                "**",
                "synchronized_25hz.csv"
            ),
            recursive=True
        )

        print(
            f"User {user}: {len(csv_files)} CSV files found"
        )

        all_files.extend(csv_files)

    return all_files


# ============================================================
# GET LABEL FROM CSV
# ============================================================

def get_file_label(file_path):

    try:

        header_df = pd.read_csv(
            file_path,
            usecols=["label"],
            nrows=1,
        )

        if "label" not in header_df.columns:
            return None

        label_name = str(
            header_df["label"].iloc[0]
        )

        if label_name not in LABEL_MAP:
            return None

        return LABEL_MAP[label_name]

    except Exception as e:

        print(
            "Could not read label:",
            file_path
        )

        print("Error:", e)

        return None


# ============================================================
# PROCESS ONE CHUNK
# ============================================================

def process_chunk(chunk):

    sensor_data = chunk[CHANNELS]

    # Convert malformed values to NaN
    for col in CHANNELS:

        if sensor_data[col].dtype != np.float32:

            sensor_data[col] = pd.to_numeric(
                sensor_data[col],
                errors="coerce"
            )

    sensor_data = sensor_data.dropna()

    values = sensor_data.to_numpy(
        dtype=np.float32
    )

    del sensor_data

    return values


# ============================================================
# COUNT WINDOWS IN ONE FILE
# ============================================================

def count_windows_in_file(file_path):

    total_rows = 0

    try:

        for chunk in pd.read_csv(
            file_path,
            usecols=CHANNELS,
            dtype=DTYPE_MAP,
            chunksize=CHUNK_SIZE,
        ):

            values = process_chunk(chunk)

            total_rows += len(values)

            del values
            del chunk

        return total_rows // WINDOW_SIZE

    except Exception as e:

        print(
            "Could not count windows:",
            file_path
        )

        print("Error:", e)

        return 0


# ============================================================
# FIRST PASS
#
# 1. Fit scaler using TRAINING DATA ONLY
# 2. Count windows for train/val/test
# ============================================================

def first_pass(
    train_files,
    val_files,
    test_files,
):

    print("\n========================================")
    print("FIRST PASS")
    print("========================================")

    print(
        "This pass fits the scaler and counts windows."
    )

    scaler = StandardScaler()

    train_window_count = 0
    val_window_count = 0
    test_window_count = 0

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\nProcessing TRAIN files...")

    for file_index, file_path in enumerate(train_files):

        label = get_file_label(file_path)

        if label is None:
            continue

        try:

            file_windows = 0

            for chunk in pd.read_csv(
                file_path,
                usecols=CHANNELS,
                dtype=DTYPE_MAP,
                chunksize=CHUNK_SIZE,
            ):

                values = process_chunk(chunk)

                # ------------------------------------------------
                # FIT SCALER
                # ------------------------------------------------

                if len(values) > 0:

                    scaler.partial_fit(values)

                # ------------------------------------------------
                # COUNT WINDOWS
                # ------------------------------------------------

                file_windows += (
                    len(values) // WINDOW_SIZE
                )

                del values
                del chunk

            train_window_count += file_windows

        except Exception as e:

            print(
                "\nError processing:",
                file_path
            )

            print("Error:", e)

        if (file_index + 1) % 100 == 0:

            print(
                f"Train files processed: "
                f"{file_index + 1}/{len(train_files)} "
                f"| Windows: {train_window_count:,}"
            )

        if (file_index + 1) % 25 == 0:
            gc.collect()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print("\nProcessing VALIDATION files...")

    for file_index, file_path in enumerate(val_files):

        label = get_file_label(file_path)

        if label is None:
            continue

        val_window_count += count_windows_in_file(
            file_path
        )

        if (file_index + 1) % 100 == 0:

            print(
                f"Validation files processed: "
                f"{file_index + 1}/{len(val_files)} "
                f"| Windows: {val_window_count:,}"
            )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("\nProcessing TEST files...")

    for file_index, file_path in enumerate(test_files):

        label = get_file_label(file_path)

        if label is None:
            continue

        test_window_count += count_windows_in_file(
            file_path
        )

        if (file_index + 1) % 100 == 0:

            print(
                f"Test files processed: "
                f"{file_index + 1}/{len(test_files)} "
                f"| Windows: {test_window_count:,}"
            )

    # --------------------------------------------------------
    # SAVE SCALER
    # --------------------------------------------------------

    with open(
        SCALER_PATH,
        "wb"
    ) as file:

        pickle.dump(
            scaler,
            file
        )

    print(
        "\nScaler saved to:",
        SCALER_PATH
    )

    print("\n========================================")
    print("WINDOW COUNTS")
    print("========================================")

    print(
        "Training windows:",
        f"{train_window_count:,}"
    )

    print(
        "Validation windows:",
        f"{val_window_count:,}"
    )

    print(
        "Test windows:",
        f"{test_window_count:,}"
    )

    return (
        scaler,
        train_window_count,
        val_window_count,
        test_window_count,
    )


# ============================================================
# CREATE MEMMAP
# ============================================================

def create_memmaps(
    train_count,
    val_count,
    test_count,
):

    print("\n========================================")
    print("CREATING DISK-BACKED DATASETS")
    print("========================================")

    X_train = np.memmap(
        TRAIN_X_PATH,
        dtype=np.float32,
        mode="w+",
        shape=(
            train_count,
            WINDOW_SIZE,
            NUM_CHANNELS,
        ),
    )

    y_train = np.memmap(
        TRAIN_Y_PATH,
        dtype=np.int32,
        mode="w+",
        shape=(train_count,),
    )

    X_val = np.memmap(
        VAL_X_PATH,
        dtype=np.float32,
        mode="w+",
        shape=(
            val_count,
            WINDOW_SIZE,
            NUM_CHANNELS,
        ),
    )

    y_val = np.memmap(
        VAL_Y_PATH,
        dtype=np.int32,
        mode="w+",
        shape=(val_count,),
    )

    X_test = np.memmap(
        TEST_X_PATH,
        dtype=np.float32,
        mode="w+",
        shape=(
            test_count,
            WINDOW_SIZE,
            NUM_CHANNELS,
        ),
    )

    y_test = np.memmap(
        TEST_Y_PATH,
        dtype=np.int32,
        mode="w+",
        shape=(test_count,),
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


# ============================================================
# SECOND PASS
#
# Read CSVs again, normalize, create windows,
# directly write them to disk.
# ============================================================

def fill_memmap(
    files,
    scaler,
    X_memmap,
    y_memmap,
    dataset_name,
):

    print("\n========================================")
    print(f"WRITING {dataset_name.upper()} DATA")
    print("========================================")

    write_index = 0

    total_files = len(files)

    for file_index, file_path in enumerate(files):

        label = get_file_label(file_path)

        if label is None:
            continue

        try:

            for chunk in pd.read_csv(
                file_path,
                usecols=CHANNELS,
                dtype=DTYPE_MAP,
                chunksize=CHUNK_SIZE,
            ):

                values = process_chunk(chunk)

                if len(values) == 0:
                    del values
                    del chunk
                    continue

                # ------------------------------------------------
                # Normalize
                # ------------------------------------------------

                values = scaler.transform(
                    values
                ).astype(
                    np.float32,
                    copy=False
                )

                # ------------------------------------------------
                # Create complete windows
                # ------------------------------------------------

                number_of_windows = (
                    len(values) // WINDOW_SIZE
                )

                usable_length = (
                    number_of_windows
                    * WINDOW_SIZE
                )

                if number_of_windows > 0:

                    windows = values[
                        :usable_length
                    ].reshape(
                        number_of_windows,
                        WINDOW_SIZE,
                        NUM_CHANNELS,
                    )

                    end_index = (
                        write_index
                        + number_of_windows
                    )

                    X_memmap[
                        write_index:end_index
                    ] = windows

                    y_memmap[
                        write_index:end_index
                    ] = label

                    write_index = end_index

                    del windows

                del values
                del chunk

            if (file_index + 1) % 100 == 0:

                print(
                    f"{dataset_name}: "
                    f"{file_index + 1}/{total_files} files "
                    f"| {write_index:,} windows"
                )

            if (file_index + 1) % 25 == 0:
                gc.collect()

        except Exception as e:

            print(
                "\nCould not process:",
                file_path
            )

            print(
                "Error:",
                e
            )

    X_memmap.flush()
    y_memmap.flush()

    print(
        f"\n{dataset_name} complete."
    )

    print(
        "Windows written:",
        f"{write_index:,}"
    )

    return write_index


# ============================================================
# LOAD / CREATE CACHE
# ============================================================

train_files = get_user_csv_files(
    train_users
)

val_files = get_user_csv_files(
    val_users
)

test_files = get_user_csv_files(
    test_users
)


cache_exists = (
    TRAIN_X_PATH.exists()
    and TRAIN_Y_PATH.exists()
    and VAL_X_PATH.exists()
    and VAL_Y_PATH.exists()
    and TEST_X_PATH.exists()
    and TEST_Y_PATH.exists()
    and CACHE_INFO_PATH.exists()
)


if cache_exists:

    print("\n========================================")
    print("CACHE FOUND")
    print("========================================")

    with open(
        CACHE_INFO_PATH,
        "r"
    ) as f:

        cache_info = json.load(f)

    train_count = cache_info["train_count"]
    val_count = cache_info["val_count"]
    test_count = cache_info["test_count"]

    print(
        "Training windows:",
        f"{train_count:,}"
    )

    print(
        "Validation windows:",
        f"{val_count:,}"
    )

    print(
        "Test windows:",
        f"{test_count:,}"
    )

    with open(
        SCALER_PATH,
        "rb"
    ) as file:

        scaler = pickle.load(file)

else:

    print("\n========================================")
    print("NO CACHE FOUND")
    print("========================================")

    (
        scaler,
        train_count,
        val_count,
        test_count,
    ) = first_pass(
        train_files,
        val_files,
        test_files,
    )

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = create_memmaps(
        train_count,
        val_count,
        test_count,
    )

    # --------------------------------------------------------
    # Fill training data
    # --------------------------------------------------------

    fill_memmap(
        train_files,
        scaler,
        X_train,
        y_train,
        "TRAIN",
    )

    # --------------------------------------------------------
    # Fill validation data
    # --------------------------------------------------------

    fill_memmap(
        val_files,
        scaler,
        X_val,
        y_val,
        "VALIDATION",
    )

    # --------------------------------------------------------
    # Fill test data
    # --------------------------------------------------------

    fill_memmap(
        test_files,
        scaler,
        X_test,
        y_test,
        "TEST",
    )

    cache_info = {
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "window_size": WINDOW_SIZE,
        "num_channels": NUM_CHANNELS,
    }

    with open(
        CACHE_INFO_PATH,
        "w"
    ) as f:

        json.dump(
            cache_info,
            f,
            indent=4
        )

    print(
        "\nCache information saved to:",
        CACHE_INFO_PATH
    )


# ============================================================
# OPEN MEMMAPS
# ============================================================

print("\n========================================")
print("OPENING DATASETS")
print("========================================")

X_train = np.memmap(
    TRAIN_X_PATH,
    dtype=np.float32,
    mode="r",
    shape=(
        train_count,
        WINDOW_SIZE,
        NUM_CHANNELS,
    ),
)

y_train = np.memmap(
    TRAIN_Y_PATH,
    dtype=np.int32,
    mode="r",
    shape=(train_count,),
)

X_val = np.memmap(
    VAL_X_PATH,
    dtype=np.float32,
    mode="r",
    shape=(
        val_count,
        WINDOW_SIZE,
        NUM_CHANNELS,
    ),
)

y_val = np.memmap(
    VAL_Y_PATH,
    dtype=np.int32,
    mode="r",
    shape=(val_count,),
)

X_test = np.memmap(
    TEST_X_PATH,
    dtype=np.float32,
    mode="r",
    shape=(
        test_count,
        WINDOW_SIZE,
        NUM_CHANNELS,
    ),
)

y_test = np.memmap(
    TEST_Y_PATH,
    dtype=np.int32,
    mode="r",
    shape=(test_count,),
)


print("\nTraining shape:")
print(X_train.shape)

print(y_train.shape)

print("\nValidation shape:")
print(X_val.shape)

print(y_val.shape)

print("\nTest shape:")
print(X_test.shape)

print(y_test.shape)


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

    train_count_class = np.sum(
        y_train == i
    )

    val_count_class = np.sum(
        y_val == i
    )

    test_count_class = np.sum(
        y_test == i
    )

    print(
        f"{class_name:10s} | "
        f"Train: {train_count_class:8d} | "
        f"Val: {val_count_class:8d} | "
        f"Test: {test_count_class:8d}"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = np.unique(
    y_train
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=np.asarray(y_train),
)

class_weights = dict(
    zip(
        classes,
        weights
    )
)


print("\n========================================")
print("CLASS WEIGHTS")
print("========================================")

for class_id, weight in class_weights.items():

    print(
        CLASS_NAMES[class_id],
        ":",
        round(weight, 4)
    )


# ============================================================
# TF.DATA INPUT PIPELINE
# ============================================================

BATCH_SIZE = 64


def create_dataset(
    X,
    y,
    batch_size,
    shuffle,
):

    # --------------------------------------------------------
    # Generator
    #
    # Reads only enough data for batches instead of converting
    # the complete memmap into a normal NumPy array.
    # --------------------------------------------------------

    def generator():

        indices = np.arange(
            len(X)
        )

        if shuffle:

            np.random.shuffle(
                indices
            )

        for start in range(
            0,
            len(indices),
            batch_size,
        ):

            batch_indices = indices[
                start:start + batch_size
            ]

            batch_x = np.asarray(
                X[batch_indices],
                dtype=np.float32
            )

            batch_y = np.asarray(
                y[batch_indices],
                dtype=np.int32
            )

            yield (
                batch_x,
                batch_y
            )

    output_signature = (
        tf.TensorSpec(
            shape=(
                None,
                WINDOW_SIZE,
                NUM_CHANNELS,
            ),
            dtype=tf.float32,
        ),
        tf.TensorSpec(
            shape=(None,),
            dtype=tf.int32,
        ),
    )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature,
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# DATASETS
# ============================================================

train_dataset = create_dataset(
    X_train,
    y_train,
    BATCH_SIZE,
    shuffle=True,
)

val_dataset = create_dataset(
    X_val,
    y_val,
    BATCH_SIZE,
    shuffle=False,
)

test_dataset = create_dataset(
    X_test,
    y_test,
    BATCH_SIZE,
    shuffle=False,
)


TRAIN_STEPS = math.ceil(
    train_count / BATCH_SIZE
)

VAL_STEPS = math.ceil(
    val_count / BATCH_SIZE
)

TEST_STEPS = math.ceil(
    test_count / BATCH_SIZE
)


print("\n========================================")
print("DATASET STEPS")
print("========================================")

print(
    "Train steps/epoch:",
    TRAIN_STEPS
)

print(
    "Validation steps:",
    VAL_STEPS
)

print(
    "Test steps:",
    TEST_STEPS
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

x = layers.Dropout(
    0.3
)(x)

x = layers.LSTM(
    32
)(x)

x = layers.Dropout(
    0.3
)(x)

x = layers.Dense(
    32,
    activation="relu",
)(x)

x = layers.Dropout(
    0.3
)(x)

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

print("\n========================================")
print("COMPILE")
print("========================================")

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

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Training windows:",
    f"{train_count:,}"
)

print(
    "Steps per epoch:",
    TRAIN_STEPS
)

print(
    "Expected training epochs:",
    50
)

print(
    "GPU devices:",
    tf.config.list_physical_devices("GPU")
)


history = model.fit(
    train_dataset,
    validation_data=val_dataset,

    epochs=50,

    steps_per_epoch=TRAIN_STEPS,

    validation_steps=VAL_STEPS,

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

model.save(
    MODEL_PATH
)

print(
    "\nModel saved to:",
    MODEL_PATH
)


# ============================================================
# TEST
# ============================================================

print("\n========================================")
print("TESTING")
print("========================================")

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    steps=TEST_STEPS,
    verbose=1,
)

print("\nTest Loss:")
print(test_loss)

print("\nTest Accuracy:")
print(test_accuracy)


# ============================================================
# PREDICTIONS
# ============================================================

print("\n========================================")
print("PREDICTIONS")
print("========================================")

y_probability = model.predict(
    test_dataset,
    steps=TEST_STEPS,
    verbose=1,
)

y_pred = np.argmax(
    y_probability,
    axis=1,
)

# Ensure exact test length
y_pred = y_pred[:len(y_test)]


# ============================================================
# METRICS
# ============================================================

y_test_array = np.asarray(
    y_test
)

accuracy = accuracy_score(
    y_test_array,
    y_pred,
)

balanced_accuracy = balanced_accuracy_score(
    y_test_array,
    y_pred,
)

macro_f1 = f1_score(
    y_test_array,
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
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test_array,
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

    file.write(
        report
    )


# ============================================================
# SAVE METRICS
# ============================================================

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
    y_test_array,
    y_pred,
    labels=np.arange(
        NUM_CLASSES
    ),
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

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

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

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

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
    len(y_test_array),
)

for i in range(
    number_to_show
):

    actual = CLASS_NAMES[
        y_test_array[i]
    ]

    predicted = CLASS_NAMES[
        y_pred[i]
    ]

    confidence = (
        np.max(
            y_probability[i]
        )
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

print(
    "Cache:",
    CACHE_DIR,
)