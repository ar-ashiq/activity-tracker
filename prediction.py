import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from paths import CNN_BEST_MODEL_PATH, CNN_SCALER_PATH, TESTING_USER_DIR

TESTING_DIR = TESTING_USER_DIR
MODEL_PATH = CNN_BEST_MODEL_PATH
SCALER_PATH = CNN_SCALER_PATH
OUTPUT_FILE = TESTING_DIR.joinpath("testing_predictions.json")
CONSOLIDATED_OUTPUT_FILE = TESTING_DIR.joinpath(
    "testing_predictions_consolidated.json"
)

WINDOW_SIZE = 100
MAX_ACTIVITY_GAP_SECONDS = 60.0
GYRO_FILENAME_FORMAT = "{timestamp}_25Hz.csv"
ACC_FILENAME_SUFFIX = ".m_raw_acc_25Hz.csv"
CLASS_NAMES = ["Walking", "Running", "Sitting"]

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


def load_scaler():
    import pickle

    with open(SCALER_PATH, "rb") as file:
        return pickle.load(file)


def timestamp_id_from_filename(filename):
    """Return the timestamp from an acc or gyro filename."""
    for suffix in (ACC_FILENAME_SUFFIX, "_25Hz.csv"):
        if filename.name.endswith(suffix):
            return filename.name[: -len(suffix)]

    return None


def find_gyro_file(gyro_dir, timestamp_id):
    if timestamp_id is None:
        return None

    gyro_file = gyro_dir / GYRO_FILENAME_FORMAT.format(timestamp=timestamp_id)
    return gyro_file if gyro_file.exists() else None


def calculate_user_shift(acc_dir, gyro_dir):
    shifts = []
    acc_files = sorted(acc_dir.glob("*.csv"))
    total_files = len(acc_files)
    start_time = time.perf_counter()

    print(f"[1/3] Calculating sensor time shift from {total_files} file pairs...", flush=True)

    for file_number, acc_file in enumerate(acc_files, start=1):
        timestamp_id = timestamp_id_from_filename(acc_file)
        if timestamp_id is None:
            continue

        gyro_file = find_gyro_file(gyro_dir, timestamp_id)

        if gyro_file is None:
            continue

        try:
            acc = pd.read_csv(acc_file)
            gyro = pd.read_csv(gyro_file)
        except Exception:
            continue

        required = ["timestamp", "gx", "gy", "gz"]
        if not all(column in acc.columns for column in required):
            continue
        if not all(column in gyro.columns for column in required):
            continue

        acc = acc.apply(pd.to_numeric, errors="coerce").dropna()
        gyro = gyro.apply(pd.to_numeric, errors="coerce").dropna()

        sample_count = min(len(acc), len(gyro))
        if sample_count == 0:
            continue

        shifts.append(
            np.median(
                gyro["timestamp"].iloc[:sample_count].to_numpy()
                - acc["timestamp"].iloc[:sample_count].to_numpy()
            )
        )

        if file_number == 1 or file_number % 500 == 0 or file_number == total_files:
            elapsed = time.perf_counter() - start_time
            print(
                f"  Shift analysis: {file_number}/{total_files} files "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    user_shift = float(np.mean(shifts)) if shifts else 0.0
    print(f"  Time shift: {user_shift:.6f}s", flush=True)
    return user_shift


def combine_sensor_data(acc_file, gyro_file, user_shift):
    acc = pd.read_csv(acc_file)
    gyro = pd.read_csv(gyro_file)

    required = ["timestamp", "gx", "gy", "gz"]
    if not all(column in acc.columns for column in required):
        return None
    if not all(column in gyro.columns for column in required):
        return None

    acc = acc[required].apply(pd.to_numeric, errors="coerce").dropna()
    gyro = gyro[required].apply(pd.to_numeric, errors="coerce").dropna()

    if len(acc) == 0 or len(gyro) == 0:
        return None

    gyro["timestamp_corrected"] = gyro["timestamp"] - user_shift
    acc = acc[
        (acc["timestamp"] >= gyro["timestamp_corrected"].min())
        & (acc["timestamp"] <= gyro["timestamp_corrected"].max())
    ].reset_index(drop=True)

    if len(acc) < WINDOW_SIZE:
        return None

    combined = pd.DataFrame(
        {
            "timestamp": acc["timestamp"].to_numpy(),
            "acc_x": acc["gx"].to_numpy(),
            "acc_y": acc["gy"].to_numpy(),
            "acc_z": acc["gz"].to_numpy(),
        }
    )

    gyro_timestamps = gyro["timestamp_corrected"].to_numpy()
    for axis in ("x", "y", "z"):
        combined[f"gyro_{axis}"] = np.interp(
            combined["timestamp"], gyro_timestamps, gyro[f"g{axis}"].to_numpy()
        )

    combined["acc_magnitude"] = np.sqrt(
        combined["acc_x"] ** 2 + combined["acc_y"] ** 2 + combined["acc_z"] ** 2
    )
    combined["gyro_magnitude"] = np.sqrt(
        combined["gyro_x"] ** 2 + combined["gyro_y"] ** 2 + combined["gyro_z"] ** 2
    )
    combined["acc_magnitude_change"] = combined["acc_magnitude"].diff().fillna(0)
    combined["gyro_magnitude_change"] = combined["gyro_magnitude"].diff().fillna(0)

    return combined


def predict_windows(model, scaler, combined, user_id, source_file):
    windows = []
    window_count = len(combined) // WINDOW_SIZE

    for window_id in range(window_count):
        start = window_id * WINDOW_SIZE
        end = start + WINDOW_SIZE
        window = combined.iloc[start:end]
        values = window[CHANNELS].to_numpy(dtype=np.float32)
        scaled_values = scaler.transform(values).reshape(1, WINDOW_SIZE, len(CHANNELS))
        probabilities = model.predict(scaled_values, verbose=0)[0]
        prediction_index = int(np.argmax(probabilities))

        windows.append(
            {
                "user_id": user_id,
                "file": source_file.name,
                "window_id": window_id,
                "start_timestamp": float(window["timestamp"].iloc[0]),
                "end_timestamp": float(window["timestamp"].iloc[-1]),
                "prediction": CLASS_NAMES[prediction_index],
                "confidence": float(probabilities[prediction_index]),
            }
        )

    return windows


def consolidate_predictions(predictions):
    """Merge nearby same-activity predictions into activity time ranges."""
    if not predictions:
        return []

    ordered_predictions = sorted(
        predictions,
        key=lambda item: item["start_timestamp"],
    )
    consolidated = []

    for prediction in ordered_predictions:
        current = {
            "activity": prediction["prediction"],
            "start_timestamp": prediction["start_timestamp"],
            "end_timestamp": prediction["end_timestamp"],
        }

        if not consolidated:
            consolidated.append(current)
            continue

        previous = consolidated[-1]
        is_same_activity = previous["activity"] == current["activity"]
        gap_seconds = current["start_timestamp"] - previous["end_timestamp"]
        is_nearby = gap_seconds <= MAX_ACTIVITY_GAP_SECONDS

        if is_same_activity and is_nearby:
            previous["end_timestamp"] = max(
                previous["end_timestamp"], current["end_timestamp"]
            )
        else:
            consolidated.append(current)

    return consolidated


def predict_testing_data():
    start_time = time.perf_counter()

    print(f"Testing folder: {TESTING_DIR}", flush=True)
    print(f"Model: {MODEL_PATH.name}", flush=True)
    print("Loading model and scaler...", flush=True)

    if not TESTING_DIR.exists():
        raise FileNotFoundError(f"Testing directory not found: {TESTING_DIR}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = load_scaler()
    predictions = []
    print(f"Model and scaler loaded ({time.perf_counter() - start_time:.1f}s)", flush=True)

    user_id = TESTING_DIR.name
    acc_dir = TESTING_DIR.joinpath("acc")
    gyro_dir = TESTING_DIR.joinpath("gyro")

    if not acc_dir.is_dir() or not gyro_dir.is_dir():
        raise FileNotFoundError(f"Expected acc and gyro directories under {TESTING_DIR}")

    user_shift = calculate_user_shift(acc_dir, gyro_dir)
    acc_files = sorted(acc_dir.glob("*.csv"))
    total_files = len(acc_files)
    processed_files = 0
    skipped_files = 0
    total_windows = 0
    prediction_start = time.perf_counter()

    print(f"[2/3] Predicting {total_files} accelerometer files...", flush=True)

    for file_number, acc_file in enumerate(acc_files, start=1):
        timestamp_id = timestamp_id_from_filename(acc_file)
        if timestamp_id is None:
            print(f"Skipping {acc_file.name}: invalid accelerometer filename")
            skipped_files += 1
            continue

        gyro_file = find_gyro_file(gyro_dir, timestamp_id)

        if gyro_file is None:
            print(f"Skipping {acc_file.name}: matching gyro file not found")
            skipped_files += 1
            continue

        try:
            combined = combine_sensor_data(acc_file, gyro_file, user_shift)
        except Exception as error:
            print(f"Skipping {acc_file.name}: {error}")
            skipped_files += 1
            continue

        if combined is not None:
            file_predictions = predict_windows(
                model, scaler, combined, user_id, acc_file
            )
            predictions.extend(file_predictions)
            processed_files += 1
            total_windows += len(file_predictions)
        else:
            skipped_files += 1

        if file_number == 1 or file_number % 100 == 0 or file_number == total_files:
            elapsed = time.perf_counter() - prediction_start
            rate = file_number / elapsed if elapsed else 0
            remaining = (total_files - file_number) / rate if rate else 0
            print(
                f"  Prediction progress: {file_number}/{total_files} files, "
                f"{total_windows} windows, {elapsed:.1f}s elapsed, "
                f"~{remaining:.1f}s remaining",
                flush=True,
            )

    print(
        f"[3/3] Writing {len(predictions)} predictions "
        f"({processed_files} files processed, {skipped_files} skipped)...",
        flush=True,
    )
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(predictions, file, indent=2)

    consolidated_predictions = consolidate_predictions(predictions)
    with open(CONSOLIDATED_OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(consolidated_predictions, file, indent=2)

    preview = predictions[:3]
    if preview:
        print("Prediction preview:")
        print(json.dumps(preview, indent=2))
    else:
        print("No predictions were generated.")
    print(f"Saved {len(predictions)} predictions to {OUTPUT_FILE}", flush=True)
    print(
        f"Saved {len(consolidated_predictions)} consolidated activities to "
        f"{CONSOLIDATED_OUTPUT_FILE}",
        flush=True,
    )
    print(f"Total time: {time.perf_counter() - start_time:.1f}s", flush=True)
    return predictions


if __name__ == "__main__":
    predict_testing_data()
