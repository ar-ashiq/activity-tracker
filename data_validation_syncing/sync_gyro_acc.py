import os
import csv
import numpy as np
from pathlib import Path
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Root folders
#
# Your actual structure is:
#
# raw_acc/
#     raw_acc/
#         USER_ID/
#             1449601597.m_raw_acc.dat
#             1449601658.m_raw_acc.dat
#             ...
#
# raw_gyro/
#     raw_gyro/
#         USER_ID/
#             1449601597.m_proc_gyro.dat
#             1449601658.m_proc_gyro.dat
#             ...
# ------------------------------------------------------------

ACC_ROOT = Path(
    "/Users/huzaifa/Documents/raw_acc/raw_acc"
)

GYRO_ROOT = Path(
    "/Users/huzaifa/Documents/raw_gyro/raw_gyro"
)

# ------------------------------------------------------------
# Output folder
# ------------------------------------------------------------

# CONFIGURATION
# ============================================================
LABEL_ROOT = Path("/Users/huzaifa/Downloads/ExtraSensory.per_uuid_features_labels/")  
OUTPUT_ROOT = Path("/Users/huzaifa/Documents/sync_temp3")

TARGET_FS = 25.0
DT = 1.0 / TARGET_FS


ACTIVITIES = {
    "label:LYING_DOWN":"Lying",
    "label:SITTING":"Sitting",
    "label:FIX_walking":"Walking",
    "label:FIX_running":"Running",
    "label:BICYCLING":"Bicycling",
    "label:OR_standing":"Standing"
}


# ============================================================
# LOAD LABELS FOR ONE USER
# ============================================================

def load_user_labels(user_name):
    """
    Load and clean the per-user label CSV.

    Returns a DataFrame with a numeric 'timestamp' column,
    or None if the file is missing / has no valid rows.
    """

    label_file = LABEL_ROOT / f"{user_name}.features_labels.csv"

    if not label_file.exists():
        print(f"{user_name}: Label file not found")
        return None

    labels = pd.read_csv(label_file)
    labels["timestamp"] = pd.to_numeric(labels["timestamp"], errors="coerce")
    labels = labels.dropna(subset=["timestamp"]).reset_index(drop=True)

    if len(labels) == 0:
        print(f"{user_name}: No valid label timestamps")
        return None

    return labels


# ============================================================
# GET ACTIVITY FOR ONE RECORDING
# ============================================================

def get_activity(labels, recording_id):
    """
    Find the activity label closest in time to this recording's
    timestamp, mirroring file 1's nearest-timestamp match.
    """

    burst_timestamp = float(recording_id)

    label_time_diff = np.abs(labels["timestamp"] - burst_timestamp)
    closest_index = label_time_diff.idxmin()
    label_row = labels.loc[closest_index]

    for label_column, activity_name in ACTIVITIES.items():

        if label_column not in labels.columns:
            continue

        value = label_row[label_column]

        if pd.notna(value) and bool(value):
            return activity_name

    return None


# ============================================================
# LOAD SENSOR FILE  (unchanged)
# ============================================================

def load_sensor_file(file_path):
    try:
        data = np.loadtxt(file_path, dtype=np.float64)
    except Exception as e:
        raise RuntimeError(f"Could not read file:\n{file_path}\nError: {e}")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 4:
        raise ValueError(f"Invalid file format:\n{file_path}")

    data = data[:, :4]
    valid = np.all(np.isfinite(data), axis=1)
    data = data[valid]
    order = np.argsort(data[:, 0])
    data = data[order]

    return data


# ============================================================
# EXTRACT RECORDING ID FROM FILENAME  (unchanged)
# ============================================================

def get_recording_id(filename):
    if filename.endswith(".m_raw_acc.dat"):
        return filename[:-len(".m_raw_acc.dat")]
    if filename.endswith(".m_proc_gyro.dat"):
        return filename[:-len(".m_proc_gyro.dat")]
    return None


# ============================================================
# FIND ACC / GYRO FILES  (unchanged)
# ============================================================

def find_acc_files(user_dir):
    acc_files = {}
    with os.scandir(user_dir) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            filename = entry.name
            if not filename.endswith(".m_raw_acc.dat"):
                continue
            recording_id = get_recording_id(filename)
            if recording_id is None:
                continue
            acc_files[recording_id] = Path(entry.path)
    return acc_files


def find_gyro_files(user_dir):
    gyro_files = {}
    with os.scandir(user_dir) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            filename = entry.name
            if not filename.endswith(".m_proc_gyro.dat"):
                continue
            recording_id = get_recording_id(filename)
            if recording_id is None:
                continue
            gyro_files[recording_id] = Path(entry.path)
    return gyro_files


# ============================================================
# PROCESS ONE ACC + GYRO PAIR
# ============================================================

def process_recording(recording_id, acc_file, gyro_file, output_file, labels):

    if output_file.exists():
        return "SKIPPED"

    try:
        acc = load_sensor_file(acc_file)
        gyro = load_sensor_file(gyro_file)

        if len(acc) < 2:
            return "ACC_TOO_SHORT"
        if len(gyro) < 2:
            return "GYRO_TOO_SHORT"

        acc_time = acc[:, 0]
        gyro_time = gyro[:, 0]

        start_time = max(acc_time[0], gyro_time[0])
        end_time = min(acc_time[-1], gyro_time[-1])

        if start_time >= end_time:
            return "NO_OVERLAP"

        common_time = np.arange(start_time, end_time, DT, dtype=np.float64)

        if len(common_time) < 2:
            return "TOO_SHORT"

        ax = np.interp(common_time, acc_time, acc[:, 1])
        ay = np.interp(common_time, acc_time, acc[:, 2])
        az = np.interp(common_time, acc_time, acc[:, 3])

        gx = np.interp(common_time, gyro_time, gyro[:, 1])
        gy = np.interp(common_time, gyro_time, gyro[:, 2])
        gz = np.interp(common_time, gyro_time, gyro[:, 3])

        acc_magnitude = np.sqrt(ax**2 + ay**2 + az**2)
        gyro_magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
        acc_magnitude_change = np.diff(acc_magnitude, prepend=acc_magnitude[0])
        gyro_magnitude_change = np.diff(gyro_magnitude, prepend=gyro_magnitude[0])

        # ------------------------------------------------------
        # LABEL LOOKUP (one activity per whole recording, same
        # as file 1 — applied to every row of this burst)
        # ------------------------------------------------------

        activity = get_activity(labels, recording_id)

        if activity is None:
            return "NO_LABEL"

        synchronized = np.column_stack([
            common_time,
            ax, ay, az,
            gx, gy, gz,
            acc_magnitude,
            gyro_magnitude,
            acc_magnitude_change,
            gyro_magnitude_change,
        ])

        output_file.parent.mkdir(parents=True, exist_ok=True)

        temp_file = output_file.with_suffix(".tmp")

        with open(temp_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "ax", "ay", "az", "gx", "gy", "gz",
                "acc_magnitude", "gyro_magnitude",
                "acc_magnitude_change", "gyro_magnitude_change",
                "label",
            ])
            for row in synchronized:
                writer.writerow([f"{v:.10f}" for v in row] + [activity])

        os.replace(temp_file, output_file)

        return "SUCCESS"

    except Exception as e:
        print()
        print("-" * 70)
        print("ERROR")
        print(f"Recording ID : {recording_id}")
        print(f"ACC file     : {acc_file}")
        print(f"GYRO file    : {gyro_file}")
        print(f"Error        : {e}")
        print("-" * 70)

        temp_file = output_file.with_suffix(".tmp")
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass

        return "ERROR"


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("EXTRASENSORY ACC + GYRO SYNCHRONIZATION")
    print("=" * 70)
    print()
    print(f"ACC root    : {ACC_ROOT}")
    print(f"GYRO root   : {GYRO_ROOT}")
    print(f"LABEL root  : {LABEL_ROOT}")
    print(f"Output root : {OUTPUT_ROOT}")
    print(f"Target rate : {TARGET_FS} Hz")
    print(f"Interval    : {DT} seconds")
    print()

    if not ACC_ROOT.exists():
        raise FileNotFoundError(f"ACC directory does not exist:\n{ACC_ROOT}")
    if not GYRO_ROOT.exists():
        raise FileNotFoundError(f"GYRO directory does not exist:\n{GYRO_ROOT}")
    if not LABEL_ROOT.exists():
        raise FileNotFoundError(f"LABEL directory does not exist:\n{LABEL_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Finding users...")

    acc_users = {}
    with os.scandir(ACC_ROOT) as entries:
        for entry in entries:
            if entry.is_dir():
                acc_users[entry.name] = Path(entry.path)

    gyro_users = {}
    with os.scandir(GYRO_ROOT) as entries:
        for entry in entries:
            if entry.is_dir():
                gyro_users[entry.name] = Path(entry.path)

    common_users = sorted(acc_users.keys() & gyro_users.keys())

    print()
    print(f"ACC users    : {len(acc_users)}")
    print(f"GYRO users   : {len(gyro_users)}")
    print(f"Common users : {len(common_users)}")

    total_recordings = 0
    processed = 0
    skipped = 0
    errors = 0
    no_overlap = 0
    acc_short = 0
    gyro_short = 0
    too_short = 0
    no_label = 0
    users_without_labels = 0

    log_file = OUTPUT_ROOT / "processing_log.csv"
    log_exists = log_file.exists()
    log_handle = open(log_file, "a", newline="")
    log_writer = csv.writer(log_handle)

    if not log_exists:
        log_writer.writerow(["user", "recording_id", "acc_file", "gyro_file", "status"])

    try:
        for user_index, user_name in enumerate(common_users, start=1):

            print()
            print("=" * 70)
            print(f"[USER {user_index}/{len(common_users)}]")
            print(user_name)
            print("=" * 70)

            # ------------------------------------------------
            # LOAD LABELS FOR THIS USER — skip user entirely
            # if no usable label file (same as file 1)
            # ------------------------------------------------

            labels = load_user_labels(user_name)

            if labels is None:
                users_without_labels += 1
                continue

            acc_user_dir = acc_users[user_name]
            gyro_user_dir = gyro_users[user_name]

            print("Finding ACC files...")
            acc_files = find_acc_files(acc_user_dir)

            print("Finding Gyro files...")
            gyro_files = find_gyro_files(gyro_user_dir)

            common_recording_ids = sorted(acc_files.keys() & gyro_files.keys())

            print()
            print(f"ACC recordings  : {len(acc_files)}")
            print(f"GYRO recordings : {len(gyro_files)}")
            print(f"Matched         : {len(common_recording_ids)}")

            for recording_index, recording_id in enumerate(common_recording_ids, start=1):

                total_recordings += 1

                acc_file = acc_files[recording_id]
                gyro_file = gyro_files[recording_id]

                output_file = (
                    OUTPUT_ROOT / user_name / recording_id / "synchronized_25hz.csv"
                )

                print(
                    f"[{recording_index}/{len(common_recording_ids)}] {recording_id}",
                    end=" "
                )

                result = process_recording(
                    recording_id, acc_file, gyro_file, output_file, labels
                )

                print(f"-> {result}")

                if result == "SUCCESS":
                    processed += 1
                elif result == "SKIPPED":
                    skipped += 1
                elif result == "NO_OVERLAP":
                    no_overlap += 1
                elif result == "ACC_TOO_SHORT":
                    acc_short += 1
                elif result == "GYRO_TOO_SHORT":
                    gyro_short += 1
                elif result == "TOO_SHORT":
                    too_short += 1
                elif result == "NO_LABEL":
                    no_label += 1
                else:
                    errors += 1

                log_writer.writerow([
                    user_name, recording_id, str(acc_file), str(gyro_file), result
                ])
                log_handle.flush()

    finally:
        log_handle.close()

    print()
    print()
    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total recordings      : {total_recordings}")
    print(f"Processed             : {processed}")
    print(f"Skipped               : {skipped}")
    print(f"No overlap            : {no_overlap}")
    print(f"ACC too short         : {acc_short}")
    print(f"Gyro too short        : {gyro_short}")
    print(f"Too short             : {too_short}")
    print(f"No label match        : {no_label}")
    print(f"Users without labels  : {users_without_labels}")
    print(f"Errors                : {errors}")
    print()
    print(f"Processing log: {log_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()