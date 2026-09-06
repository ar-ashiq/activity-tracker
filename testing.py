import os
import numpy as np
import pandas as pd

ACC_DIR = "/Users/ashiqar/chore/ubiquitous/Codes/datasets/acc_outputs"
GYRO_DIR = "/Users/ashiqar/chore/ubiquitous/Codes/datasets/gyro_outputs"

OUTPUT_FILE = "/Users/ashiqar/chore/ubiquitous/Codes/datasets/new_data.csv"

WINDOW_SIZE = 100


def calculate_user_shift(user_id):
    acc_user_dir = os.path.join(ACC_DIR, user_id)
    gyro_user_dir = os.path.join(GYRO_DIR, user_id)

    file_medians = []

    for acc_filename in sorted(os.listdir(acc_user_dir)):

        if not acc_filename.endswith(".csv"):
            continue

        timestamp_id = acc_filename.split(".")[0]

        gyro_filename = f"{timestamp_id}_25Hz.csv"

        acc_file = os.path.join(acc_user_dir, acc_filename)

        gyro_file = os.path.join(gyro_user_dir, gyro_filename)

        if not os.path.exists(gyro_file):
            continue

        acc = pd.read_csv(acc_file)
        gyro = pd.read_csv(gyro_file)

        gyro = gyro.dropna(subset=["gx", "gy", "gz"]).reset_index(drop=True)

        if len(gyro) == 0:
            continue

        n = min(len(acc), len(gyro))

        shifts = (
            gyro["timestamp"].iloc[:n].to_numpy() - acc["timestamp"].iloc[:n].to_numpy()
        )

        file_medians.append(np.median(shifts))

    if not file_medians:
        return None

    return np.mean(file_medians)


def process_burst(user_id, timestamp_id, user_shift):

    acc_file = os.path.join(ACC_DIR, user_id, f"{timestamp_id}.m_raw_acc_25Hz.csv")

    gyro_file = os.path.join(GYRO_DIR, user_id, f"{timestamp_id}_25Hz.csv")

    if not os.path.exists(acc_file):
        return []

    if not os.path.exists(gyro_file):
        return []

    acc = pd.read_csv(acc_file)
    gyro = pd.read_csv(gyro_file)

    gyro = gyro.dropna(subset=["gx", "gy", "gz"]).reset_index(drop=True)

    if len(gyro) == 0:
        return []

    gyro["timestamp_corrected"] = gyro["timestamp"] - user_shift

    min_time = gyro["timestamp_corrected"].min()
    max_time = gyro["timestamp_corrected"].max()

    acc = acc[(acc["timestamp"] >= min_time) & (acc["timestamp"] <= max_time)].copy()

    if len(acc) < WINDOW_SIZE:
        return []

    acc = acc.reset_index(drop=True)

    acc = acc.rename(columns={"gx": "acc_x", "gy": "acc_y", "gz": "acc_z"})

    gyro = gyro.rename(columns={"gx": "gyro_x", "gy": "gyro_y", "gz": "gyro_z"})

    combined = pd.DataFrame()

    combined["timestamp"] = acc["timestamp"]

    combined["acc_x"] = acc["acc_x"].to_numpy()
    combined["acc_y"] = acc["acc_y"].to_numpy()
    combined["acc_z"] = acc["acc_z"].to_numpy()

    combined["gyro_x"] = np.interp(
        combined["timestamp"], gyro["timestamp_corrected"], gyro["gyro_x"]
    )

    combined["gyro_y"] = np.interp(
        combined["timestamp"], gyro["timestamp_corrected"], gyro["gyro_y"]
    )

    combined["gyro_z"] = np.interp(
        combined["timestamp"], gyro["timestamp_corrected"], gyro["gyro_z"]
    )

    # Sample-level features

    combined["acc_magnitude"] = np.sqrt(
        combined["acc_x"] ** 2 + combined["acc_y"] ** 2 + combined["acc_z"] ** 2
    )

    combined["gyro_magnitude"] = np.sqrt(
        combined["gyro_x"] ** 2 + combined["gyro_y"] ** 2 + combined["gyro_z"] ** 2
    )

    combined["acc_magnitude_change"] = combined["acc_magnitude"].diff().fillna(0)

    combined["gyro_magnitude_change"] = combined["gyro_magnitude"].diff().fillna(0)

    results = []

    number_of_windows = len(combined) // WINDOW_SIZE

    for window_id in range(number_of_windows):

        start = window_id * WINDOW_SIZE
        end = start + WINDOW_SIZE

        window = combined.iloc[start:end].copy()

        acc_values = window[["acc_x", "acc_y", "acc_z"]].to_numpy()

        gyro_values = window[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()

        # Window-level features

        window["acc_mean"] = np.mean(acc_values)

        window["acc_std"] = np.std(acc_values)

        window["gyro_mean"] = np.mean(gyro_values)

        window["gyro_std"] = np.std(gyro_values)

        window["acc_energy"] = np.mean(acc_values**2)

        window["gyro_energy"] = np.mean(gyro_values**2)

        window["user_id"] = user_id
        window["source_timestamp"] = timestamp_id
        window["window_id"] = window_id

        window["window_start"] = window["timestamp"].iloc[0]

        window["window_end"] = window["timestamp"].iloc[-1]

        results.append(window)

    return results


all_windows = []

users = sorted(
    [user for user in os.listdir(ACC_DIR) if os.path.isdir(os.path.join(ACC_DIR, user))]
)

print("Users found:", len(users))

for user_id in users:

    print("\nProcessing:", user_id)

    user_shift = calculate_user_shift(user_id)

    if user_shift is None:
        print("No valid synchronization data")
        continue

    print("Time shift:", user_shift)

    acc_user_dir = os.path.join(ACC_DIR, user_id)

    for acc_filename in sorted(os.listdir(acc_user_dir)):

        if not acc_filename.endswith(".csv"):
            continue

        timestamp_id = acc_filename.split(".")[0]

        windows = process_burst(user_id, timestamp_id, user_shift)

        all_windows.extend(windows)

        print(timestamp_id, "->", len(windows), "windows")


if all_windows:

    new_data = pd.concat(all_windows, ignore_index=True)

    new_data.to_csv(OUTPUT_FILE, index=False)

    print("\n================================")
    print("KNOWLEDGE BASE CREATED")
    print("================================")

    print("Rows:", len(new_data))

    print(
        "Windows:",
        new_data[["user_id", "source_timestamp", "window_id"]]
        .drop_duplicates()
        .shape[0],
    )

    print("Saved to:", OUTPUT_FILE)

else:

    print("No valid windows found.")
