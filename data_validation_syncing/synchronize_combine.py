import os
import pandas as pd
import numpy as np

ACC_DIR = "/Users/ashiqar/chore/ubiquitous/Codes/datasets/acc_outputs"

GYRO_DIR = "/Users/ashiqar/chore/ubiquitous/Codes/datasets/gyro_outputs"

LABEL_DIR = "/Users/ashiqar/chore/ubiquitous/Codes/datasets/ExtraSensory.per_uuid_features_labels"

OUTPUT_DIR = "/Users/ashiqar/chore/ubiquitous/Codes/datasets/synchronized_outputs"


os.makedirs(OUTPUT_DIR, exist_ok=True)


ACTIVITIES = {
    "label:FIX_walking": "Walking",
    "label:FIX_running": "Running",
    "label:SITTING": "Sitting",
}


for user_id in sorted(os.listdir(ACC_DIR)):

    acc_user_dir = os.path.join(ACC_DIR, user_id)

    gyro_user_dir = os.path.join(GYRO_DIR, user_id)

    label_file = os.path.join(LABEL_DIR, f"{user_id}.features_labels.csv")

    if not os.path.isdir(acc_user_dir):
        continue

    if not os.path.isdir(gyro_user_dir):

        print(f"{user_id}: Gyro directory not found")

        continue

    if not os.path.exists(label_file):

        print(f"{user_id}: Label file not found")

        continue

    file_medians = []

    for acc_filename in sorted(os.listdir(acc_user_dir)):

        if not acc_filename.endswith(".csv"):
            continue

        timestamp_id = acc_filename.split(".")[0]

        gyro_filename = f"{timestamp_id}_25Hz.csv"

        acc_file = os.path.join(acc_user_dir, acc_filename)

        gyro_file = os.path.join(gyro_user_dir, gyro_filename)

        if not os.path.exists(gyro_file):

            print(f"{user_id}: Gyro file missing -> " f"{gyro_filename}")

            continue

        try:

            acc = pd.read_csv(acc_file)

            gyro = pd.read_csv(gyro_file)

        except Exception as e:

            print(f"{user_id}: Error reading " f"{timestamp_id}: {e}")

            continue

        gyro_valid = gyro.dropna(subset=["gx", "gy", "gz"]).reset_index(drop=True)

        if len(gyro_valid) == 0:
            continue

        acc["timestamp"] = pd.to_numeric(acc["timestamp"], errors="coerce")

        gyro_valid["timestamp"] = pd.to_numeric(
            gyro_valid["timestamp"], errors="coerce"
        )

        acc = acc.dropna(subset=["timestamp"]).reset_index(drop=True)

        gyro_valid = gyro_valid.dropna(subset=["timestamp"]).reset_index(drop=True)

        if len(acc) == 0 or len(gyro_valid) == 0:
            continue

        n = min(len(acc), len(gyro_valid))

        shifts = (
            gyro_valid["timestamp"].iloc[:n].to_numpy()
            - acc["timestamp"].iloc[:n].to_numpy()
        )

        file_median = np.median(shifts)

        file_medians.append(file_median)

    if not file_medians:

        print(f"{user_id}: No valid Acc/Gyro files")

        continue

    user_shift = np.mean(file_medians)

    print("\n==========================================")
    print(f"USER: {user_id}")

    print(f"Average median time shift: " f"{user_shift:.9f} seconds")

    print(f"Number of bursts used: " f"{len(file_medians)}")

    print("==========================================\n")

    labels = pd.read_csv(label_file)

    labels["timestamp"] = pd.to_numeric(labels["timestamp"], errors="coerce")

    labels = labels.dropna(subset=["timestamp"]).reset_index(drop=True)

    if len(labels) == 0:

        print(f"{user_id}: No valid label timestamps")

        continue

    user_output_dir = os.path.join(OUTPUT_DIR, user_id)

    os.makedirs(user_output_dir, exist_ok=True)

    for acc_filename in sorted(os.listdir(acc_user_dir)):

        if not acc_filename.endswith(".csv"):
            continue

        timestamp_id = acc_filename.split(".")[0]

        gyro_filename = f"{timestamp_id}_25Hz.csv"

        acc_file = os.path.join(acc_user_dir, acc_filename)

        gyro_file = os.path.join(gyro_user_dir, gyro_filename)

        if not os.path.exists(gyro_file):
            continue

        try:

            acc = pd.read_csv(acc_file)

            gyro = pd.read_csv(gyro_file)

        except Exception as e:

            print(f"Error reading " f"{timestamp_id}: {e}")

            continue

        required_columns = ["timestamp", "gx", "gy", "gz"]

        if not all(col in acc.columns for col in required_columns):

            print(f"Skipping {timestamp_id}: " f"invalid accelerometer columns")

            continue

        if not all(col in gyro.columns for col in required_columns):

            print(f"Skipping {timestamp_id}: " f"invalid gyroscope columns")

            continue

        acc["timestamp"] = pd.to_numeric(acc["timestamp"], errors="coerce")

        gyro["timestamp"] = pd.to_numeric(gyro["timestamp"], errors="coerce")

        acc["gx"] = pd.to_numeric(acc["gx"], errors="coerce")

        acc["gy"] = pd.to_numeric(acc["gy"], errors="coerce")

        acc["gz"] = pd.to_numeric(acc["gz"], errors="coerce")

        gyro["gx"] = pd.to_numeric(gyro["gx"], errors="coerce")

        gyro["gy"] = pd.to_numeric(gyro["gy"], errors="coerce")

        gyro["gz"] = pd.to_numeric(gyro["gz"], errors="coerce")

        acc = acc.dropna(subset=["timestamp", "gx", "gy", "gz"]).reset_index(drop=True)

        gyro = gyro.dropna(subset=["timestamp", "gx", "gy", "gz"]).reset_index(
            drop=True
        )

        if len(acc) == 0:

            print(f"Skipping {timestamp_id}: " f"no valid accelerometer data")

            continue

        if len(gyro) == 0:

            print(f"Skipping {timestamp_id}: " f"no valid gyro data")

            continue

        gyro["timestamp_corrected"] = gyro["timestamp"] - user_shift

        min_gyro_time = gyro["timestamp_corrected"].min()

        max_gyro_time = gyro["timestamp_corrected"].max()

        acc = acc[
            (acc["timestamp"] >= min_gyro_time) & (acc["timestamp"] <= max_gyro_time)
        ].copy()

        acc = acc.reset_index(drop=True)

        if len(acc) == 0:

            print(f"Skipping {timestamp_id}: " f"no overlapping Acc/Gyro time")

            continue

        burst_timestamp = float(timestamp_id)

        label_time_diff = np.abs(labels["timestamp"] - burst_timestamp)

        closest_index = label_time_diff.idxmin()

        label_row = labels.loc[closest_index]

        activity = None

        for label_column, activity_name in ACTIVITIES.items():

            if label_column not in labels.columns:
                continue

            value = label_row[label_column]

            if pd.notna(value) and bool(value):

                activity = activity_name

                break

        if activity is None:
            continue

        acc = acc.rename(columns={"gx": "acc_x", "gy": "acc_y", "gz": "acc_z"})

        gyro = gyro.rename(columns={"gx": "gyro_x", "gy": "gyro_y", "gz": "gyro_z"})

        combined = pd.DataFrame()

        combined["timestamp"] = acc["timestamp"].to_numpy()

        combined["acc_x"] = acc["acc_x"].to_numpy()

        combined["acc_y"] = acc["acc_y"].to_numpy()

        combined["acc_z"] = acc["acc_z"].to_numpy()

        gyro_time = gyro["timestamp_corrected"].to_numpy()

        gyro_x = gyro["gyro_x"].to_numpy()

        gyro_y = gyro["gyro_y"].to_numpy()

        gyro_z = gyro["gyro_z"].to_numpy()

        combined["gyro_x"] = np.interp(combined["timestamp"], gyro_time, gyro_x)

        combined["gyro_y"] = np.interp(combined["timestamp"], gyro_time, gyro_y)

        combined["gyro_z"] = np.interp(combined["timestamp"], gyro_time, gyro_z)

        combined["acc_magnitude"] = np.sqrt(
            combined["acc_x"] ** 2 + combined["acc_y"] ** 2 + combined["acc_z"] ** 2
        )

        combined["gyro_magnitude"] = np.sqrt(
            combined["gyro_x"] ** 2 + combined["gyro_y"] ** 2 + combined["gyro_z"] ** 2
        )

        combined["acc_magnitude_change"] = combined["acc_magnitude"].diff().fillna(0)

        combined["gyro_magnitude_change"] = combined["gyro_magnitude"].diff().fillna(0)

        combined["label"] = activity

        combined = combined[
            [
                "timestamp",
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
                "label",
            ]
        ]

        output_filename = f"{timestamp_id}_{activity}.csv"

        output_file = os.path.join(user_output_dir, output_filename)

        combined.to_csv(output_file, index=False)

        print(f"Saved: {user_id}/" f"{output_filename} " f"({len(combined)} samples)")


print("\n==========================================")

print("PROCESSING COMPLETED")

print("==========================================")
