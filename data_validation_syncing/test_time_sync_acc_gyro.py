import os
import pandas as pd
import numpy as np

ACC_DIR = "/Users/ashiqar/chore/ubiquitous/Codes/datasets/acc_outputs"
GYRO_DIR = "/Users/ashiqar/chore/ubiquitous/Codes/datasets/gyro_outputs"

for user_id in sorted(os.listdir(ACC_DIR)):

    acc_user_dir = os.path.join(ACC_DIR, user_id)
    gyro_user_dir = os.path.join(GYRO_DIR, user_id)

    if not os.path.isdir(acc_user_dir):
        continue

    if not os.path.isdir(gyro_user_dir):
        print(f"{user_id}: Gyro folder not found")
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
            print(f"{user_id}: Gyro file missing -> {gyro_filename}")
            continue

        acc = pd.read_csv(acc_file)
        gyro = pd.read_csv(gyro_file)

        n = min(len(acc), len(gyro))

        if n == 0:
            continue

        shift = (
            gyro["timestamp"].iloc[:n].to_numpy() - acc["timestamp"].iloc[:n].to_numpy()
        )

        file_median = np.median(shift)

        file_medians.append(file_median)

    if file_medians:

        user_average_median = np.mean(file_medians)

        print(
            f"{user_id} -> "
            f"Average of median shifts = "
            f"{user_average_median:.9f} seconds"
        )
