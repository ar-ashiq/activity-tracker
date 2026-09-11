import os
import csv
import numpy as np
from pathlib import Path


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

OUTPUT_ROOT = Path(
    "/Users/huzaifa/Documents/sync"
)

# ------------------------------------------------------------
# Desired output frequency
# ------------------------------------------------------------

TARGET_FS = 25.0

# 25 Hz means:
#
#     1 / 25 = 0.04 seconds
#
# So consecutive output timestamps are 40 ms apart.

DT = 1.0 / TARGET_FS


# ============================================================
# LOAD SENSOR FILE
# ============================================================

def load_sensor_file(file_path):
    """
    Load one ACC or Gyro file.

    File format:

        timestamp x y z

    Returns:
        NumPy array with shape (N, 4)

        column 0 -> timestamp
        column 1 -> x
        column 2 -> y
        column 3 -> z
    """

    try:

        data = np.loadtxt(
            file_path,
            dtype=np.float64
        )

    except Exception as e:

        raise RuntimeError(
            f"Could not read file:\n"
            f"{file_path}\n"
            f"Error: {e}"
        )

    # --------------------------------------------------------
    # If the file has only one row, np.loadtxt() returns a
    # 1-dimensional array.
    #
    # Convert it to:
    #
    #     (1, 4)
    # --------------------------------------------------------

    if data.ndim == 1:

        data = data.reshape(1, -1)


    # --------------------------------------------------------
    # We need at least 4 columns:
    #
    # timestamp, x, y, z
    # --------------------------------------------------------

    if data.shape[1] < 4:

        raise ValueError(
            f"Invalid file format:\n{file_path}"
        )


    # --------------------------------------------------------
    # Keep only:
    #
    # timestamp, x, y, z
    # --------------------------------------------------------

    data = data[:, :4]


    # --------------------------------------------------------
    # Remove NaN / Inf rows
    # --------------------------------------------------------

    valid = np.all(
        np.isfinite(data),
        axis=1
    )

    data = data[valid]


    # --------------------------------------------------------
    # Make sure timestamps are sorted.
    #
    # np.interp() requires sorted timestamps.
    # --------------------------------------------------------

    order = np.argsort(
        data[:, 0]
    )

    data = data[order]


    return data


# ============================================================
# EXTRACT RECORDING ID FROM FILENAME
# ============================================================

def get_recording_id(filename):
    """
    Extract the timestamp / recording ID.

    Example:

        1449601597.m_raw_acc.dat

    becomes:

        1449601597

    and:

        1449601597.m_proc_gyro.dat

    also becomes:

        1449601597
    """

    if filename.endswith(".m_raw_acc.dat"):

        return filename[
            :-len(".m_raw_acc.dat")
        ]


    if filename.endswith(".m_proc_gyro.dat"):

        return filename[
            :-len(".m_proc_gyro.dat")
        ]


    return None


# ============================================================
# FIND ALL ACC FILES FOR ONE USER
# ============================================================

def find_acc_files(user_dir):
    """
    Find all ACC files directly inside one user's folder.

    Example:

        USER_ID/
            1449601597.m_raw_acc.dat
            1449601658.m_raw_acc.dat
            ...

    Returns:

        {
            "1449601597": Path(...),
            "1449601658": Path(...),
            ...
        }
    """

    acc_files = {}


    # --------------------------------------------------------
    # Since the files are directly inside the user folder,
    # there is NO need for rglob().
    #
    # We just scan the directory.
    # --------------------------------------------------------

    with os.scandir(user_dir) as entries:

        for entry in entries:

            if not entry.is_file():
                continue


            filename = entry.name


            if not filename.endswith(
                ".m_raw_acc.dat"
            ):
                continue


            recording_id = get_recording_id(
                filename
            )


            if recording_id is None:
                continue


            acc_files[recording_id] = Path(
                entry.path
            )


    return acc_files


# ============================================================
# FIND ALL GYRO FILES FOR ONE USER
# ============================================================

def find_gyro_files(user_dir):
    """
    Find all Gyro files directly inside one user's folder.

    Example:

        USER_ID/
            1449601597.m_proc_gyro.dat
            1449601658.m_proc_gyro.dat
            ...

    Returns:

        {
            "1449601597": Path(...),
            "1449601658": Path(...),
            ...
        }
    """

    gyro_files = {}


    with os.scandir(user_dir) as entries:

        for entry in entries:

            if not entry.is_file():
                continue


            filename = entry.name


            if not filename.endswith(
                ".m_proc_gyro.dat"
            ):
                continue


            recording_id = get_recording_id(
                filename
            )


            if recording_id is None:
                continue


            gyro_files[recording_id] = Path(
                entry.path
            )


    return gyro_files


# ============================================================
# PROCESS ONE ACC + GYRO PAIR
# ============================================================

def process_recording(
    recording_id,
    acc_file,
    gyro_file,
    output_file
):
    """
    Synchronize ONE ACC + Gyro recording.

    Steps:

        1. Load ACC
        2. Load Gyro
        3. Find common time interval
        4. Create 25 Hz timeline
        5. Interpolate ACC
        6. Interpolate Gyro
        7. Save output
    """


    # ========================================================
    # STEP 0: SKIP IF ALREADY PROCESSED
    # ========================================================

    if output_file.exists():

        return "SKIPPED"


    try:

        # ====================================================
        # STEP 1: LOAD ACC
        # ====================================================

        acc = load_sensor_file(
            acc_file
        )


        # ====================================================
        # STEP 2: LOAD GYRO
        # ====================================================

        gyro = load_sensor_file(
            gyro_file
        )


        # ====================================================
        # STEP 3: CHECK FILES
        # ====================================================

        if len(acc) < 2:

            return "ACC_TOO_SHORT"


        if len(gyro) < 2:

            return "GYRO_TOO_SHORT"


        # ====================================================
        # STEP 4: GET TIMESTAMP ARRAYS
        # ====================================================

        acc_time = acc[:, 0]
        gyro_time = gyro[:, 0]


        # ====================================================
        # STEP 5: FIND OVERLAPPING TIME
        # ====================================================
        #
        # Example:
        #
        # ACC:
        #
        # 267868.298 ------------------------- 267891.000
        #
        #
        # GYRO:
        #
        #       267868.530 ------------------- 267888.500
        #
        #
        # Therefore the common region is:
        #
        #       267868.530 ------------------- 267888.500
        #
        # We do NOT require an exact timestamp to exist in
        # both files.
        # ====================================================

        start_time = max(
            acc_time[0],
            gyro_time[0]
        )

        end_time = min(
            acc_time[-1],
            gyro_time[-1]
        )


        # ----------------------------------------------------
        # If there is no common time range
        # ----------------------------------------------------

        if start_time >= end_time:

            return "NO_OVERLAP"


        # ====================================================
        # STEP 6: CREATE 25 Hz COMMON TIMELINE
        # ====================================================
        #
        # 25 Hz:
        #
        #     1 / 25 = 0.04 sec
        #
        # So:
        #
        #     t
        #     t + 0.04
        #     t + 0.08
        #     t + 0.12
        #     ...
        #
        # This is our NEW common timeline.
        # ====================================================

        common_time = np.arange(
            start_time,
            end_time,
            DT,
            dtype=np.float64
        )


        if len(common_time) < 2:

            return "TOO_SHORT"


        # ====================================================
        # STEP 7: INTERPOLATE ACC
        # ====================================================
        #
        # For every timestamp in common_time, find the
        # estimated ACC value at that exact time.
        #
        # We do it separately for:
        #
        #     ax
        #     ay
        #     az
        # ====================================================

        ax = np.interp(
            common_time,
            acc_time,
            acc[:, 1]
        )

        ay = np.interp(
            common_time,
            acc_time,
            acc[:, 2]
        )

        az = np.interp(
            common_time,
            acc_time,
            acc[:, 3]
        )


        # ====================================================
        # STEP 8: INTERPOLATE GYRO
        # ====================================================
        #
        # Same thing for:
        #
        #     gx
        #     gy
        #     gz
        #
        # using the SAME common_time.
        # ====================================================

        gx = np.interp(
            common_time,
            gyro_time,
            gyro[:, 1]
        )

        gy = np.interp(
            common_time,
            gyro_time,
            gyro[:, 2]
        )

        gz = np.interp(
            common_time,
            gyro_time,
            gyro[:, 3]
        )


        # ====================================================
        # STEP 9: COMBINE EVERYTHING
        # ====================================================

        synchronized = np.column_stack([
            common_time,
            ax,
            ay,
            az,
            gx,
            gy,
            gz
        ])


        # ====================================================
        # STEP 10: CREATE OUTPUT DIRECTORY
        # ====================================================

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        # ====================================================
        # STEP 11: WRITE TO TEMP FILE
        # ====================================================
        #
        # Example:
        #
        # synchronized_25hz.tmp
        #
        # Once completely written, it is renamed to:
        #
        # synchronized_25hz.csv
        #
        # This prevents incomplete files from being treated
        # as successfully processed files.
        # ====================================================

        temp_file = output_file.with_suffix(
            ".tmp"
        )


        np.savetxt(
            temp_file,
            synchronized,
            delimiter=",",
            header="timestamp,ax,ay,az,gx,gy,gz",
            comments="",
            fmt="%.10f"
        )


        # ----------------------------------------------------
        # Rename temporary file to final filename
        # ----------------------------------------------------

        os.replace(
            temp_file,
            output_file
        )


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


        # ----------------------------------------------------
        # Delete temporary file if it exists.
        # ----------------------------------------------------

        temp_file = output_file.with_suffix(
            ".tmp"
        )

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
    print(f"Output root : {OUTPUT_ROOT}")
    print(f"Target rate : {TARGET_FS} Hz")
    print(f"Interval    : {DT} seconds")
    print()


    # ========================================================
    # CHECK ROOT DIRECTORIES
    # ========================================================

    if not ACC_ROOT.exists():

        raise FileNotFoundError(
            f"ACC directory does not exist:\n"
            f"{ACC_ROOT}"
        )


    if not GYRO_ROOT.exists():

        raise FileNotFoundError(
            f"GYRO directory does not exist:\n"
            f"{GYRO_ROOT}"
        )


    # ========================================================
    # CREATE OUTPUT ROOT
    # ========================================================

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # FIND USERS
    # ========================================================

    print("Finding users...")


    acc_users = {}

    with os.scandir(ACC_ROOT) as entries:

        for entry in entries:

            if entry.is_dir():

                acc_users[
                    entry.name
                ] = Path(entry.path)


    gyro_users = {}

    with os.scandir(GYRO_ROOT) as entries:

        for entry in entries:

            if entry.is_dir():

                gyro_users[
                    entry.name
                ] = Path(entry.path)


    # ========================================================
    # MATCH USERS
    # ========================================================

    common_users = sorted(
        acc_users.keys()
        &
        gyro_users.keys()
    )


    print()
    print(
        f"ACC users    : {len(acc_users)}"
    )

    print(
        f"GYRO users   : {len(gyro_users)}"
    )

    print(
        f"Common users : {len(common_users)}"
    )


    # ========================================================
    # STATISTICS
    # ========================================================

    total_recordings = 0
    processed = 0
    skipped = 0
    errors = 0
    no_overlap = 0
    acc_short = 0
    gyro_short = 0
    too_short = 0


    # ========================================================
    # PROCESSING LOG
    # ========================================================

    log_file = (
        OUTPUT_ROOT /
        "processing_log.csv"
    )


    log_exists = log_file.exists()


    log_handle = open(
        log_file,
        "a",
        newline=""
    )


    log_writer = csv.writer(
        log_handle
    )


    if not log_exists:

        log_writer.writerow([
            "user",
            "recording_id",
            "acc_file",
            "gyro_file",
            "status"
        ])


    # ========================================================
    # PROCESS ALL USERS
    # ========================================================

    try:

        for user_index, user_name in enumerate(
            common_users,
            start=1
        ):

            print()
            print("=" * 70)

            print(
                f"[USER {user_index}/"
                f"{len(common_users)}]"
            )

            print(
                user_name
            )

            print("=" * 70)


            acc_user_dir = acc_users[
                user_name
            ]

            gyro_user_dir = gyro_users[
                user_name
            ]


            # =================================================
            # FIND ACC FILES
            # =================================================

            print(
                "Finding ACC files..."
            )

            acc_files = find_acc_files(
                acc_user_dir
            )


            # =================================================
            # FIND GYRO FILES
            # =================================================

            print(
                "Finding Gyro files..."
            )

            gyro_files = find_gyro_files(
                gyro_user_dir
            )


            # =================================================
            # MATCH FILES
            # =================================================
            #
            # Example:
            #
            # ACC:
            # 1449601597
            #
            # GYRO:
            # 1449601597
            #
            # => MATCH
            # =================================================

            common_recording_ids = sorted(
                acc_files.keys()
                &
                gyro_files.keys()
            )


            print()
            print(
                f"ACC recordings  : "
                f"{len(acc_files)}"
            )

            print(
                f"GYRO recordings : "
                f"{len(gyro_files)}"
            )

            print(
                f"Matched         : "
                f"{len(common_recording_ids)}"
            )


            # =================================================
            # PROCESS EACH MATCHED FILE
            # =================================================

            for recording_index, recording_id in enumerate(
                common_recording_ids,
                start=1
            ):

                total_recordings += 1


                acc_file = acc_files[
                    recording_id
                ]

                gyro_file = gyro_files[
                    recording_id
                ]


                # ------------------------------------------------
                # Output:
                #
                # sync/
                #     USER_ID/
                #         1449601597/
                #             synchronized_25hz.csv
                # ------------------------------------------------

                output_file = (
                    OUTPUT_ROOT
                    / user_name
                    / recording_id
                    / "synchronized_25hz.csv"
                )


                # ------------------------------------------------
                # Progress
                # ------------------------------------------------

                print(
                    f"[{recording_index}/"
                    f"{len(common_recording_ids)}] "
                    f"{recording_id}",
                    end=" "
                )


                # =================================================
                # PROCESS
                # =================================================

                result = process_recording(
                    recording_id,
                    acc_file,
                    gyro_file,
                    output_file
                )


                print(
                    f"-> {result}"
                )


                # =================================================
                # UPDATE STATISTICS
                # =================================================

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

                else:

                    errors += 1


                # =================================================
                # WRITE LOG
                # =================================================

                log_writer.writerow([
                    user_name,
                    recording_id,
                    str(acc_file),
                    str(gyro_file),
                    result
                ])


                # Save log immediately
                log_handle.flush()


    finally:

        log_handle.close()


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Total recordings : "
        f"{total_recordings}"
    )

    print(
        f"Processed        : "
        f"{processed}"
    )

    print(
        f"Skipped          : "
        f"{skipped}"
    )

    print(
        f"No overlap       : "
        f"{no_overlap}"
    )

    print(
        f"ACC too short    : "
        f"{acc_short}"
    )

    print(
        f"Gyro too short   : "
        f"{gyro_short}"
    )

    print(
        f"Too short        : "
        f"{too_short}"
    )

    print(
        f"Errors           : "
        f"{errors}"
    )

    print()
    print(
        f"Processing log: "
        f"{log_file}"
    )

    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()