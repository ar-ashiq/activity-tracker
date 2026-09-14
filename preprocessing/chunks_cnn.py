import gc
import numpy as np
import pandas as pd

from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

# Example:
#
# sync/
#   USER_UUID/
#       1449601597/
#           synchronized_25hz.csv
#
SENSOR_ROOT = Path(
    "/Users/huzaifa/Documents/sync"
)


# Example:
#
# labels/
#   USER_UUID.features_labels.csv.gz
#
LABEL_ROOT = Path(
    "/Users/huzaifa/Downloads/ExtraSensory.per_uuid_features_labels"
)


# Where processed ML data will be saved
OUTPUT_ROOT = Path(
    "/Users/huzaifa/Documents/processed_dataset"
)


# ============================================================
# 2. WINDOW SETTINGS
# ============================================================

SAMPLING_RATE = 25

WINDOW_SECONDS = 5

STRIDE_SECONDS = 5

WINDOW_SIZE = (
    SAMPLING_RATE *
    WINDOW_SECONDS
)

STRIDE_SIZE = (
    SAMPLING_RATE *
    STRIDE_SECONDS
)


# Result:
#
# WINDOW_SIZE = 125
# STRIDE_SIZE = 125


# ============================================================
# 3. CHUNK SIZE
# ============================================================

CHUNK_SIZE = 10_000


# ============================================================
# 4. SENSOR COLUMNS
# ============================================================

SENSOR_COLUMNS = [
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
]


# ============================================================
# 5. ACTIVITY LABELS
# ============================================================

ACTIVITY_COLUMNS = {

    "label:LYING_DOWN": "lying",

    "label:SITTING": "sitting",

    "label:FIX_walking": "walking",

    "label:FIX_running": "running",

    "label:BICYCLING": "bicycling",

    "label:OR_standing": "standing",
}


# ============================================================
# 6. ACTIVITY → INTEGER
# ============================================================

ACTIVITY_TO_ID = {

    "lying": 0,

    "sitting": 1,

    "walking": 2,

    "running": 3,

    "bicycling": 4,

    "standing": 5,
}


# ============================================================
# 7. YOUR EXISTING USER SPLIT
# ============================================================
#
# IMPORTANT:
# Put the exact lists from your split_users.py here.
#
# Example:
#
# TRAIN_USERS = [
#     "UUID1",
#     "UUID2",
# ]
#
# ============================================================

TRAIN_USERS = [
    ]


VAL_USERS = [
        "83CF687B-7CEC-434B-9FE8-00C3D5799BE6",
        "A76A5AF5-5A93-4CF2-A16E-62353BB70E8A",
        "11B5EC4D-4133-4289-B475-4E737182A406",
        "D7D20E2E-FC78-405D-B346-DBD3FD8FC92B",
        "1538C99F-BA1E-4EFB-A949-6C7C47701B20",
        "3600D531-0C55-44A7-AE95-A7A38519464E"
    ]


TEST_USERS = [
        "40E170A7-607B-4578-AF04-F021C3B0384A",
        "4E98F91F-4654-42EF-B908-A3389443F2E7",
        "BE3CA5A6-A561-4BBD-B7C9-5DF6805400FC",
        "098A72A5-E3E5-4F54-A152-BBDA0DF7B694",
        "136562B6-95B2-483D-88DC-065F28409FD2",
        "A5A30F76-581E-4757-97A2-957553A2C6AA"
    ]


# ============================================================
# 8. GET USER ID FROM LABEL FILENAME
# ============================================================

def get_user_id_from_label_file(
    label_file
):

    """
    Example:

    0BFC35E2-4817-4865-BFA7-764742302A2D.features_labels.csv.gz

    becomes:

    0BFC35E2-4817-4865-BFA7-764742302A2D
    """

    filename = label_file.name


    # --------------------------------------------------------
    # Remove .csv.gz
    # --------------------------------------------------------

    if filename.endswith(".csv.gz"):

        filename = filename[
            :-len(".csv.gz")
        ]


    # --------------------------------------------------------
    # Remove .features_labels
    # --------------------------------------------------------

    suffix = ".features_labels"


    if filename.endswith(suffix):

        filename = filename[
            :-len(suffix)
        ]


    return filename


# ============================================================
# 9. DISCOVER USERS
# ============================================================

def discover_users():

    print()
    print("=" * 70)
    print("DISCOVERING USERS")
    print("=" * 70)


    # ========================================================
    # SENSOR USERS
    # ========================================================

    sensor_users = {}


    for path in SENSOR_ROOT.iterdir():

        if path.is_dir():

            sensor_users[
                path.name.lower()
            ] = path


    # ========================================================
    # LABEL USERS
    # ========================================================

    label_users = {}


    for path in LABEL_ROOT.glob(
        "*.features_labels.csv.gz"
    ):

        user_id = (
            get_user_id_from_label_file(
                path
            )
        )


        label_users[
            user_id.lower()
        ] = path


    # ========================================================
    # COMMON USERS
    # ========================================================

    common_users = sorted(
        sensor_users.keys()
        &
        label_users.keys()
    )


    # ========================================================
    # SENSOR-ONLY
    # ========================================================

    sensor_only = sorted(
        sensor_users.keys()
        -
        label_users.keys()
    )


    # ========================================================
    # LABEL-ONLY
    # ========================================================

    label_only = sorted(
        label_users.keys()
        -
        sensor_users.keys()
    )


    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print(
        f"\nSensor users : "
        f"{len(sensor_users)}"
    )


    print(
        f"Label users  : "
        f"{len(label_users)}"
    )


    print(
        f"Common users : "
        f"{len(common_users)}"
    )


    print(
        "\nLABEL ONLY "
        "(label exists, sensor missing):"
    )


    if not label_only:

        print("  None")

    else:

        for key in label_only:

            print(
                f"  "
                f"{get_user_id_from_label_file(label_users[key])}"
            )


    print(
        "\nSENSOR ONLY "
        "(sensor exists, label missing):"
    )


    if not sensor_only:

        print("  None")

    else:

        for key in sensor_only:

            print(
                f"  "
                f"{sensor_users[key].name}"
            )


    return (
        sensor_users,
        label_users,
        common_users
    )


# ============================================================
# 10. LOAD LABEL MAP FOR ONE USER
# ============================================================

def load_label_map(
    label_file
):

    """
    Create:

    recording ID → activity

    Example:

    1449601855 → sitting
    1449601918 → sitting
    """

    labels = pd.read_csv(
        label_file
    )


    # --------------------------------------------------------
    # Check timestamp
    # --------------------------------------------------------

    if "timestamp" not in labels.columns:

        raise ValueError(
            f"timestamp column missing in "
            f"{label_file}"
        )


    # --------------------------------------------------------
    # Convert timestamp to recording ID
    # --------------------------------------------------------

    labels["recording_id"] = (

        pd.to_numeric(
            labels["timestamp"],
            errors="coerce"
        )

        .round()

        .astype("Int64")

        .astype(str)
    )


    label_map = {}


    # --------------------------------------------------------
    # Go row by row
    # --------------------------------------------------------

    for _, row in labels.iterrows():

        recording_id = (
            row["recording_id"]
        )


        if recording_id == "<NA>":

            continue


        positive_activities = []


        # ----------------------------------------------------
        # Check six physical activities
        # ----------------------------------------------------

        for column, activity in (
            ACTIVITY_COLUMNS.items()
        ):

            if column not in labels.columns:

                continue


            value = row[column]


            # We only consider 1 as positive.
            #
            # 0   = no label
            # NaN = unavailable

            if (
                pd.notna(value)
                and value == 1
            ):

                positive_activities.append(
                    activity
                )


        # ----------------------------------------------------
        # Require exactly one activity
        # ----------------------------------------------------

        if len(
            positive_activities
        ) == 1:

            label_map[
                recording_id
            ] = positive_activities[0]


    return label_map


# ============================================================
# 11. CHUNK WRITER
# ============================================================

class ChunkWriter:

    def __init__(
        self,
        output_dir
    ):

        self.output_dir = Path(
            output_dir
        )


        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        self.windows = []

        self.labels = []

        self.users = []

        self.recordings = []


        self.chunk_number = 0


    # --------------------------------------------------------
    # Add one sample
    # --------------------------------------------------------

    def add(
        self,
        window,
        label,
        user,
        recording_id
    ):

        self.windows.append(
            window
        )


        self.labels.append(
            ACTIVITY_TO_ID[label]
        )


        self.users.append(
            user
        )


        self.recordings.append(
            recording_id
        )


        if len(
            self.windows
        ) >= CHUNK_SIZE:

            self.flush()


    # --------------------------------------------------------
    # Save chunk
    # --------------------------------------------------------

    def flush(self):

        if len(
            self.windows
        ) == 0:

            return


        # ----------------------------------------------------
        # X
        # ----------------------------------------------------

        X = np.stack(
            self.windows
        ).astype(
            np.float32
        )


        # ----------------------------------------------------
        # y
        # ----------------------------------------------------

        y = np.asarray(
            self.labels,
            dtype=np.int64
        )


        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        users = np.asarray(
            self.users
        )


        recordings = np.asarray(
            self.recordings
        )


        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        output_file = (

            self.output_dir

            /

            f"chunk_{self.chunk_number:04d}.npz"
        )


        np.savez_compressed(

            output_file,

            X=X,

            y=y,

            users=users,

            recordings=recordings
        )


        print(
            f"      SAVED "
            f"{output_file.name} "
            f"X={X.shape}"
        )


        # ----------------------------------------------------
        # Clear
        # ----------------------------------------------------

        self.windows.clear()

        self.labels.clear()

        self.users.clear()

        self.recordings.clear()


        self.chunk_number += 1


        del X
        del y
        del users
        del recordings


        gc.collect()


    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    def finish(self):

        self.flush()


# ============================================================
# 12. PROCESS ONE USER
# ============================================================

def process_user(
    user_name,
    split_name,
    sensor_users,
    label_users,
    writer
):

    print()
    print("=" * 70)

    print(
        f"USER : {user_name}"
    )

    print(
        f"SPLIT: {split_name}"
    )

    print("=" * 70)


    user_key = user_name.lower()


    # --------------------------------------------------------
    # Find sensor directory
    # --------------------------------------------------------

    sensor_user_dir = sensor_users.get(
        user_key
    )


    if sensor_user_dir is None:

        print(
            "  No sensor directory."
        )

        return {
            "user": user_name,
            "split": split_name,
            "recordings": 0,
            "windows": 0,
            "skipped": 0,
            "status": "SENSOR_MISSING"
        }


    # --------------------------------------------------------
    # Find label file
    # --------------------------------------------------------

    label_file = label_users.get(
        user_key
    )


    if label_file is None:

        print(
            "  No label file."
        )

        return {
            "user": user_name,
            "split": split_name,
            "recordings": 0,
            "windows": 0,
            "skipped": 0,
            "status": "LABEL_MISSING"
        }


    # --------------------------------------------------------
    # Load label map
    # --------------------------------------------------------

    print(
        f"  Loading labels:"
        f" {label_file.name}"
    )


    label_map = load_label_map(
        label_file
    )


    print(
        f"  Labelled recordings:"
        f" {len(label_map)}"
    )


    # --------------------------------------------------------
    # Recording folders
    # --------------------------------------------------------

    recording_dirs = sorted(
        [
            path
            for path
            in sensor_user_dir.iterdir()
            if path.is_dir()
        ],
        key=lambda p: p.name
    )


    print(
        f"  Sensor recordings:"
        f" {len(recording_dirs)}"
    )


    processed_recordings = 0

    skipped_recordings = 0

    total_windows = 0


    # ========================================================
    # PROCESS RECORDINGS
    # ========================================================

    for recording_dir in recording_dirs:

        recording_id = (
            recording_dir.name
        )


        # ----------------------------------------------------
        # Get activity
        # ----------------------------------------------------

        activity = label_map.get(
            recording_id
        )


        if activity is None:

            skipped_recordings += 1

            continue


        # ----------------------------------------------------
        # Sensor file
        # ----------------------------------------------------

        sensor_file = (

            recording_dir
            /
            "synchronized_25hz.csv"
        )


        if not sensor_file.exists():

            print(
                f"  Missing:"
                f" {sensor_file}"
            )

            skipped_recordings += 1

            continue


        # ----------------------------------------------------
        # Read sensor data
        # ----------------------------------------------------

        try:

            df = pd.read_csv(

                sensor_file,

                usecols=[
                    "timestamp",
                    "ax",
                    "ay",
                    "az",
                    "gx",
                    "gy",
                    "gz"
                ]
            )


        except Exception as e:

            print(
                f"  ERROR reading:"
                f" {sensor_file}"
            )

            print(
                f"  {e}"
            )

            skipped_recordings += 1

            continue


        # ----------------------------------------------------
        # Remove bad rows
        # ----------------------------------------------------

        df = df.dropna(

            subset=SENSOR_COLUMNS

        ).reset_index(
            drop=True
        )


        # ----------------------------------------------------
        # Sensor values
        # ----------------------------------------------------

        sensor_data = df[
            SENSOR_COLUMNS
        ].to_numpy(
            dtype=np.float32
        )


        # ----------------------------------------------------
        # Check minimum length
        # ----------------------------------------------------

        if (
            len(sensor_data)
            < WINDOW_SIZE
        ):

            skipped_recordings += 1

            del df
            del sensor_data

            continue


        # ----------------------------------------------------
        # CREATE WINDOWS
        # ----------------------------------------------------

        windows_this_recording = 0


        for start in range(

            0,

            len(sensor_data)
            - WINDOW_SIZE
            + 1,

            STRIDE_SIZE
        ):


            end = (

                start
                +
                WINDOW_SIZE
            )


            window = sensor_data[
                start:end
            ]


            # Expected:
            #
            # (125, 6)

            if window.shape != (
                WINDOW_SIZE,
                len(SENSOR_COLUMNS)
            ):

                continue


            writer.add(

                window=window,

                label=activity,

                user=user_name,

                recording_id=recording_id
            )


            windows_this_recording += 1

            total_windows += 1


        print(
            f"  {recording_id}"
            f" -> {activity}"
            f" -> {windows_this_recording}"
            f" windows"
        )


        processed_recordings += 1


        # ----------------------------------------------------
        # Free memory
        # ----------------------------------------------------

        del df

        del sensor_data

        gc.collect()


    # ========================================================
    # USER SUMMARY
    # ========================================================

    print()
    print(
        f"  {user_name} COMPLETE"
    )

    print(
        f"  Processed recordings:"
        f" {processed_recordings}"
    )

    print(
        f"  Skipped recordings:"
        f" {skipped_recordings}"
    )

    print(
        f"  Windows:"
        f" {total_windows}"
    )


    return {

        "user": user_name,

        "split": split_name,

        "recordings": processed_recordings,

        "windows": total_windows,

        "skipped": skipped_recordings,

        "status": "SUCCESS"
    }


# ============================================================
# 13. CHECK USER SPLIT
# ============================================================

def check_split(
    train_users,
    val_users,
    test_users
):

    train_set = {
        u.lower()
        for u in train_users
    }


    val_set = {
        u.lower()
        for u in val_users
    }


    test_set = {
        u.lower()
        for u in test_users
    }


    if train_set & val_set:

        raise ValueError(
            "TRAIN and VALIDATION "
            "contain the same user."
        )


    if train_set & test_set:

        raise ValueError(
            "TRAIN and TEST "
            "contain the same user."
        )


    if val_set & test_set:

        raise ValueError(
            "VALIDATION and TEST "
            "contain the same user."
        )


# ============================================================
# 14. MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "ACTIVITY RECOGNITION "
        "PREPROCESSING"
    )

    print("=" * 70)


    print()
    print(
        f"Sensor root:"
        f" {SENSOR_ROOT}"
    )

    print(
        f"Label root:"
        f" {LABEL_ROOT}"
    )

    print(
        f"Output root:"
        f" {OUTPUT_ROOT}"
    )

    print(
        f"Sampling rate:"
        f" {SAMPLING_RATE} Hz"
    )

    print(
        f"Window:"
        f" {WINDOW_SECONDS} sec"
    )

    print(
        f"Window size:"
        f" {WINDOW_SIZE}"
    )

    print(
        f"Stride:"
        f" {STRIDE_SECONDS} sec"
    )

    print(
        f"Chunk size:"
        f" {CHUNK_SIZE}"
    )


    # ========================================================
    # CHECK PATHS
    # ========================================================

    if not SENSOR_ROOT.exists():

        raise FileNotFoundError(
            f"Sensor root does not exist:\n"
            f"{SENSOR_ROOT}"
        )


    if not LABEL_ROOT.exists():

        raise FileNotFoundError(
            f"Label root does not exist:\n"
            f"{LABEL_ROOT}"
        )


    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # DISCOVER USERS
    # ========================================================

    (
        sensor_users,
        label_users,
        common_users
    ) = discover_users()


    if len(common_users) == 0:

        raise RuntimeError(
            "No common users found."
        )


    # ========================================================
    # CHECK EXISTING SPLIT
    # ========================================================

    check_split(

        TRAIN_USERS,

        VAL_USERS,

        TEST_USERS
    )


    # ========================================================
    # FILTER SPLIT TO COMMON USERS
    # ========================================================

    common_set = set(
        common_users
    )


    train_users = [

        user

        for user in TRAIN_USERS

        if user.lower()
        in common_set
    ]


    val_users = [

        user

        for user in VAL_USERS

        if user.lower()
        in common_set
    ]


    test_users = [

        user

        for user in TEST_USERS

        if user.lower()
        in common_set
    ]


    print()
    print("=" * 70)
    print("FINAL USABLE SPLIT")
    print("=" * 70)


    print(
        f"Train users:"
        f" {len(train_users)}"
    )


    print(
        f"Validation users:"
        f" {len(val_users)}"
    )


    print(
        f"Test users:"
        f" {len(test_users)}"
    )


    # ========================================================
    # CREATE WRITERS
    # ========================================================

    train_writer = ChunkWriter(
        OUTPUT_ROOT / "train"
    )


    val_writer = ChunkWriter(
        OUTPUT_ROOT / "validation"
    )


    test_writer = ChunkWriter(
        OUTPUT_ROOT / "test"
    )


    # ========================================================
    # PROCESS TRAIN
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("TRAIN")
    print("=" * 70)


    train_results = []


    for user in train_users:

        result = process_user(

            user,

            "train",

            sensor_users,

            label_users,

            train_writer
        )


        train_results.append(
            result
        )


    # ========================================================
    # PROCESS VALIDATION
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)


    val_results = []


    for user in val_users:

        result = process_user(

            user,

            "validation",

            sensor_users,

            label_users,

            val_writer
        )


        val_results.append(
            result
        )


    # ========================================================
    # PROCESS TEST
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("TEST")
    print("=" * 70)


    test_results = []


    for user in test_users:

        result = process_user(

            user,

            "test",

            sensor_users,

            label_users,

            test_writer
        )


        test_results.append(
            result
        )


    # ========================================================
    # FINISH CHUNKS
    # ========================================================

    print()
    print(
        "Saving remaining chunks..."
    )


    train_writer.finish()

    val_writer.finish()

    test_writer.finish()


    # ========================================================
    # SAVE CLASS NAMES
    # ========================================================

    class_names = np.array([

        "lying",

        "sitting",

        "walking",

        "running",

        "bicycling",

        "standing"
    ])


    np.save(

        OUTPUT_ROOT
        /
        "class_names.npy",

        class_names
    )


    # ========================================================
    # SAVE USER SPLIT
    # ========================================================

    split_records = []


    for user in train_users:

        split_records.append({

            "user": user,

            "split": "train"
        })


    for user in val_users:

        split_records.append({

            "user": user,

            "split": "validation"
        })


    for user in test_users:

        split_records.append({

            "user": user,

            "split": "test"
        })


    split_df = pd.DataFrame(
        split_records
    )


    split_df.to_csv(

        OUTPUT_ROOT
        /
        "user_split.csv",

        index=False
    )


    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    all_results = (

        train_results
        +
        val_results
        +
        test_results
    )


    summary_df = pd.DataFrame(
        all_results
    )


    summary_df.to_csv(

        OUTPUT_ROOT
        /
        "processing_summary.csv",

        index=False
    )


    # ========================================================
    # FINAL
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)


    print(
        f"Train users:"
        f" {len(train_users)}"
    )


    print(
        f"Validation users:"
        f" {len(val_users)}"
    )


    print(
        f"Test users:"
        f" {len(test_users)}"
    )


    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_ROOT
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()