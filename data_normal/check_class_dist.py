from pathlib import Path
from collections import Counter

import numpy as np


# ============================================================
# PATH
# ============================================================

DATASET_ROOT = Path(
    "/Users/huzaifa/Documents/processed_dataset"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "lying",
    1: "sitting",
    2: "walking",
    3: "running",
    4: "bicycling",
    5: "standing",
}


# ============================================================
# CHECK EACH SPLIT
# ============================================================

for split in ["train", "validation", "test"]:

    split_dir = DATASET_ROOT / split

    chunk_files = sorted(
        split_dir.glob("chunk_*.npz")
    )

    print()
    print("=" * 70)
    print(split.upper())
    print("=" * 70)

    print(
        "Number of chunks:",
        len(chunk_files)
    )


    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    class_counter = Counter()
    user_counter = Counter()

    total_windows = 0


    # --------------------------------------------------------
    # Process chunks
    # --------------------------------------------------------

    for chunk_file in chunk_files:

        data = np.load(
            chunk_file
        )

        y = data["y"]
        users = data["users"]


        # Count labels
        for label in y:
            class_counter[int(label)] += 1


        # Count users
        for user in users:
            user_counter[str(user)] += 1


        total_windows += len(y)


    # --------------------------------------------------------
    # Print total
    # --------------------------------------------------------

    print(
        "\nTotal windows:",
        total_windows
    )


    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\nClass distribution:")

    for class_id, class_name in CLASS_NAMES.items():

        count = class_counter[class_id]


        if total_windows > 0:

            percentage = (
                count /
                total_windows
                *
                100
            )

        else:

            percentage = 0


        print(
            f"{class_id} "
            f"{class_name:10s} "
            f"{count:10d} "
            f"({percentage:6.2f}%)"
        )


    # --------------------------------------------------------
    # Users
    # --------------------------------------------------------

    print(
        "\nUnique users:"
    )

    print(
        len(user_counter)
    )


    # --------------------------------------------------------
    # Windows per user
    # --------------------------------------------------------

    if user_counter:

        values = list(
            user_counter.values()
        )


        print(
            "\nWindows per user:"
        )

        print(
            "Minimum:",
            min(values)
        )

        print(
            "Maximum:",
            max(values)
        )

        print(
            "Average:",
            sum(values) / len(values)
        )