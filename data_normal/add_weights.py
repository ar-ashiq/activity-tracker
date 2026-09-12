from pathlib import Path

import numpy as np


# ============================================================
# PATH
# ============================================================

TRAIN_DIR = Path(
    "/Users/huzaifa/Documents/processed_dataset/train"
)


# ============================================================
# CLASSES
# ============================================================

CLASS_NAMES = {
    0: "lying",
    1: "sitting",
    2: "walking",
    3: "running",
    4: "bicycling",
    5: "standing",
}


NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# COUNT CLASSES
# ============================================================

counts = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)


chunk_files = sorted(
    TRAIN_DIR.glob("chunk_*.npz")
)


print(
    f"Found {len(chunk_files)} chunks."
)


for i, chunk_file in enumerate(
    chunk_files,
    start=1
):

    print(
        f"[{i}/{len(chunk_files)}] "
        f"{chunk_file.name}"
    )

    data = np.load(
        chunk_file
    )

    y = data["y"]

    counts += np.bincount(
        y,
        minlength=NUM_CLASSES
    )


# ============================================================
# RAW CLASS WEIGHTS
# ============================================================

total = counts.sum()

weights = (
    total /
    (
        NUM_CLASSES *
        counts
    )
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 70)
print("CLASS COUNTS AND WEIGHTS")
print("=" * 70)

for class_id in range(NUM_CLASSES):

    print(
        f"{class_id} "
        f"{CLASS_NAMES[class_id]:10s} "
        f"count = {counts[class_id]:10d} "
        f"weight = {weights[class_id]:.4f}"
    )


# ============================================================
# SAVE
# ============================================================

moderated_weights = np.sqrt(weights)

print("\nRaw weights:")

for class_id in range(NUM_CLASSES):
    print(
        f"{class_id} "
        f"{CLASS_NAMES[class_id]:10s} "
        f"{weights[class_id]:.4f}"
    )


print("\nModerated weights:")

for class_id in range(NUM_CLASSES):
    print(
        f"{class_id} "
        f"{CLASS_NAMES[class_id]:10s} "
        f"{moderated_weights[class_id]:.4f}"
    )

np.save(
    "/Users/huzaifa/Documents/"
    "processed_dataset/class_weights.npy",
    moderated_weights.astype(np.float32)
)


print()
print("Saved:")
print(
    "/Users/huzaifa/Documents/"
    "processed_dataset/class_weights.npy"
)