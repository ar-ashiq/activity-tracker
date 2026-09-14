from pathlib import Path
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_DIR = Path(
    "/Users/huzaifa/Documents/processed_dataset/train"
)

OUTPUT_FILE = Path(
    "/Users/huzaifa/Documents/"
    "processed_dataset/normalization_stats.npz"
)


# ============================================================
# SENSOR CHANNELS
# ============================================================

CHANNELS = [
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz"
]

NUM_CHANNELS = len(CHANNELS)


# ============================================================
# RUNNING STATISTICS
# ============================================================

total_rows = 0

sum_values = np.zeros(
    NUM_CHANNELS,
    dtype=np.float64
)

sum_squared = np.zeros(
    NUM_CHANNELS,
    dtype=np.float64
)


# ============================================================
# FIND TRAINING CHUNKS
# ============================================================

chunk_files = sorted(
    TRAIN_DIR.glob("chunk_*.npz")
)

if not chunk_files:

    raise RuntimeError(
        "No training chunks found."
    )


print(
    f"Found {len(chunk_files)} training chunks."
)


# ============================================================
# PROCESS CHUNK BY CHUNK
# ============================================================

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

    X = data["X"]


    # --------------------------------------------------------
    # X shape:
    #
    # (number_of_windows, 125, 6)
    #
    # Convert to:
    #
    # (all_sensor_rows, 6)
    # --------------------------------------------------------

    X_flat = X.reshape(
        -1,
        NUM_CHANNELS
    ).astype(
        np.float64
    )


    # --------------------------------------------------------
    # Running sum
    # --------------------------------------------------------

    sum_values += X_flat.sum(
        axis=0
    )


    # --------------------------------------------------------
    # Running squared sum
    # --------------------------------------------------------

    sum_squared += (
        X_flat ** 2
    ).sum(
        axis=0
    )


    total_rows += (
        X_flat.shape[0]
    )


# ============================================================
# MEAN
# ============================================================

mean = (
    sum_values /
    total_rows
)


# ============================================================
# VARIANCE
# ============================================================

variance = (
    sum_squared /
    total_rows
) - (
    mean ** 2
)


# Numerical safety
variance = np.maximum(
    variance,
    1e-12
)


# ============================================================
# STANDARD DEVIATION
# ============================================================

std = np.sqrt(
    variance
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 70)
print("NORMALIZATION STATISTICS")
print("=" * 70)

print(
    f"Total sensor rows: {total_rows:,}"
)

print()

for channel, m, s in zip(
    CHANNELS,
    mean,
    std
):

    print(
        f"{channel:>3}  "
        f"mean = {m: .8f}   "
        f"std = {s: .8f}"
    )


# ============================================================
# SAVE
# ============================================================

np.savez(
    OUTPUT_FILE,
    mean=mean,
    std=std
)


print()
print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)