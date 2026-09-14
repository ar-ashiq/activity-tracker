from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


# ============================================================
# PATHS
# ============================================================

TRAIN_DIR = Path(
    "/Users/huzaifa/Documents/processed_dataset/train"
)

STATS_FILE = Path(
    "/Users/huzaifa/Documents/processed_dataset/"
    "normalization_stats.npz"
)


# ============================================================
# DATASET
# ============================================================

class SensorDataset(Dataset):

    def __init__(
        self,
        data_dir,
        stats_file
    ):

        self.data_dir = Path(data_dir)

        # Find all chunks
        self.chunk_files = sorted(
            self.data_dir.glob(
                "chunk_*.npz"
            )
        )

        if not self.chunk_files:
            raise RuntimeError(
                f"No chunks found in {self.data_dir}"
            )

        # Load normalization statistics
        stats = np.load(stats_file)

        self.mean = stats["mean"].astype(
            np.float32
        )

        self.std = stats["std"].astype(
            np.float32
        )

        # ----------------------------------------------------
        # Find size of each chunk
        # ----------------------------------------------------

        self.chunk_sizes = []

        for chunk_file in self.chunk_files:

            data = np.load(chunk_file)

            self.chunk_sizes.append(
                len(data["y"])
            )

        # ----------------------------------------------------
        # Cumulative sizes
        # ----------------------------------------------------

        self.cumulative_sizes = np.cumsum(
            self.chunk_sizes
        )

        # ----------------------------------------------------
        # Cache
        # ----------------------------------------------------

        self.cached_chunk_index = None
        self.cached_data = None


    # ========================================================
    # TOTAL NUMBER OF SAMPLES
    # ========================================================

    def __len__(self):

        return int(
            self.cumulative_sizes[-1]
        )


    # ========================================================
    # GET ONE SAMPLE
    # ========================================================

    def __getitem__(self, index):

        # Find which chunk contains this index
        chunk_index = int(
            np.searchsorted(
                self.cumulative_sizes,
                index,
                side="right"
            )
        )

        # Find position inside that chunk
        if chunk_index == 0:

            local_index = index

        else:

            local_index = (
                index
                -
                self.cumulative_sizes[
                    chunk_index - 1
                ]
            )

        # Load chunk if not already cached
        if (
            self.cached_chunk_index
            != chunk_index
        ):

            self.cached_data = np.load(
                self.chunk_files[
                    chunk_index
                ]
            )

            self.cached_chunk_index = (
                chunk_index
            )

        # Get one window
        X = self.cached_data[
            "X"
        ][local_index]

        # Get label
        y = self.cached_data[
            "y"
        ][local_index]

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        X = (
            X - self.mean
        ) / self.std

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        X = torch.tensor(
            X,
            dtype=torch.float32
        )

        y = torch.tensor(
            y,
            dtype=torch.long
        )

        return X, y


# ============================================================
# CREATE DATASET
# ============================================================

train_dataset = SensorDataset(
    TRAIN_DIR,
    STATS_FILE
)


print(
    "Total samples:",
    len(train_dataset)
)


# ============================================================
# CREATE DATALOADER
# ============================================================

BATCH_SIZE = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# ============================================================
# GET ONE BATCH
# ============================================================

X_batch, y_batch = next(
    iter(train_loader)
)


# ============================================================
# PRINT INFORMATION
# ============================================================

print(
    "\nBatch X shape:",
    X_batch.shape
)

print(
    "Batch y shape:",
    y_batch.shape
)

print(
    "\nFirst 10 labels:"
)

print(
    y_batch[:10]
)

print(
    "\nX min:",
    X_batch.min().item()
)

print(
    "X max:",
    X_batch.max().item()
)

print(
    "X mean:",
    X_batch.mean().item()
)

print(
    "X std:",
    X_batch.std().item()
)