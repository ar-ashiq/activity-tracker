from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


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

        self.data_dir = Path(
            data_dir
        )


        # ----------------------------------------------------
        # Find chunks
        # ----------------------------------------------------

        self.chunk_files = sorted(
            self.data_dir.glob(
                "chunk_*.npz"
            )
        )


        if len(
            self.chunk_files
        ) == 0:

            raise RuntimeError(
                f"No chunks found in "
                f"{self.data_dir}"
            )


        # ----------------------------------------------------
        # Load normalization statistics
        # ----------------------------------------------------

        stats = np.load(
            stats_file
        )


        self.mean = stats[
            "mean"
        ].astype(
            np.float32
        )


        self.std = stats[
            "std"
        ].astype(
            np.float32
        )


        # ----------------------------------------------------
        # Find size of every chunk
        # ----------------------------------------------------

        self.chunk_sizes = []


        for file in self.chunk_files:

            data = np.load(
                file
            )

            size = len(
                data["y"]
            )

            self.chunk_sizes.append(
                size
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
    # NUMBER OF SAMPLES
    # ========================================================

    def __len__(self):

        return int(
            self.cumulative_sizes[-1]
        )


    # ========================================================
    # GET ONE SAMPLE
    # ========================================================

    def __getitem__(
        self,
        index
    ):

        # ----------------------------------------------------
        # Find chunk
        # ----------------------------------------------------

        chunk_index = int(
            np.searchsorted(
                self.cumulative_sizes,
                index,
                side="right"
            )
        )


        # ----------------------------------------------------
        # Find position inside chunk
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Load chunk only when necessary
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Get data
        # ----------------------------------------------------

        X = self.cached_data[
            "X"
        ][local_index]


        y = self.cached_data[
            "y"
        ][local_index]


        # ----------------------------------------------------
        # Normalize
        #
        # X shape:
        #
        # (125, 6)
        #
        # mean/std:
        #
        # (6,)
        # ----------------------------------------------------

        X = (
            X - self.mean
        ) / self.std


        # ----------------------------------------------------
        # Convert to PyTorch
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
# TEST
# ============================================================

dataset = SensorDataset(
    TRAIN_DIR,
    STATS_FILE
)


print(
    "Number of samples:",
    len(dataset)
)


# Get first sample

X, y = dataset[0]


print(
    "X shape:",
    X.shape
)


print(
    "y:",
    y.item()
)


print(
    "X mean:",
    X.mean().item()
)


print(
    "X std:",
    X.std().item()
)