import os
import numpy as np
import torch

from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    "/Users/huzaifa/Documents/processed_dataset"
)

TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "validation"

STATS_FILE = (
    DATASET_ROOT /
    "normalization_stats.npz"
)

WEIGHTS_FILE = (
    DATASET_ROOT /
    "class_weights.npy"
)

MODEL_FILE = (
    DATASET_ROOT /
    "best_cnn.pt"
)


# ------------------------------------------------------------
# Training settings
# ------------------------------------------------------------

BATCH_SIZE = 64

LEARNING_RATE = 0.001

EPOCHS = 50          # generous upper limit; early stopping decides the real cutoff

PATIENCE = 3         # stop if val Macro-F1 hasn't improved for this many epochs in a row


NUM_CLASSES = 6


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


print("=" * 70)

print("DEVICE")

print("=" * 70)

print(device)


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


        if not self.chunk_files:

            raise RuntimeError(
                f"No chunks found in "
                f"{self.data_dir}"
            )


        # ----------------------------------------------------
        # Normalization statistics
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
        # Chunk sizes
        # ----------------------------------------------------

        self.chunk_sizes = []


        for chunk_file in self.chunk_files:

            data = np.load(
                chunk_file
            )

            self.chunk_sizes.append(
                len(
                    data["y"]
                )
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
    # LENGTH
    # ========================================================

    def __len__(self):

        return int(
            self.cumulative_sizes[-1]
        )


    # ========================================================
    # GET SAMPLE
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
        # Local index
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
        # Load chunk
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
        # Get X
        # ----------------------------------------------------

        X = self.cached_data[
            "X"
        ][local_index]


        # ----------------------------------------------------
        # Get label
        # ----------------------------------------------------

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
        # Convert to tensor
        # ----------------------------------------------------

        X = torch.from_numpy(
            X.astype(
                np.float32
            )
        )


        y = torch.tensor(
            y,
            dtype=torch.long
        )


        return X, y


# ============================================================
# CREATE DATASETS
# ============================================================

print()
print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)


train_dataset = SensorDataset(
    TRAIN_DIR,
    STATS_FILE
)


val_dataset = SensorDataset(
    VAL_DIR,
    STATS_FILE
)


print(
    "Training samples:",
    len(train_dataset)
)


print(
    "Validation samples:",
    len(val_dataset)
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0
)


# ============================================================
# CNN MODEL
# ============================================================

class ActivityCNN(nn.Module):

    def __init__(
        self,
        num_classes=6
    ):

        super().__init__()


        self.features = nn.Sequential(

            # -----------------------------------------------
            # Block 1
            # -----------------------------------------------

            nn.Conv1d(
                in_channels=6,
                out_channels=64,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),


            # -----------------------------------------------
            # Block 2
            # -----------------------------------------------

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),


            # -----------------------------------------------
            # Block 3
            # -----------------------------------------------

            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(256),

            nn.ReLU()
        )


        # ----------------------------------------------------
        # Global average pooling
        # ----------------------------------------------------

        self.global_pool = (
            nn.AdaptiveAvgPool1d(1)
        )


        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            256,
            num_classes
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x
    ):

        # Dataset:
        #
        # [batch, time, channels]
        #
        # Example:
        #
        # [64, 125, 6]
        #

        x = x.permute(
            0,
            2,
            1
        )


        # Now:
        #
        # [64, 6, 125]


        x = self.features(x)


        x = self.global_pool(x)


        # [batch, 256, 1]
        #
        # Remove last dimension

        x = x.squeeze(
            -1
        )


        # [batch, 256]

        x = self.classifier(x)


        # [batch, 6]

        return x


# ============================================================
# CREATE MODEL
# ============================================================

model = ActivityCNN(
    NUM_CLASSES
)


model = model.to(
    device
)


print()
print("=" * 70)
print("MODEL")
print("=" * 70)

print(model)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_weights = np.load(
    WEIGHTS_FILE
)


print()
print("Class weights:")

print(
    class_weights
)


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=device
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()


    total_loss = 0.0

    correct = 0

    total = 0


    for batch_index, (
        X,
        y
    ) in enumerate(
        train_loader,
        start=1
    ):


        # ----------------------------------------------------
        # Move to device
        # ----------------------------------------------------

        X = X.to(
            device
        )

        y = y.to(
            device
        )


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits = model(
            X
        )


        # ----------------------------------------------------
        # Calculate loss
        # ----------------------------------------------------

        loss = criterion(
            logits,
            y
        )


        # ----------------------------------------------------
        # Clear old gradients
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Update weights
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_loss += (
            loss.item()
            *
            X.size(0)
        )


        predictions = (
            logits.argmax(
                dim=1
            )
        )


        correct += (
            predictions == y
        ).sum().item()


        total += X.size(0)


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if batch_index % 500 == 0:

            print(
                f"  Batch "
                f"{batch_index} "
                f"/ "
                f"{len(train_loader)}"
            )


    epoch_loss = (
        total_loss /
        total
    )


    epoch_accuracy = (
        correct /
        total
    )


    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate():

    model.eval()


    total_loss = 0.0

    correct = 0

    total = 0


    # Predictions and labels
    # for macro F1 later

    all_predictions = []

    all_labels = []


    with torch.no_grad():

        for X, y in val_loader:

            X = X.to(
                device
            )

            y = y.to(
                device
            )


            logits = model(
                X
            )


            loss = criterion(
                logits,
                y
            )


            total_loss += (
                loss.item()
                *
                X.size(0)
            )


            predictions = (
                logits.argmax(
                    dim=1
                )
            )


            correct += (
                predictions == y
            ).sum().item()


            total += X.size(0)


            all_predictions.extend(
                predictions.cpu().numpy()
            )


            all_labels.extend(
                y.cpu().numpy()
            )


    val_loss = (
        total_loss /
        total
    )


    val_accuracy = (
        correct /
        total
    )


    return (
        val_loss,
        val_accuracy,
        np.array(all_labels),
        np.array(all_predictions)
    )


# ============================================================
# SIMPLE MACRO F1
# ============================================================

def macro_f1(
    y_true,
    y_pred
):

    f1_scores = []


    for class_id in range(
        NUM_CLASSES
    ):

        true_positive = np.sum(
            (
                y_true == class_id
            )
            &
            (
                y_pred == class_id
            )
        )


        false_positive = np.sum(
            (
                y_true != class_id
            )
            &
            (
                y_pred == class_id
            )
        )


        false_negative = np.sum(
            (
                y_true == class_id
            )
            &
            (
                y_pred != class_id
            )
        )


        precision = (

            true_positive /
            (
                true_positive
                +
                false_positive
            )

            if (
                true_positive
                +
                false_positive
            ) > 0

            else 0.0
        )


        recall = (

            true_positive /
            (
                true_positive
                +
                false_negative
            )

            if (
                true_positive
                +
                false_negative
            ) > 0

            else 0.0
        )


        if (
            precision
            +
            recall
        ) > 0:

            f1 = (

                2 *
                precision *
                recall
                /
                (
                    precision
                    +
                    recall
                )
            )

        else:

            f1 = 0.0


        f1_scores.append(
            f1
        )


    return np.mean(
        f1_scores
    )


# ============================================================
# TRAINING LOOP (with best-epoch tracking + early stopping)
# ============================================================

best_val_f1 = -1.0

best_epoch = -1

epochs_without_improvement = 0


print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(
    1,
    EPOCHS + 1
):


    print()
    print(
        f"Epoch "
        f"{epoch}/{EPOCHS}"
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_loss, train_accuracy = (
        train_one_epoch()
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    (
        val_loss,
        val_accuracy,
        y_true,
        y_pred
    ) = validate()


    val_f1 = macro_f1(
        y_true,
        y_pred
    )


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()

    print(
        f"Train loss     : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train accuracy : "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Val loss       : "
        f"{val_loss:.4f}"
    )

    print(
        f"Val accuracy   : "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Val Macro-F1   : "
        f"{val_f1:.4f}"
    )


    # --------------------------------------------------------
    # Save best model / track patience
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        best_epoch = epoch

        epochs_without_improvement = 0


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "epoch":
                    epoch,

                "val_f1":
                    val_f1,
            },
            MODEL_FILE
        )


        print(
            ">>> Best model saved."
        )

    else:

        epochs_without_improvement += 1

        print(
            f">>> No improvement for "
            f"{epochs_without_improvement} epoch(s)."
        )


        if epochs_without_improvement >= PATIENCE:

            print()
            print(
                f"Early stopping triggered "
                f"(no improvement for {PATIENCE} epochs)."
            )

            break


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Best validation Macro-F1:",
    best_val_f1
)

print(
    "Model saved to:",
    MODEL_FILE
)