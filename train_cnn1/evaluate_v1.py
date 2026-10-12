import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score
)

import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = "/Users/huzaifa/Documents/processed_dataset"

BATCH_SIZE = 64
NUM_CLASSES = 6

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {DEVICE}")


# CNN MODEL


class ActivityCNN(nn.Module):

    def __init__(self, num_classes=6):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                in_channels=6,
                out_channels=64,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):

        # Input:
        # [batch, time, channels]
        # Example: [64, 125, 6]

        # Conv1D expects:
        # [batch, channels, time]

        x = x.permute(0, 2, 1)

        x = self.features(x)

        x = x.squeeze(-1)

        x = self.classifier(x)

        return x


# DATASET


class ValidationDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# LOAD NORMALIZATION STATISTICS

stats_path = os.path.join(
    DATASET_DIR,
    "normalization_stats.npz"
)

stats = np.load(stats_path)

mean = stats["mean"]
std = stats["std"]

print("\nNormalization statistics loaded.")

print("Mean:", mean)
print("Std :", std)


# LOAD CLASS NAMES


class_names_path = os.path.join(
    DATASET_DIR,
    "class_names.npy"
)

class_names = np.load(
    class_names_path,
    allow_pickle=True
)

class_names = list(class_names)

print("\nClasses:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# LOAD VALIDATION CHUNKS


validation_dir = os.path.join(
    DATASET_DIR,
    "validation"
)

chunk_files = sorted([
    os.path.join(validation_dir, f)
    for f in os.listdir(validation_dir)
    if f.endswith(".npz")
])

print(f"\nValidation chunks found: {len(chunk_files)}")


X_list = []
y_list = []

for i, file_path in enumerate(chunk_files):

    data = np.load(file_path)

    X = data["X"]
    y = data["y"]

    # --------------------------------------------------------
    # NORMALIZE USING TRAINING STATISTICS
    # --------------------------------------------------------

    X = (X - mean) / std

    X_list.append(X)
    y_list.append(y)

    print(
        f"Loaded chunk {i + 1}/{len(chunk_files)}"
    )


# ============================================================
# COMBINE ALL VALIDATION DATA
# ============================================================

X_val = np.concatenate(X_list, axis=0)
y_val = np.concatenate(y_list, axis=0)

print("\nValidation data:")
print("X shape:", X_val.shape)
print("y shape:", y_val.shape)

print("Total validation samples:", len(y_val))


# ============================================================
# DATALOADER
# ============================================================

val_dataset = ValidationDataset(
    X_val,
    y_val
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# LOAD BEST MODEL

model = ActivityCNN(
    num_classes=NUM_CLASSES
)

model_path = os.path.join(
    DATASET_DIR,
    "best_cnn.pt"
)

checkpoint = torch.load(
    model_path,
    map_location=DEVICE,
    weights_only=False
)

# Handles both:
# 1. directly saved state_dict
# 2. checkpoint dictionaries

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)

model.eval()

print("\nBest model loaded successfully.")


# ============================================================
# EVALUATION
# ============================================================

all_predictions = []
all_labels = []

correct = 0
total = 0

with torch.no_grad():

    for X_batch, y_batch in val_loader:

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        outputs = model(X_batch)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == y_batch
        ).sum().item()

        total += y_batch.size(0)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            y_batch.cpu().numpy()
        )


# Convert to NumPy
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)


# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro"
)

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print(f"\nAccuracy : {accuracy * 100:.2f}%")
print(f"Macro-F1 : {macro_f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    labels=np.arange(NUM_CLASSES),
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("\nClassification Report:")
print(report)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_dict = classification_report(
    all_labels,
    all_predictions,
    labels=np.arange(NUM_CLASSES),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_rows = []

for class_name in class_names:

    row = report_dict[class_name]

    report_rows.append({
        "class": class_name,
        "precision": row["precision"],
        "recall": row["recall"],
        "f1_score": row["f1-score"],
        "support": int(row["support"])
    })


# Add macro and weighted averages
report_rows.append({
    "class": "macro_avg",
    "precision": report_dict["macro avg"]["precision"],
    "recall": report_dict["macro avg"]["recall"],
    "f1_score": report_dict["macro avg"]["f1-score"],
    "support": int(report_dict["macro avg"]["support"])
})

report_rows.append({
    "class": "weighted_avg",
    "precision": report_dict["weighted avg"]["precision"],
    "recall": report_dict["weighted avg"]["recall"],
    "f1_score": report_dict["weighted avg"]["f1-score"],
    "support": int(report_dict["weighted avg"]["support"])
})


import pandas as pd

report_df = pd.DataFrame(report_rows)

report_path = os.path.join(
    DATASET_DIR,
    "classification_report.csv"
)

report_df.to_csv(
    report_path,
    index=False
)

print(
    f"\nClassification report saved to:\n"
    f"{report_path}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=np.arange(NUM_CLASSES)
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# SAVE RAW CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_csv_path = os.path.join(
    DATASET_DIR,
    "confusion_matrix.csv"
)

cm_df.to_csv(
    cm_csv_path
)

print(
    f"\nConfusion matrix CSV saved to:\n"
    f"{cm_csv_path}"
)



# PLOT CONFUSION MATRIX


fig, ax = plt.subplots(
    figsize=(9, 7)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    xticks_rotation=45
)

ax.set_title(
    "Validation Confusion Matrix"
)

plt.tight_layout()


cm_png_path = os.path.join(
    DATASET_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_png_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Confusion matrix image saved to:\n"
    f"{cm_png_path}"
)


# NORMALIZED CONFUSION MATRIX

cm_normalized = (
    cm.astype(float)
    /
    cm.sum(axis=1, keepdims=True)
)

# Handle classes with zero samples
cm_normalized = np.nan_to_num(
    cm_normalized
)


fig, ax = plt.subplots(
    figsize=(9, 7)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_normalized,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format=".2f",
    xticks_rotation=45
)

ax.set_title(
    "Normalized Validation Confusion Matrix"
)

plt.tight_layout()


cm_norm_png_path = os.path.join(
    DATASET_DIR,
    "confusion_matrix_normalized.png"
)

plt.savefig(
    cm_norm_png_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Normalized confusion matrix saved to:\n"
    f"{cm_norm_png_path}"
)


# FINAL

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print("\nSaved files:")

print("1.", model_path)
print("2.", report_path)
print("3.", cm_csv_path)
print("4.", cm_png_path)
print("5.", cm_norm_png_path)