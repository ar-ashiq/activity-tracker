
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score
)
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# CONFIGURATION
# This is the Windows dataset path visible in your screenshot.
# Change DATASET_DIR if your processed_dataset folder is elsewhere.
DATASET_DIR = "/Users/huzaifa/Documents/processed_dataset"
TEST_DIR = DATASET_DIR / "test"
STATS_FILE = DATASET_DIR / "normalization_stats.npz"
CLASS_NAMES_FILE = DATASET_DIR / "class_names.npy"
MODEL_FILE = DATASET_DIR / "best_cnn.pt"

BATCH_SIZE = 64
NUM_CLASSES = 6

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {DEVICE}")


# MODEL ARCHITECTURE MUST MATCH THE TRAINING SCRIPT
class ActivityCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # [batch, time, channels] -> [batch, channels, time]
        x = x.permute(0, 2, 1)
        x = self.features(x)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)


# CHECK REQUIRED FILES
for path in (TEST_DIR, STATS_FILE, CLASS_NAMES_FILE, MODEL_FILE):
    if not path.exists():
        raise FileNotFoundError(
            f"Required path not found:\n{path}\n\n"
            "Check DATASET_DIR near the top of evaluate_test.py."
        )

# LOAD TRAINING-SET NORMALIZATION STATISTICS AND CLASS NAMES
stats = np.load(STATS_FILE)
mean = stats["mean"].astype(np.float32)
std = stats["std"].astype(np.float32)
std = np.where(std == 0, 1.0, std)

class_names = list(np.load(CLASS_NAMES_FILE, allow_pickle=True))
if len(class_names) != NUM_CLASSES:
    raise ValueError(f"Expected 6 class names; found {len(class_names)}: {class_names}")

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

# LOAD THE BEST CHECKPOINT
model = ActivityCNN(num_classes=NUM_CLASSES)
try:
    checkpoint = torch.load(MODEL_FILE, map_location=DEVICE, weights_only=False)
except TypeError:
    # Compatibility with older PyTorch versions.
    checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

if isinstance(checkpoint, dict):
    if "epoch" in checkpoint:
        print(f"\nBest checkpoint epoch: {checkpoint['epoch']}")
    if "val_f1" in checkpoint:
        print(f"Checkpoint validation Macro-F1: {checkpoint['val_f1']:.4f}")
print("Best model loaded successfully.")


# TEST CHUNK DATASET: LOAD ONE CHUNK AT A TIME TO LIMIT RAM USE
class TestChunkDataset(Dataset):
    def __init__(self, chunk_path):
        with np.load(chunk_path) as data:
            self.X = data["X"].astype(np.float32)
            self.y = data["y"].astype(np.int64)

        # Use training statistics; do not calculate new statistics on test data.
        self.X = ((self.X - mean) / std).astype(np.float32, copy=False)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.X[index]),
            torch.tensor(self.y[index], dtype=torch.long)
        )


chunk_files = sorted(TEST_DIR.glob("chunk_*.npz"))
if not chunk_files:
    raise RuntimeError(f"No chunk_*.npz files found in: {TEST_DIR}")

print(f"\nTest chunks found: {len(chunk_files)}")
print(f"Test directory: {TEST_DIR}")

all_predictions = []
all_labels = []

# Evaluate one chunk at a time rather than concatenating all test windows.
with torch.no_grad():
    for chunk_num, chunk_path in enumerate(chunk_files, start=1):
        print(f"Evaluating chunk {chunk_num}/{len(chunk_files)}: {chunk_path.name}")
        test_dataset = TestChunkDataset(chunk_path)
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            predictions = logits.argmax(dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(y_batch.numpy())

        del test_loader, test_dataset

all_predictions = np.asarray(all_predictions, dtype=np.int64)
all_labels = np.asarray(all_labels, dtype=np.int64)

if len(all_labels) == 0:
    raise RuntimeError("No test samples were evaluated.")
if len(all_labels) != len(all_predictions):
    raise RuntimeError("Prediction count does not match label count.")

# OVERALL METRICS
accuracy = accuracy_score(all_labels, all_predictions)
macro_f1 = f1_score(
    all_labels, all_predictions,
    labels=np.arange(NUM_CLASSES), average="macro", zero_division=0
)

print("\n" + "=" * 65)
print("FINAL TEST RESULTS")
print("=" * 65)
print(f"Test samples : {len(all_labels):,}")
print(f"Accuracy     : {accuracy * 100:.2f}%")
print(f"Macro-F1     : {macro_f1:.4f}")

report_text = classification_report(
    all_labels, all_predictions,
    labels=np.arange(NUM_CLASSES),
    target_names=class_names,
    digits=4, zero_division=0
)
print("\nClassification Report:")
print(report_text)

report_dict = classification_report(
    all_labels, all_predictions,
    labels=np.arange(NUM_CLASSES),
    target_names=class_names,
    output_dict=True, zero_division=0
)

report_rows = []
for class_name in class_names:
    row = report_dict[class_name]
    report_rows.append({
        "class": class_name,
        "precision": row["precision"],
        "recall": row["recall"],
        "f1_score": row["f1-score"],
        "support": int(row["support"]),
    })

for avg_name, output_name in (
    ("macro avg", "macro_avg"),
    ("weighted avg", "weighted_avg"),
):
    row = report_dict[avg_name]
    report_rows.append({
        "class": output_name,
        "precision": row["precision"],
        "recall": row["recall"],
        "f1_score": row["f1-score"],
        "support": int(row["support"]),
    })

report_path = DATASET_DIR / "classification_report_test.csv"
pd.DataFrame(report_rows).to_csv(report_path, index=False)

# CONFUSION MATRIX (ROWS = ACTUAL; COLUMNS = PREDICTED)
cm = confusion_matrix(
    all_labels, all_predictions, labels=np.arange(NUM_CLASSES)
)
print("\nConfusion Matrix (rows = actual, columns = predicted):")
print(cm)

cm_csv_path = DATASET_DIR / "confusion_matrix_test.csv"
pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(cm_csv_path)

fig, ax = plt.subplots(figsize=(9, 7))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
    ax=ax, cmap="Blues", values_format="d", xticks_rotation=45
)
ax.set_title("Final Test Confusion Matrix")
plt.tight_layout()
cm_png_path = DATASET_DIR / "confusion_matrix_test.png"
plt.savefig(cm_png_path, dpi=300, bbox_inches="tight")
plt.close(fig)

# ROW-NORMALIZED CONFUSION MATRIX
row_sums = cm.sum(axis=1, keepdims=True)
cm_normalized = np.divide(
    cm.astype(np.float64), row_sums,
    out=np.zeros_like(cm, dtype=np.float64),
    where=row_sums != 0
)

fig, ax = plt.subplots(figsize=(9, 7))
ConfusionMatrixDisplay(cm_normalized, display_labels=class_names).plot(
    ax=ax, cmap="Blues", values_format=".2f", xticks_rotation=45
)
ax.set_title("Normalized Final Test Confusion Matrix")
plt.tight_layout()
cm_norm_png_path = DATASET_DIR / "confusion_matrix_test_normalized.png"
plt.savefig(cm_norm_png_path, dpi=300, bbox_inches="tight")
plt.close(fig)

summary_path = DATASET_DIR / "test_metrics_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("FINAL TEST RESULTS\n")
    f.write("==================\n")
    f.write(f"Test samples: {len(all_labels)}\n")
    f.write(f"Accuracy: {accuracy:.6f} ({accuracy * 100:.2f}%)\n")
    f.write(f"Macro-F1: {macro_f1:.6f}\n")
    f.write(f"Checkpoint: {MODEL_FILE}\n")
    f.write(f"Test directory: {TEST_DIR}\n")

print("\n" + "=" * 65)
print("TEST EVALUATION COMPLETE")
print("=" * 65)
print("Saved files:")
print(f"1. {report_path}")
print(f"2. {cm_csv_path}")
print(f"3. {cm_png_path}")
print(f"4. {cm_norm_png_path}")
print(f"5. {summary_path}")
