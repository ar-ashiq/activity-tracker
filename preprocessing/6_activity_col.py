import pandas as pd
import os
import glob

# Folder containing your CSV files with the labels
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# folder_path = os.path.join(project_root, "datasets/ExtraSensory.per_uuid_features_labels")
folder_path = "/Users/huzaifa/Downloads/ExtraSensory.per_uuid_features_labels/"

# Phrases to search for in column headings
# Columns we want to KEEP
columns_to_keep = [
    "timestamp",
    "label:LYING_DOWN",
    "label:SITTING",
    "label:FIX_walking",
    "label:FIX_running",
    "label:BICYCLING",
    "label:OR_standing"
]


def keep_required_columns(input_folder=folder_path):

    # Find both .csv and .csv.gz files
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    gz_files = glob.glob(os.path.join(input_folder, "*.csv.gz"))

    files = csv_files + gz_files

    print(f"Found {len(files)} files.")

    for file in files:

        print(f"\nProcessing: {os.path.basename(file)}")

        # Read CSV / CSV.GZ
        df = pd.read_csv(file)

        # Check which required columns actually exist
        existing_columns = [
            col for col in columns_to_keep
            if col in df.columns
        ]

        # Check for missing columns
        missing_columns = [
            col for col in columns_to_keep
            if col not in df.columns
        ]

        if missing_columns:
            print("WARNING - Missing columns:")
            for col in missing_columns:
                print(f"  - {col}")

        # Keep only required columns
        df = df[existing_columns]

        # Save back to the same file
        if file.endswith(".csv.gz"):

            df.to_csv(
                file,
                index=False,
                compression="gzip"
            )

        else:

            df.to_csv(
                file,
                index=False
            )

        print("Remaining columns:")
        for col in df.columns:
            print(f"  + {col}")

    print("\nDone! All files have been processed.")


if __name__ == "__main__":
    keep_required_columns()