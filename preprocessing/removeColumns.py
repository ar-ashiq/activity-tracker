import pandas as pd
import os
import glob

# Folder containing your CSV files with the labels
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder_path = os.path.join(project_root, "datasets/ExtraSensory.per_uuid_features_labels")

# Phrases to search for in column headings
phrases = [
    "raw_magnet",
    "watch_acceleration",
    "location",
    "audio_naive",
    "audio_properties",
    "lf_measurements",
    "watch_heading",
    "discrete:battery",
    "discrete:app_state",
    "discrete:ringer",
    "discrete:wifi_",

    "WITH_FRIENDS",
    "WITH_CO-WORKERS",
    "PHONE_ON_TABLE",
    "PHONE_IN_BAG",
    "PHONE_IN_HAND",
    "AT_SCHOOL",
    "ELEVATOR",
    "STAIRS_-_GOING_DOWN",
    "STAIRS_-_GOING_UP",
    "AT_THE_GYM",
    "label:DRIVE_-_I_M_A_PASSENGER",
    "GROOMING",
    "DRESSING",
    "TOILET",
    "AT_A_PARTY",
    "LOC_beach",
    "label:FIX_restaurant",
    "label:PHONE_IN_POCKET",
    "AT_A_BAR",
    "LOC_home",
    "label:DRIVE_-_I_M_THE_DRIVER",
    "SLEEPING",
    "SURFING_THE_INTERNET",
    "LAB_WORK",
    "IN_CLASS",
    "OR_indoors",
    "ON_A_BUS",
    "IN_A_CAR",
    "LOC_main_workplace",
    "IN_A_MEETING",
    "OR_outside",
    "label:EATING",
    "label:TALKING",
    "label:COOKING",
    "label:SINGING",
    "label:WATCHING_TV",
    "label:SHOPPING",
    "label:STROLLING",
    "label:OR_exercise",
    "label:WASHING_DISHES",
    "label:DRINKING__ALCOHOL_",
    "label:BATHING_-_SHOWER",
    "label:DOING_LAUNDRY",
    "label:CLEANING"
]

# Get all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

for file in csv_files:

    print(f"\nProcessing: {os.path.basename(file)}")

    # Read CSV
    df = pd.read_csv(file)

    # --------------------------------------------------
    # Find columns whose heading contains any phrase
    # --------------------------------------------------

    columns_to_remove = [
        col for col in df.columns
        if any(phrase in col for phrase in phrases)
    ]

    # Remove those columns
    df = df.drop(columns=columns_to_remove)

    # Combine COMPUTER_WORK and SITTING using OR logic
    computer_col = "label:COMPUTER_WORK"
    sitting_col = "label:SITTING"

    if computer_col in df.columns and sitting_col in df.columns:

        # OR logic: if either column is 1.0, result is 1.0
        df[sitting_col] = df[[computer_col, sitting_col]].max(axis=1, skipna=True)

        # Keep both empty as empty
        both_empty = df[[computer_col, sitting_col]].isna().all(axis=1)
        df.loc[both_empty, sitting_col] = float("nan")

        # Remove COMPUTER_WORK column
        df.drop(columns=[computer_col], inplace=True)

    # Save back to the same CSV file
    df.to_csv(file, index=False)

    print("Removed columns:")
    for col in columns_to_remove:
        print(f"  - {col}")

print("\nDone! All CSV files have been processed.")