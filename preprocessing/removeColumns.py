import pandas as pd
from pathlib import Path

from paths import LABEL_DIR, RAW_FEATURES_LABELS_DIR

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
    "label:CLEANING",
]


def remove_columns(
    input_folder: Path = RAW_FEATURES_LABELS_DIR,
    output_folder: Path = LABEL_DIR,
):
    """Remove unwanted columns and write cleaned CSVs to the labels directory."""
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(input_folder.glob("*.csv"))

    for file in csv_files:
        print(f"\nProcessing: {file.name}")

        df = pd.read_csv(file)

        columns_to_remove = [
            col for col in df.columns if any(phrase in col for phrase in phrases)
        ]
        df = df.drop(columns=columns_to_remove)

        computer_col = "label:COMPUTER_WORK"
        sitting_col = "label:SITTING"

        if computer_col in df.columns and sitting_col in df.columns:
            df[sitting_col] = df[[computer_col, sitting_col]].max(axis=1, skipna=True)

            both_empty = df[[computer_col, sitting_col]].isna().all(axis=1)
            df.loc[both_empty, sitting_col] = float("nan")

            df.drop(columns=[computer_col], inplace=True)

        output_file = output_folder / file.name
        df.to_csv(output_file, index=False)
        print(f"Saved cleaned CSV: {output_file}")

        print("Removed columns:")
        for col in columns_to_remove:
            print(f"  - {col}")

    print("\nDone! All CSV files have been processed.")


if __name__ == "__main__":
    remove_columns()