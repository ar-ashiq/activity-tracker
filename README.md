# Human Activity Recognition Using Accelerometer and Gyroscope Data

## Overview

This project aims to develop a machine learning model that recognizes a user's physical activity using raw **accelerometer** and **gyroscope** sensor data.

The model will classify the user's activity into five categories:

* Sitting
* Standing
* Walking
* Running
* Bicycling

The sensor data is stored in CSV files along with timestamp information and activity labels.

## Dataset layout

All dataset paths are centralized in `paths.py`. Keep the files organized as:

```text
datasets/
├── source/
│   └── labels/                  # ExtraSensory features_labels CSV files
├── raw/
│   ├── acc/                     # Raw accelerometer recordings
│   └── gyro/                    # Raw gyroscope recordings
├── processed/
│   ├── acc/                     # Converted accelerometer CSV files
│   ├── gyro/                    # Converted gyroscope CSV files
│   ├── synchronized/            # Training CSV files with activity labels
│   └── raw_sync/                # Output from raw sensor synchronization
└── testing/<user-id>/           # Raw acc/gyro files and prediction output
```

## Datasets to download

Dataset downloaded from: [ExtraSensory Dataset](http://extrasensory.ucsd.edu).

1. Download the timestamps and labels from [ExtraSensory.per_uuid_features_labels.zip](http://extrasensory.ucsd.edu/data/primary_data_files/ExtraSensory.per_uuid_features_labels.zip) and place the extracted files in `datasets/source/labels`.

   ### Run the Script

   Run the following command:

   ```bash
   ./datasets/unzip_gz.sh
   ```

2. Download the raw accelerometer readings from [ExtraSensory.raw_measurements.raw_acc.zip](http://extrasensory.ucsd.edu/data/raw_measurements/ExtraSensory.raw_measurements.raw_acc.zip) and place them under `datasets/raw/acc`.

3. Download the raw gyroscope readings from [ExtraSensory.raw_measurements.proc_gyro.zip](http://extrasensory.ucsd.edu/data/raw_measurements/ExtraSensory.raw_measurements.proc_gyro.zip) and place them under `datasets/raw/gyro`.

The preprocessing pipeline writes converted and synchronized data to
`datasets/processed`; do not place generated files back into the source or raw
directories.


## Setup

Create and activate the virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the preprocessing and training scripts from the project root:
```bash
python -m preprocessing.removeColumns
python -m preprocessing.synchronize_combine
python cnnTraining.py
python lstmTraining.py
```

Training artifacts are kept separately by model:

```text
trainingOutputs/
├── cnn/
│   ├── models/
│   ├── scalers/
│   ├── plots/
│   └── metrics/
└── lstm/
   ├── models/
   ├── scalers/
   ├── plots/
   └── metrics/
```

The prediction script uses the best CNN model from
`trainingOutputs/cnn/models` and its scaler from `trainingOutputs/cnn/scalers`.

Run the main file:
```bash
python main.py
```