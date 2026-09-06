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

## Datasets to download and place in the `datasets` folder for training

Dataset downloaded from: [ExtraSensory Dataset](http://extrasensory.ucsd.edu) and put the datasets in `/datasets`.

1. Timestamps and corresponding labels for each user: [ExtraSensory.per_uuid_features_labels.zip](http://extrasensory.ucsd.edu/data/primary_data_files/ExtraSensory.per_uuid_features_labels.zip) - link to download the zip file

   ### Run the Script

   Run the following command:

   ```bash
   cd datasets
   ./unzip_files.sh
   ```

2. Raw Accelerometer Readings: [ExtraSensory.raw_measurements.raw_acc.zip](http://extrasensory.ucsd.edu/data/raw_measurements/ExtraSensory.raw_measurements.raw_acc.zip) - link to download the zip file

3. Raw Gyroscope Readings: [ExtraSensory.raw_measurements.proc_gyro.zip](http://extrasensory.ucsd.edu/data/raw_measurements/ExtraSensory.raw_measurements.proc_gyro.zip) - link to download the zip file


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