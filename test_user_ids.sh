#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ACC_DIR="$SCRIPT_DIR/datasets/processed/acc"
DIR1="$SCRIPT_DIR/datasets/processed/gyro"
DIR2="$SCRIPT_DIR/datasets/source/labels"

for folder in "$ACC_DIR"/*/; do

    name=$(basename "$folder")

    # Check in processed gyro data
    if [ ! -d "$DIR1/$name" ]; then
        echo "$name -> Missing in processed/gyro"
    fi

    # Check as CSV file in source labels
    if [ ! -f "$DIR2/$name.features_labels.csv" ]; then
        echo "$name -> Missing in source/labels"
    fi

done