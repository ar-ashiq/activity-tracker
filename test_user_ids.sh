#!/bin/bash

ACC_DIR="/Users/ashiqar/chore/ubiquitous/Codes/datasets/acc_outputs"
DIR1="/Users/ashiqar/chore/ubiquitous/Codes/datasets/gyro_outputs"
DIR2="/Users/ashiqar/chore/ubiquitous/Codes/datasets/ExtraSensory.per_uuid_features_labels"

for folder in "$ACC_DIR"/*/; do

    name=$(basename "$folder")

    # Check in gyro_outputs
    if [ ! -d "$DIR1/$name" ]; then
        echo "$name -> Missing in gyro_outputs"
    fi

    # Check as CSV file in ExtraSensory.per_uuid_features_labels
    if [ ! -f "$DIR2/$name.features_labels.csv" ]; then
        echo "$name -> Missing in ExtraSensory.per_uuid_features_labels"
    fi

done