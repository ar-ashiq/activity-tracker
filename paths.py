"""Shared project paths used by preprocessing, training, prediction, and the API."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT.joinpath("datasets")

SOURCE_DIR = DATASETS_DIR.joinpath("source")
LABEL_DIR = SOURCE_DIR.joinpath("labels")

RAW_DIR = DATASETS_DIR.joinpath("raw")
RAW_ACC_DIR = RAW_DIR.joinpath("acc")
RAW_GYRO_DIR = RAW_DIR.joinpath("gyro")

PROCESSED_DIR = DATASETS_DIR.joinpath("processed")
ACC_DIR = PROCESSED_DIR.joinpath("acc")
GYRO_DIR = PROCESSED_DIR.joinpath("gyro")
SYNCHRONIZED_DIR = PROCESSED_DIR.joinpath("synchronized")
RAW_SYNC_DIR = PROCESSED_DIR.joinpath("raw_sync")

TESTING_DIR = DATASETS_DIR.joinpath("testing")
TESTING_USER_ID = os.getenv(
    "ACTIVITY_TEST_USER_ID",
    "0A986513-7828-4D53-AA1F-E02D6DF9561B",
)
TESTING_USER_DIR = TESTING_DIR.joinpath(TESTING_USER_ID)
TESTING_PREDICTIONS_FILE = TESTING_USER_DIR.joinpath("testing_predictions.json")
TESTING_CONSOLIDATED_FILE = TESTING_USER_DIR.joinpath(
    "testing_predictions_consolidated.json"
)

TRAINING_OUTPUTS_DIR = PROJECT_ROOT.joinpath("trainingOutputs")
CNN_OUTPUT_DIR = TRAINING_OUTPUTS_DIR.joinpath("cnn")
CNN_MODEL_DIR = CNN_OUTPUT_DIR.joinpath("models")
CNN_SCALER_DIR = CNN_OUTPUT_DIR.joinpath("scalers")
CNN_PLOT_DIR = CNN_OUTPUT_DIR.joinpath("plots")
CNN_METRICS_DIR = CNN_OUTPUT_DIR.joinpath("metrics")

LSTM_OUTPUT_DIR = TRAINING_OUTPUTS_DIR.joinpath("lstm")
LSTM_MODEL_DIR = LSTM_OUTPUT_DIR.joinpath("models")
LSTM_SCALER_DIR = LSTM_OUTPUT_DIR.joinpath("scalers")
LSTM_PLOT_DIR = LSTM_OUTPUT_DIR.joinpath("plots")
LSTM_METRICS_DIR = LSTM_OUTPUT_DIR.joinpath("metrics")

CNN_MODEL_PATH = CNN_MODEL_DIR.joinpath("activity_recognition_cnn.keras")
CNN_BEST_MODEL_PATH = CNN_MODEL_DIR.joinpath("activity_cnn_best.keras")
CNN_SCALER_PATH = CNN_SCALER_DIR.joinpath("activity_scaler.pkl")

LSTM_MODEL_PATH = LSTM_MODEL_DIR.joinpath("activity_recognition_lstm.keras")
LSTM_BEST_MODEL_PATH = LSTM_MODEL_DIR.joinpath("activity_lstm_best.keras")
LSTM_SEQUENCE_SCALER_PATH = LSTM_SCALER_DIR.joinpath("activity_sequence_scaler.pkl")
LSTM_STAT_SCALER_PATH = LSTM_SCALER_DIR.joinpath("activity_stat_scaler.pkl")
