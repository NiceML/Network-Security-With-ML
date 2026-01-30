import os
from datetime import datetime

# Define MongoDB connection string
DATABASE_NAME = "NETWORK_SECURITY_DB"
COLLECTION_NAME = "PhishingData"

# Training pipeline constants
PIPELINE_NAME: str = "networksecurity"
ARTIFACT_DIR: str = "artifacts"

# Data ingestion constants
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# Data validation constants
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

# Data transformation constants
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

# Model trainer constants
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD: float = 0.05

# Model evaluation constants
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_EVALUATION_REPORT_NAME: str = "report.yaml"

# Model pusher constants
MODEL_PUSHER_DIR_NAME: str = "model_pusher"
MODEL_PUSHER_SAVED_MODEL_DIR: str = "saved_models"

# Target column
TARGET_COLUMN = "Result"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

# Model file name
MODEL_FILE_NAME = "model.pkl"
