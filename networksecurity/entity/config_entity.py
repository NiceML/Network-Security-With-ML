from dataclasses import dataclass
from datetime import datetime
import os
from networksecurity.constants import PIPELINE_NAME, ARTIFACT_DIR


@dataclass
class TrainingPipelineConfig:
    """
    Configuration for training pipeline
    """
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}")
    timestamp: str = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')


@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion
    """
    data_ingestion_dir: str
    feature_store_file_path: str
    training_file_path: str
    testing_file_path: str
    train_test_split_ratio: float


@dataclass
class DataValidationConfig:
    """
    Configuration for data validation
    """
    data_validation_dir: str
    valid_data_dir: str
    invalid_data_dir: str
    drift_report_file_path: str


@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation
    """
    data_transformation_dir: str
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_object_file_path: str


@dataclass
class ModelTrainerConfig:
    """
    Configuration for model trainer
    """
    model_trainer_dir: str
    trained_model_file_path: str
    expected_accuracy: float
    overfitting_underfitting_threshold: float


@dataclass
class ModelEvaluationConfig:
    """
    Configuration for model evaluation
    """
    model_evaluation_dir: str
    model_evaluation_report_file_path: str
    changed_threshold_score: float


@dataclass
class ModelPusherConfig:
    """
    Configuration for model pusher
    """
    model_pusher_dir: str
    model_file_path: str
    saved_model_path: str
