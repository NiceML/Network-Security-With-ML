from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    """
    Artifact for data ingestion
    """
    trained_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    """
    Artifact for data validation
    """
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str


@dataclass
class DataTransformationArtifact:
    """
    Artifact for data transformation
    """
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str


@dataclass
class ClassificationMetricArtifact:
    """
    Artifact for classification metrics
    """
    f1_score: float
    precision_score: float
    recall_score: float


@dataclass
class ModelTrainerArtifact:
    """
    Artifact for model trainer
    """
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact


@dataclass
class ModelEvaluationArtifact:
    """
    Artifact for model evaluation
    """
    is_model_accepted: bool
    improved_accuracy: float
    best_model_path: str
    trained_model_path: str
    train_model_metric_artifact: ClassificationMetricArtifact
    best_model_metric_artifact: ClassificationMetricArtifact


@dataclass
class ModelPusherArtifact:
    """
    Artifact for model pusher
    """
    saved_model_path: str
    model_file_path: str
