import os
import sys
from networksecurity.exception import NetworkSecurityException
from networksecurity.logging import logging
from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)
from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.constants import (
    DATA_INGESTION_DIR_NAME,
    DATA_INGESTION_FEATURE_STORE_DIR,
    DATA_INGESTION_INGESTED_DIR,
    DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO,
    DATA_VALIDATION_DIR_NAME,
    DATA_VALIDATION_VALID_DIR,
    DATA_VALIDATION_INVALID_DIR,
    DATA_VALIDATION_DRIFT_REPORT_DIR,
    DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
    DATA_TRANSFORMATION_DIR_NAME,
    DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
    DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
    MODEL_TRAINER_DIR_NAME,
    MODEL_TRAINER_TRAINED_MODEL_DIR,
    MODEL_TRAINER_TRAINED_MODEL_NAME,
    MODEL_TRAINER_EXPECTED_SCORE,
    MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD,
    PREPROCESSING_OBJECT_FILE_NAME
)


class TrainPipeline:
    """
    Training Pipeline for the ML system
    """
    
    def __init__(self):
        """
        Initialize training pipeline with configuration
        """
        self.training_pipeline_config = TrainingPipelineConfig()
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Start data ingestion stage
        
        Returns:
            DataIngestionArtifact
        """
        try:
            logging.info("Starting data ingestion")
            
            data_ingestion_config = DataIngestionConfig(
                data_ingestion_dir=os.path.join(
                    self.training_pipeline_config.artifact_dir,
                    DATA_INGESTION_DIR_NAME
                ),
                feature_store_file_path=os.path.join(
                    self.training_pipeline_config.artifact_dir,
                    DATA_INGESTION_DIR_NAME,
                    DATA_INGESTION_FEATURE_STORE_DIR,
                    "data.csv"
                ),
                training_file_path=os.path.join(
                    self.training_pipeline_config.artifact_dir,
                    DATA_INGESTION_DIR_NAME,
                    DATA_INGESTION_INGESTED_DIR,
                    "train.csv"
                ),
                testing_file_path=os.path.join(
                    self.training_pipeline_config.artifact_dir,
                    DATA_INGESTION_DIR_NAME,
                    DATA_INGESTION_INGESTED_DIR,
                    "test.csv"
                ),
                train_test_split_ratio=DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
            )
            
            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            
            logging.info("Data ingestion completed")
            return data_ingestion_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        Start data validation stage
        
        Args:
            data_ingestion_artifact: Artifact from data ingestion
            
        Returns:
            DataValidationArtifact
        """
        try:
            logging.info("Starting data validation")
            
            data_validation_config = DataValidationConfig(
                data_validation_dir=os.path.join(
                    self.training_pipeline_config.artifact_dir,
                    DATA_VALIDATION_DIR_NAME
                ),
                valid_data_dir=os.path.join(
                    self.training_pipeline_config.artifact_dir,
                    DATA_VALIDATION_DIR_NAME,
                    DATA_VALIDATION_VALID_DIR
                ),
                invalid_data_dir=os.path.join(
                    self.training_pipeline_config.artifact_dir,
                    DATA_VALIDATION_DIR_NAME,
                    DATA_VALIDATION_INVALID_DIR
                ),
                drift_report_file_path=os.path.join(
                    self.training_pipeline_config.artifact_dir,
                    DATA_VALIDATION_DIR_NAME,
                    DATA_VALIDATION_DRIFT_REPORT_DIR,
                    DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
                )
            )
            
            data_validation = DataValidation(
                data_ingestion_artifact,
                data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            
            logging.info("Data validation completed")
            return data_validation_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        """
        Start data transformation stage
        
        Args:
            data_validation_artifact: Artifact from data validation
            
        Returns:
            DataTransformationArtifact
        """
        try:
            logging.info("Starting data transformation")
            
            data_transformation_config = DataTransformationConfig(
                data_transformation_dir=os.path.join(
                    self.training_pipeline_config.artifact_dir,
                    DATA_TRANSFORMATION_DIR_NAME
                ),
                transformed_train_file_path=os.path.join(
                    self.training_pipeline_config.artifact_dir,
                    DATA_TRANSFORMATION_DIR_NAME,
                    DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                    "train.npy"
                ),
                transformed_test_file_path=os.path.join(
                    self.training_pipeline_config.artifact_dir,
                    DATA_TRANSFORMATION_DIR_NAME,
                    DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                    "test.npy"
                ),
                transformed_object_file_path=os.path.join(
                    self.training_pipeline_config.artifact_dir,
                    DATA_TRANSFORMATION_DIR_NAME,
                    DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                    PREPROCESSING_OBJECT_FILE_NAME
                )
            )
            
            data_transformation = DataTransformation(
                data_validation_artifact,
                data_transformation_config
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            
            logging.info("Data transformation completed")
            return data_transformation_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        Start model training stage
        
        Args:
            data_transformation_artifact: Artifact from data transformation
            
        Returns:
            ModelTrainerArtifact
        """
        try:
            logging.info("Starting model training")
            
            model_trainer_config = ModelTrainerConfig(
                model_trainer_dir=os.path.join(
                    self.training_pipeline_config.artifact_dir,
                    MODEL_TRAINER_DIR_NAME
                ),
                trained_model_file_path=os.path.join(
                    self.training_pipeline_config.artifact_dir,
                    MODEL_TRAINER_DIR_NAME,
                    MODEL_TRAINER_TRAINED_MODEL_DIR,
                    MODEL_TRAINER_TRAINED_MODEL_NAME
                ),
                expected_accuracy=MODEL_TRAINER_EXPECTED_SCORE,
                overfitting_underfitting_threshold=MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD
            )
            
            model_trainer = ModelTrainer(
                model_trainer_config,
                data_transformation_artifact
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            
            logging.info("Model training completed")
            return model_trainer_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def run_pipeline(self):
        """
        Run the complete training pipeline
        """
        try:
            logging.info("Training pipeline started")
            
            # Data Ingestion
            data_ingestion_artifact = self.start_data_ingestion()
            
            # Data Validation
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            
            # Data Transformation
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            
            # Model Training
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            
            logging.info("Training pipeline completed successfully")
            logging.info(f"Trained model saved at: {model_trainer_artifact.trained_model_file_path}")
            
            return model_trainer_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
