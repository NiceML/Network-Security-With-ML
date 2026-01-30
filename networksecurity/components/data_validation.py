import os
import sys
import pandas as pd
from networksecurity.exception import NetworkSecurityException
from networksecurity.logging import logging
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.utils.main_utils import write_yaml_file


class DataValidation:
    """
    Data Validation component for validating data quality
    """
    
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        """
        Initialize data validation with artifacts and configuration
        
        Args:
            data_ingestion_artifact: Artifact from data ingestion
            data_validation_config: Configuration for data validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate if dataframe has expected number of columns
        
        Args:
            dataframe: Input dataframe to validate
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # For phishing dataset, we expect at least 30 columns (features + target)
            expected_columns = 30
            actual_columns = len(dataframe.columns)
            
            if actual_columns >= expected_columns:
                logging.info(f"Dataframe has {actual_columns} columns (>= {expected_columns})")
                return True
            else:
                logging.warning(f"Dataframe has {actual_columns} columns (< {expected_columns})")
                return False
                
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame) -> bool:
        """
        Detect data drift between base and current datasets
        
        Args:
            base_df: Base/reference dataframe
            current_df: Current dataframe to compare
            
        Returns:
            True if drift is detected, False otherwise
        """
        try:
            logging.info("Detecting data drift")
            
            # Simple drift detection based on statistical measures
            drift_detected = False
            drift_report = {"drift_detected": False, "column_drifts": {}}
            
            # Compare statistical properties of each column
            for column in base_df.columns:
                if column in current_df.columns:
                    base_mean = base_df[column].mean()
                    current_mean = current_df[column].mean()
                    base_std = base_df[column].std()
                    current_std = current_df[column].std()
                    
                    # Check if mean differs by more than 2 standard deviations
                    mean_diff = abs(base_mean - current_mean)
                    threshold = 2 * base_std if base_std > 0 else 0.1
                    
                    column_drift = mean_diff > threshold
                    drift_report["column_drifts"][column] = {
                        "drift_detected": column_drift,
                        "base_mean": float(base_mean),
                        "current_mean": float(current_mean),
                        "mean_diff": float(mean_diff)
                    }
                    
                    if column_drift:
                        drift_detected = True
            
            drift_report["drift_detected"] = drift_detected
            
            # Save the report
            report_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            write_yaml_file(file_path=report_path, content=drift_report)
            
            logging.info(f"Data drift detected: {drift_detected}")
            return drift_detected
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Initiate the data validation process
        
        Returns:
            DataValidationArtifact with validation results
        """
        try:
            logging.info("Starting data validation")
            
            # Read train and test data
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            
            train_dataframe = pd.read_csv(train_file_path)
            test_dataframe = pd.read_csv(test_file_path)
            
            # Validate number of columns
            status = self.validate_number_of_columns(train_dataframe)
            if not status:
                logging.error("Train data validation failed")
            
            status = status and self.validate_number_of_columns(test_dataframe)
            if not status:
                logging.error("Test data validation failed")
            
            # Detect data drift
            drift_status = self.detect_dataset_drift(
                base_df=train_dataframe,
                current_df=test_dataframe
            )
            
            # Create directories
            dir_path = os.path.dirname(self.data_validation_config.valid_data_dir)
            os.makedirs(dir_path, exist_ok=True)
            
            # Define file paths
            valid_train_file_path = os.path.join(
                self.data_validation_config.valid_data_dir,
                "train.csv"
            )
            valid_test_file_path = os.path.join(
                self.data_validation_config.valid_data_dir,
                "test.csv"
            )
            invalid_train_file_path = os.path.join(
                self.data_validation_config.invalid_data_dir,
                "train.csv"
            )
            invalid_test_file_path = os.path.join(
                self.data_validation_config.invalid_data_dir,
                "test.csv"
            )
            
            # Save data based on validation status
            if status:
                os.makedirs(self.data_validation_config.valid_data_dir, exist_ok=True)
                train_dataframe.to_csv(valid_train_file_path, index=False, header=True)
                test_dataframe.to_csv(valid_test_file_path, index=False, header=True)
            else:
                os.makedirs(self.data_validation_config.invalid_data_dir, exist_ok=True)
                train_dataframe.to_csv(invalid_train_file_path, index=False, header=True)
                test_dataframe.to_csv(invalid_test_file_path, index=False, header=True)
            
            # Create artifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=valid_train_file_path if status else None,
                valid_test_file_path=valid_test_file_path if status else None,
                invalid_train_file_path=invalid_train_file_path if not status else None,
                invalid_test_file_path=invalid_test_file_path if not status else None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            
            logging.info(f"Data validation completed. Artifact: {data_validation_artifact}")
            return data_validation_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
