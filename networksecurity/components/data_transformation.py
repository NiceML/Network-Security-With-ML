import os
import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from networksecurity.exception import NetworkSecurityException
from networksecurity.logging import logging
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from networksecurity.constants import TARGET_COLUMN
from networksecurity.utils.main_utils import save_numpy_array_data, save_object


class DataTransformation:
    """
    Data Transformation component for preprocessing data
    """
    
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        """
        Initialize data transformation with artifacts and configuration
        
        Args:
            data_validation_artifact: Artifact from data validation
            data_transformation_config: Configuration for data transformation
        """
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    @staticmethod
    def get_data_transformer_object() -> Pipeline:
        """
        Create and return a preprocessing pipeline
        
        Returns:
            sklearn Pipeline for data transformation
        """
        try:
            logging.info("Creating data transformation pipeline")
            
            # Create preprocessing pipeline
            preprocessor = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", RobustScaler())
                ]
            )
            
            logging.info("Data transformation pipeline created")
            return preprocessor
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiate the data transformation process
        
        Returns:
            DataTransformationArtifact with paths to transformed data
        """
        try:
            logging.info("Starting data transformation")
            
            # Read train and test data
            train_df = pd.read_csv(self.data_validation_artifact.valid_train_file_path)
            test_df = pd.read_csv(self.data_validation_artifact.valid_test_file_path)
            
            logging.info(f"Train dataframe shape: {train_df.shape}")
            logging.info(f"Test dataframe shape: {test_df.shape}")
            
            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN]
            
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN]
            
            # Replace -1 with 0 in target column for binary classification
            target_feature_train_df = target_feature_train_df.replace(-1, 0)
            target_feature_test_df = target_feature_test_df.replace(-1, 0)
            
            # Get preprocessor
            preprocessor = self.get_data_transformer_object()
            
            # Fit and transform train data
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)
            
            # Combine features and target
            train_arr = np.c_[
                transformed_input_train_feature,
                np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[
                transformed_input_test_feature,
                np.array(target_feature_test_df)
            ]
            
            # Save transformed data
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )
            
            # Save preprocessor object
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor_object
            )
            
            logging.info("Data transformation completed")
            
            # Create artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
