import os
import sys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from networksecurity.exception import NetworkSecurityException
from networksecurity.logging import logging
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.utils.main_utils import save_object, load_numpy_array_data
from networksecurity.utils.ml_utils import get_classification_score, evaluate_models


class ModelTrainer:
    """
    Model Trainer component for training ML models
    """
    
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        """
        Initialize model trainer with configuration and artifacts
        
        Args:
            model_trainer_config: Configuration for model training
            data_transformation_artifact: Artifact from data transformation
        """
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def train_model(self, X_train, y_train):
        """
        Train multiple models and select the best one
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model object
        """
        try:
            logging.info("Training models")
            
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(max_iter=1000)
            }
            
            params = {
                "Random Forest": {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                },
                "Decision Tree": {
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                },
                "Gradient Boosting": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                },
                "Logistic Regression": {
                    'C': [0.1, 1.0, 10.0]
                }
            }
            
            # Load test data for evaluation
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )
            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]
            
            # Evaluate all models
            model_report, trained_models = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )
            
            # Get best model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = trained_models[best_model_name]
            
            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")
            
            return best_model
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Initiate the model training process
        
        Returns:
            ModelTrainerArtifact with trained model path and metrics
        """
        try:
            logging.info("Starting model training")
            
            # Load transformed data
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )
            
            # Split features and target
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            
            logging.info(f"Train data shape: {X_train.shape}, {y_train.shape}")
            logging.info(f"Test data shape: {X_test.shape}, {y_test.shape}")
            
            # Train model
            model = self.train_model(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_metric = get_classification_score(y_train, y_train_pred)
            test_metric = get_classification_score(y_test, y_test_pred)
            
            logging.info(f"Train metrics: {train_metric}")
            logging.info(f"Test metrics: {test_metric}")
            
            # Check if model meets expected accuracy
            if test_metric.f1_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"Model does not meet expected accuracy. "
                    f"Expected: {self.model_trainer_config.expected_accuracy}, "
                    f"Got: {test_metric.f1_score}"
                )
            
            # Check for overfitting/underfitting
            diff = abs(train_metric.f1_score - test_metric.f1_score)
            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                logging.warning(
                    f"Model may be overfitting/underfitting. "
                    f"Difference: {diff}"
                )
            
            # Save the model
            save_object(
                self.model_trainer_config.trained_model_file_path,
                model
            )
            
            logging.info("Model training completed")
            
            # Create artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric
            )
            
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
