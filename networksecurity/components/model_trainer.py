import os
import sys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from networksecurity.exception import NetworkSecurityException
from networksecurity.logging import logging
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from networksecurity.utils.main_utils import save_object, load_numpy_array_data
from networksecurity.utils.ml_utils import get_classification_score


class ModelTrainer:
    """
    Model Trainer component for training and selecting best ML model
    """

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def _get_models_and_params(self):
        """
        Define models and hyperparameter grids
        """
        models = {
            "RandomForest": RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            ),
            "DecisionTree": DecisionTreeClassifier(
                random_state=42
            ),
            "GradientBoosting": GradientBoostingClassifier(
                random_state=42
            ),
            "LogisticRegression": LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        }

        params = {
            "RandomForest": {
                "n_estimators": [50, 100],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5]
            },
            "DecisionTree": {
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5, 10]
            },
            "GradientBoosting": {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5]
            },
            "LogisticRegression": {
                "C": [0.1, 1.0, 10.0]
            }
        }

        return models, params

    def train_and_select_model(self, X_train, y_train):
        """
        Train models using CV and return best model
        """
        try:
            logging.info("Starting model training with cross-validation")

            models, params = self._get_models_and_params()

            model_scores = {}
            trained_models = {}

            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}")

                grid = GridSearchCV(
                    estimator=model,
                    param_grid=params[model_name],
                    scoring="f1",
                    cv=5,
                    n_jobs=-1
                )

                grid.fit(X_train, y_train)

                best_model = grid.best_estimator_
                best_score = grid.best_score_

                model_scores[model_name] = best_score
                trained_models[model_name] = best_model

                logging.info(
                    f"{model_name} best CV F1 score: {best_score}"
                )

            best_model_name = max(model_scores, key=model_scores.get)
            best_model = trained_models[best_model_name]

            logging.info(
                f"Selected best model: {best_model_name} "
                f"with CV F1 score: {model_scores[best_model_name]}"
            )

            return best_model

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Initiates full model training pipeline
        """
        try:
            logging.info("Loading transformed datasets")

            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logging.info(
                f"Train shape: {X_train.shape}, Test shape: {X_test.shape}"
            )

            best_model = self.train_and_select_model(X_train, y_train)

            # Final evaluation on test set (ONLY ONCE)
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_metric = get_classification_score(
                y_train, y_train_pred
            )
            test_metric = get_classification_score(
                y_test, y_test_pred
            )

            logging.info(f"Train Metrics: {train_metric}")
            logging.info(f"Test Metrics: {test_metric}")

            # Accuracy threshold check
            if test_metric.f1_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"Model failed accuracy threshold. "
                    f"Expected: {self.model_trainer_config.expected_accuracy}, "
                    f"Got: {test_metric.f1_score}"
                )

            # Overfitting check
            diff = abs(
                train_metric.f1_score - test_metric.f1_score
            )

            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                logging.warning(
                    f"Potential overfitting/underfitting detected. "
                    f"F1 difference: {diff}"
                )

            save_object(
                self.model_trainer_config.trained_model_file_path,
                best_model
            )

            logging.info("Model training completed successfully")

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys)
