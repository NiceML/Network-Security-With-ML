import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from networksecurity.exception import NetworkSecurityException
from networksecurity.logging import logging
from networksecurity.entity.artifact_entity import ClassificationMetricArtifact


def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    """
    Calculate classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        ClassificationMetricArtifact with F1, precision, and recall scores
    """
    try:
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)
        
        classification_metric = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score
        )
        return classification_metric
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate multiple models and return their scores and trained models
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        models: Dictionary of models to evaluate
        param: Dictionary of hyperparameters for grid search
        
    Returns:
        Tuple of (report dict with test scores, dict of trained models)
    """
    try:
        report = {}
        trained_models = {}
        
        for model_name, model in models.items():
            logging.info(f"Evaluating model: {model_name}")
            
            # Get parameters for this model
            para = param.get(model_name, {})
            
            # Perform grid search if parameters are provided
            if para:
                gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=1)
                gs.fit(X_train, y_train)
                # Use the best estimator from grid search
                trained_model = gs.best_estimator_
            else:
                # Train the model without grid search
                model.fit(X_train, y_train)
                trained_model = model
            
            # Store the trained model
            trained_models[model_name] = trained_model
            
            # Make predictions
            y_train_pred = trained_model.predict(X_train)
            y_test_pred = trained_model.predict(X_test)
            
            # Calculate scores
            train_score = f1_score(y_train, y_train_pred)
            test_score = f1_score(y_test, y_test_pred)
            
            report[model_name] = test_score
            
            logging.info(f"{model_name} - Train F1 Score: {train_score}, Test F1 Score: {test_score}")
        
        return report, trained_models
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)
