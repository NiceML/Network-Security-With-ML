#!/usr/bin/env python3
"""
Main script to train the phishing detection model
"""
import sys
from networksecurity.exception import NetworkSecurityException
from networksecurity.logging import logging
from networksecurity.pipeline.training_pipeline import TrainPipeline


def main():
    """
    Main function to run the training pipeline
    """
    try:
        logging.info("=" * 80)
        logging.info("Starting Phishing Website Detection ML System")
        logging.info("=" * 80)
        
        # Create and run training pipeline
        pipeline = TrainPipeline()
        model_trainer_artifact = pipeline.run_pipeline()
        
        logging.info("=" * 80)
        logging.info("Training completed successfully!")
        logging.info(f"Model saved at: {model_trainer_artifact.trained_model_file_path}")
        logging.info(f"Train F1 Score: {model_trainer_artifact.train_metric_artifact.f1_score:.4f}")
        logging.info(f"Test F1 Score: {model_trainer_artifact.test_metric_artifact.f1_score:.4f}")
        logging.info(f"Train Precision: {model_trainer_artifact.train_metric_artifact.precision_score:.4f}")
        logging.info(f"Test Precision: {model_trainer_artifact.test_metric_artifact.precision_score:.4f}")
        logging.info(f"Train Recall: {model_trainer_artifact.train_metric_artifact.recall_score:.4f}")
        logging.info(f"Test Recall: {model_trainer_artifact.test_metric_artifact.recall_score:.4f}")
        logging.info("=" * 80)
        
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print(f"Model saved at: {model_trainer_artifact.trained_model_file_path}")
        print(f"Train F1 Score: {model_trainer_artifact.train_metric_artifact.f1_score:.4f}")
        print(f"Test F1 Score: {model_trainer_artifact.test_metric_artifact.f1_score:.4f}")
        print("=" * 80 + "\n")
        
    except Exception as e:
        logging.error(f"Error in training pipeline: {str(e)}")
        raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    main()
