#!/usr/bin/env python3
"""
Demo script to show how to use the trained model for predictions
"""
import os
import sys
import pandas as pd
from networksecurity.exception import NetworkSecurityException
from networksecurity.logging import logging
from networksecurity.pipeline.prediction_pipeline import PredictionPipeline


def demo_prediction():
    """
    Demonstrate making predictions with the trained model
    """
    try:
        print("=" * 80)
        print("Phishing Website Detection - Prediction Demo")
        print("=" * 80)
        
        # Find the latest trained model
        artifacts_dir = "artifacts"
        
        # Check if artifacts directory exists
        if not os.path.exists(artifacts_dir):
            print("\nError: No trained model found. Please run train.py first.")
            return
        
        # Get the latest artifact directory
        artifact_dirs = [d for d in os.listdir(artifacts_dir) if os.path.isdir(os.path.join(artifacts_dir, d))]
        if not artifact_dirs:
            print("\nError: No trained model found. Please run train.py first.")
            return
        
        latest_artifact = sorted(artifact_dirs)[-1]
        
        # Define paths
        model_path = os.path.join(
            artifacts_dir,
            latest_artifact,
            "model_trainer",
            "trained_model",
            "model.pkl"
        )
        
        preprocessor_path = os.path.join(
            artifacts_dir,
            latest_artifact,
            "data_transformation",
            "transformed_object",
            "preprocessing.pkl"
        )
        
        # Check if files exist
        if not os.path.exists(model_path):
            print(f"\nError: Model file not found at {model_path}")
            return
        
        if not os.path.exists(preprocessor_path):
            print(f"\nError: Preprocessor file not found at {preprocessor_path}")
            return
        
        print(f"\nUsing model from: {model_path}")
        
        # Create sample data for prediction
        sample_data = {
            'having_IP_Address': [1, -1, 1],
            'URL_Length': [0, 1, -1],
            'Shortening_Service': [1, 1, -1],
            'having_At_Symbol': [1, -1, -1],
            'double_slash_redirecting': [-1, 1, 1],
            'Prefix_Suffix': [-1, -1, 1],
            'having_Sub_Domain': [1, 0, -1],
            'SSLfinal_State': [0, -1, 1],
            'Domain_registration_length': [1, -1, 1],
            'Favicon': [1, 1, -1],
            'port': [1, -1, -1],
            'HTTPS_token': [-1, 1, -1],
            'Request_URL': [1, 1, -1],
            'URL_of_Anchor': [0, -1, 1],
            'Links_in_tags': [1, 0, -1],
            'SFH': [1, -1, 0],
            'Submitting_to_email': [-1, 1, -1],
            'Abnormal_URL': [1, -1, 1],
            'Redirect': [0, 0, -1],
            'on_mouseover': [1, 1, -1],
            'RightClick': [1, -1, -1],
            'popUpWindow': [1, 1, -1],
            'Iframe': [1, -1, -1],
            'age_of_domain': [-1, 1, 1],
            'DNSRecord': [-1, -1, 1],
            'web_traffic': [0, -1, 1],
            'Page_Rank': [-1, 1, -1],
            'Google_Index': [1, -1, 1],
            'Links_pointing_to_page': [0, 1, -1],
            'Statistical_report': [1, -1, -1]
        }
        
        sample_df = pd.DataFrame(sample_data)
        
        print("\nSample data for prediction:")
        print(sample_df)
        
        # Create prediction pipeline
        pipeline = PredictionPipeline(model_path, preprocessor_path)
        
        # Make predictions
        predictions = pipeline.predict(sample_df)
        
        print("\nPredictions:")
        print("-" * 80)
        for i, pred in enumerate(predictions):
            status = "PHISHING" if pred == -1 else "LEGITIMATE"
            print(f"Website {i+1}: {status}")
        print("-" * 80)
        
        print("\nNote:")
        print("  -1 or PHISHING: Website is predicted to be a phishing site")
        print("   1 or LEGITIMATE: Website is predicted to be legitimate")
        
        print("\n" + "=" * 80)
        print("Demo completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        logging.error(f"Error in prediction demo: {str(e)}")
        raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    demo_prediction()
