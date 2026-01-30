import os
import sys
import pandas as pd
import numpy as np
from networksecurity.exception import NetworkSecurityException
from networksecurity.logging import logging
from networksecurity.utils.main_utils import load_object


class PredictionPipeline:
    """
    Prediction Pipeline for making predictions on new data
    """
    
    def __init__(self, model_path: str, preprocessor_path: str):
        """
        Initialize prediction pipeline
        
        Args:
            model_path: Path to the trained model
            preprocessor_path: Path to the preprocessor object
        """
        try:
            self.model_path = model_path
            self.preprocessor_path = preprocessor_path
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def predict(self, features: pd.DataFrame) -> np.array:
        """
        Make predictions on new data
        
        Args:
            features: DataFrame with input features
            
        Returns:
            Array of predictions
        """
        try:
            logging.info("Starting prediction")
            
            # Expected feature names
            expected_features = [
                'having_IP_Address', 'URL_Length', 'Shortening_Service', 'having_At_Symbol',
                'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
                'Domain_registration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL',
                'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
                'Redirect', 'on_mouseover', 'RightClick', 'popUpWindow', 'Iframe',
                'age_of_domain', 'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index',
                'Links_pointing_to_page', 'Statistical_report'
            ]
            
            # Validate input features
            missing_features = set(expected_features) - set(features.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Select only expected features in the right order
            features = features[expected_features]
            
            # Load preprocessor and model
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)
            
            # Transform features
            transformed_features = preprocessor.transform(features)
            
            # Make predictions
            predictions = model.predict(transformed_features)
            
            # Convert predictions back to -1 and 1
            # Model predicts 0 (phishing) and 1 (legitimate)
            # Convert to -1 (phishing) and 1 (legitimate) as per original encoding
            predictions = np.where(predictions == 0, -1, 1)
            
            logging.info("Prediction completed")
            return predictions
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
