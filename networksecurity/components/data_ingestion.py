import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from networksecurity.exception import NetworkSecurityException
from networksecurity.logging import logging
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact


class DataIngestion:
    """
    Data Ingestion component for loading and splitting data
    """
    
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initialize data ingestion with configuration
        
        Args:
            data_ingestion_config: Configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def export_data_into_feature_store(self) -> pd.DataFrame:
        """
        Export data from MongoDB or CSV file into feature store
        
        Returns:
            DataFrame with the exported data
        """
        try:
            logging.info("Exporting data from data source into feature store")
            
            # For this implementation, we'll check if a CSV file exists in data directory
            # If not, we'll create a sample dataset
            data_file_path = os.path.join("data", "phishing_data.csv")
            
            if os.path.exists(data_file_path):
                logging.info(f"Loading data from {data_file_path}")
                dataframe = pd.read_csv(data_file_path)
            else:
                # Create a sample phishing dataset for demonstration
                logging.info("Creating sample phishing dataset")
                dataframe = self._create_sample_dataset()
                os.makedirs("data", exist_ok=True)
                dataframe.to_csv(data_file_path, index=False)
                logging.info(f"Sample data saved to {data_file_path}")
            
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            
            # Create feature store directory
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            # Save to feature store
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logging.info(f"Data saved to feature store: {feature_store_file_path}")
            
            return dataframe
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def _create_sample_dataset(self) -> pd.DataFrame:
        """
        Create a sample phishing dataset with features
        
        Returns:
            DataFrame with sample phishing data
        """
        import numpy as np
        np.random.seed(42)
        
        n_samples = 1000
        
        # Create features that might indicate phishing
        data = {
            'having_IP_Address': np.random.choice([-1, 1], n_samples),
            'URL_Length': np.random.choice([-1, 0, 1], n_samples),
            'Shortening_Service': np.random.choice([-1, 1], n_samples),
            'having_At_Symbol': np.random.choice([-1, 1], n_samples),
            'double_slash_redirecting': np.random.choice([-1, 1], n_samples),
            'Prefix_Suffix': np.random.choice([-1, 1], n_samples),
            'having_Sub_Domain': np.random.choice([-1, 0, 1], n_samples),
            'SSLfinal_State': np.random.choice([-1, 0, 1], n_samples),
            'Domain_registration_length': np.random.choice([-1, 1], n_samples),
            'Favicon': np.random.choice([-1, 1], n_samples),
            'port': np.random.choice([-1, 1], n_samples),
            'HTTPS_token': np.random.choice([-1, 1], n_samples),
            'Request_URL': np.random.choice([-1, 1], n_samples),
            'URL_of_Anchor': np.random.choice([-1, 0, 1], n_samples),
            'Links_in_tags': np.random.choice([-1, 0, 1], n_samples),
            'SFH': np.random.choice([-1, 0, 1], n_samples),
            'Submitting_to_email': np.random.choice([-1, 1], n_samples),
            'Abnormal_URL': np.random.choice([-1, 1], n_samples),
            'Redirect': np.random.choice([-1, 0], n_samples),
            'on_mouseover': np.random.choice([-1, 1], n_samples),
            'RightClick': np.random.choice([-1, 1], n_samples),
            'popUpWindow': np.random.choice([-1, 1], n_samples),
            'Iframe': np.random.choice([-1, 1], n_samples),
            'age_of_domain': np.random.choice([-1, 1], n_samples),
            'DNSRecord': np.random.choice([-1, 1], n_samples),
            'web_traffic': np.random.choice([-1, 0, 1], n_samples),
            'Page_Rank': np.random.choice([-1, 1], n_samples),
            'Google_Index': np.random.choice([-1, 1], n_samples),
            'Links_pointing_to_page': np.random.choice([-1, 0, 1], n_samples),
            'Statistical_report': np.random.choice([-1, 1], n_samples),
        }
        
        # Create target variable (1 for legitimate, -1 for phishing)
        # Use a simple logic based on some features
        # More negative indicators suggest phishing
        result = []
        for i in range(n_samples):
            score = (data['having_IP_Address'][i] + 
                    data['having_At_Symbol'][i] + 
                    data['double_slash_redirecting'][i] +
                    data['Prefix_Suffix'][i])
            # Higher negative score means more phishing indicators
            result.append(-1 if score < -1 else 1)
        
        data['Result'] = result
        
        return pd.DataFrame(data)
    
    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        """
        Split the dataframe into train and test sets
        
        Args:
            dataframe: Input dataframe to split
        """
        try:
            logging.info("Splitting data into train and test sets")
            
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )
            
            logging.info("Train test split completed")
            
            # Create directories
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            # Save train and test data
            train_set.to_csv(
                self.data_ingestion_config.training_file_path,
                index=False,
                header=True
            )
            
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,
                index=False,
                header=True
            )
            
            logging.info(f"Train set shape: {train_set.shape}")
            logging.info(f"Test set shape: {test_set.shape}")
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiate the data ingestion process
        
        Returns:
            DataIngestionArtifact with paths to train and test data
        """
        try:
            logging.info("Starting data ingestion")
            
            # Export data to feature store
            dataframe = self.export_data_into_feature_store()
            
            # Split data into train and test
            self.split_data_as_train_test(dataframe)
            
            # Create and return artifact
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            
            logging.info(f"Data ingestion completed. Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
