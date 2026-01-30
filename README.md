# Network Security With ML - Phishing Website Detection

A machine learning system that predicts whether a website is phishing or legitimate. This project is useful for the cyber security domain where spam or malicious websites can be identified based on certain indicators.

## Features

- **Data Pipeline**: Complete ETL pipeline with data ingestion, validation, and transformation
- **Multiple ML Models**: Trains and compares multiple classification models (Random Forest, Decision Tree, Gradient Boosting, Logistic Regression)
- **Data Drift Detection**: Monitors data drift using Evidently AI
- **Modular Architecture**: Clean, modular code structure with proper logging and exception handling
- **Easy to Use**: Simple scripts for training and prediction

## Installation

1. Clone the repository:
```bash
git clone https://github.com/NiceML/Network-Security-With-ML.git
cd Network-Security-With-ML
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the phishing detection model:

```bash
python train.py
```

This will:
- Load or create sample phishing data
- Perform data validation and transformation
- Train multiple ML models and select the best one
- Save the trained model and preprocessor

### Making Predictions

To see a demo of predictions:

```bash
python demo.py
```

This will load the trained model and make predictions on sample websites.

### Using the Model in Your Code

```python
from networksecurity.pipeline.prediction_pipeline import PredictionPipeline
import pandas as pd

# Define paths to your trained model
model_path = "artifacts/.../model_trainer/trained_model/model.pkl"
preprocessor_path = "artifacts/.../data_transformation/transformed_object/preprocessing.pkl"

# Create prediction pipeline
pipeline = PredictionPipeline(model_path, preprocessor_path)

# Prepare your data (30 features as shown in sample data)
data = pd.DataFrame({...})

# Make predictions
predictions = pipeline.predict(data)
# -1 means PHISHING, 1 means LEGITIMATE
```

## Project Structure

```
Network-Security-With-ML/
├── networksecurity/           # Main package
│   ├── components/           # ML pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/             # Training and prediction pipelines
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   ├── entity/               # Configuration and artifact entities
│   ├── exception/            # Custom exception handling
│   ├── logging/              # Logging configuration
│   ├── utils/                # Utility functions
│   └── constants/            # Constants and configuration
├── train.py                  # Script to train the model
├── demo.py                   # Demo script for predictions
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Features in the Dataset

The system uses 30 features to detect phishing websites:

1. `having_IP_Address` - Whether URL has IP address
2. `URL_Length` - Length of the URL
3. `Shortening_Service` - Use of URL shortening service
4. `having_At_Symbol` - Presence of @ symbol
5. `double_slash_redirecting` - Double slash redirecting
6. `Prefix_Suffix` - Prefix or suffix in domain
7. `having_Sub_Domain` - Number of sub-domains
8. `SSLfinal_State` - SSL certificate state
9. `Domain_registeration_length` - Domain registration length
10. `Favicon` - Favicon loaded from different domain
... and 20 more features

## Model Performance

The system trains multiple models and selects the best one based on F1 score. Expected performance:
- F1 Score: > 0.6
- Precision: > 0.6
- Recall: > 0.6

## Logging

All operations are logged to the `logs/` directory with timestamps.

## Artifacts

Trained models and intermediate outputs are saved in the `artifacts/` directory:
- Feature store data
- Validated data
- Transformed data
- Trained models
- Data drift reports

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms in the LICENSE file.
