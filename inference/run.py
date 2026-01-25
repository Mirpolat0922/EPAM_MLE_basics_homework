"""
Script loads the latest trained model, data for inference and predicts results.
Imports necessary packages and modules.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import List
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.getenv('CONF_PATH')

from utils import get_project_dir, configure_logging

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
RESULTS_DIR = get_project_dir(conf['general']['results_dir'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--infer_file", 
                    help="Specify inference data file", 
                    default=conf['inference']['inp_table_name'])
parser.add_argument("--out_path", 
                    help="Specify the path to the output table")

class IrisNet(nn.Module):
    def __init__(self, input_size=4, hidden_size1=8, hidden_size2=4, num_classes=3):
        super(IrisNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def get_latest_model_path() -> str:
    """Gets the path of the latest saved model"""
    latest = None
    for (dirpath, dirnames, filenames) in os.walk(MODEL_DIR):
        for filename in filenames:
            if filename.endswith('.pth'):
                if not latest or datetime.strptime(latest.replace('.pth', ''), conf['general']['datetime_format']) < \
                        datetime.strptime(filename.replace('.pth', ''), conf['general']['datetime_format']):
                    latest = filename
    return os.path.join(MODEL_DIR, latest)


def get_model_by_path(path: str):
    """Loads and returns the specified PyTorch model"""
    try:
        # Load checkpoint
        checkpoint = torch.load(path, map_location=torch.device('cpu'))

        # Initialize model
        model = IrisNet(
            input_size=checkpoint.get('input_size', 4),
            num_classes=checkpoint.get('num_classes', 3)
        )

        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set to evaluation mode

        # Load scaler parameters
        scaler = StandardScaler()
        scaler.mean_ = checkpoint['scaler_mean']
        scaler.scale_ = checkpoint['scaler_scale']

        logging.info(f'Model loaded from: {path}')
        return model, scaler
    except Exception as e:
        logging.error(f'An error occurred while loading the model: {e}')
        sys.exit(1)


def get_inference_data(path: str) -> pd.DataFrame:
    """loads and returns data for inference from the specified csv file"""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.error(f"An error occurred while loading inference data: {e}")
        sys.exit(1)


def predict_results(model: IrisNet, scaler: StandardScaler, infer_data: pd.DataFrame) -> pd.DataFrame:
    """Predict the results and join it with the infer_data"""
    try:
        # Convert to numpy array
        X = infer_data.values

        # Normalize using scaler
        X_scaled = scaler.transform(X)

        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_scaled)

        # Make predictions
        with torch.no_grad():
            outputs = model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            results = predicted.numpy()

        # Add predictions to dataframe
        infer_data['results'] = results
        logging.info("Predictions completed successfully")

        return infer_data

    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}")
        sys.exit(1)


def store_results(results: pd.DataFrame, path: str = None) -> None:
    """Store the prediction results in 'results' directory with current datetime as a filename"""
    if not path:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        path = datetime.now().strftime(conf['general']['datetime_format']) + '.csv'
        path = os.path.join(RESULTS_DIR, path)
    pd.DataFrame(results).to_csv(path, index=False)
    logging.info(f'Results saved to {path}')


def main():
    """Main function"""
    configure_logging()
    args = parser.parse_args()

    model, scaler = get_model_by_path(get_latest_model_path())
    infer_file = args.infer_file
    infer_data = get_inference_data(os.path.join(DATA_DIR, infer_file))
    results = predict_results(model,scaler, infer_data)
    store_results(results, args.out_path)


if __name__ == "__main__":
    main()