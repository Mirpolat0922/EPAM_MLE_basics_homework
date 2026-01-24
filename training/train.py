"""
This script prepares the data, runs the training, and saves the model.
"""

import argparse
import os
import sys
import json
import logging
import pandas as pd
import time
from datetime import datetime
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

load_dotenv()

import mlflow
mlflow.autolog()

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

CONF_FILE = os.getenv('CONF_PATH')

from utils import get_project_dir, configure_logging

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", 
                    help="Specify inference data file", 
                    default=conf['train']['table_name'])
parser.add_argument("--model_path", 
                    help="Specify the path for the output model")


class DataProcessor():
    def __init__(self) -> None:
        self.scaler = StandardScaler()

    def prepare_data(self, max_rows: int = None) -> pd.DataFrame:
        logging.info("Preparing data for training...")
        df = self.data_extraction(TRAIN_PATH)
        df = self.data_rand_sampling(df, max_rows)
        return df

    def data_extraction(self, path: str) -> pd.DataFrame:
        logging.info(f"Loading data from {path}...")
        return pd.read_csv(path)
    
    def data_rand_sampling(self, df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
        if not max_rows or max_rows < 0:
            logging.info('Max_rows not defined. Skipping sampling.')
        elif len(df) < max_rows:
            logging.info('Size of dataframe is less than max_rows. Skipping sampling.')
        else:
            df = df.sample(n=max_rows, replace=False, random_state=conf['general']['random_state'])
            logging.info(f'Random sampling performed. Sample size: {max_rows}')
        return df


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

class Training():
    def __init__(self) -> None:
        self.model = IrisNet()
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        logging.info(f"Using device: {self.device}")

    def run_training(self, df: pd.DataFrame, out_path: str = None, test_size: float = 0.33) -> None:
        logging.info("Running training...")
        X_train, X_test, y_train, y_test = self.data_split(df, test_size=test_size)

        # Convert to PyTorch tensors
        X_train_tensor, y_train_tensor = self.prepare_tensors(X_train, y_train, fit_scaler=True)
        X_test_tensor, y_test_tensor = self.prepare_tensors(X_test, y_test, fit_scaler=False)

        start_time = time.time()
        self.train(X_train_tensor, y_train_tensor)
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time:.2f} seconds.")

        self.test(X_test_tensor, y_test_tensor)
        self.save(out_path)

    def data_split(self, df: pd.DataFrame, test_size: float = 0.33) -> tuple:
        logging.info("Splitting data into training and test sets...")
        feature_columns = [col for col in df.columns if col != 'target']
        X = df[feature_columns].values
        y = df['target'].values
        return train_test_split(X, y, test_size=test_size,
                                random_state=conf['general']['random_state'])

    def prepare_tensors(self, X, y, fit_scaler=False):
        """Convert numpy arrays to PyTorch tensors and normalize features"""
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        return X_tensor, y_tensor

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor) -> None:
        logging.info("Training the model...")

        # Training parameters
        epochs = conf['train'].get('epochs', 100)
        learning_rate = conf['train'].get('learning_rate', 0.01)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log every 10 epochs
            if (epoch + 1) % 10 == 0:
                logging.info(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    def test(self, X_test: torch.Tensor, y_test: torch.Tensor) -> float:
        logging.info("Testing the model...")
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            y_pred = predicted.cpu().numpy()
            y_true = y_test.cpu().numpy()
            accuracy = accuracy_score(y_true, y_pred)
            logging.info(f"Test Accuracy: {accuracy:.4f}")
        return accuracy

    def save(self, path: str) -> None:
        logging.info("Saving the model...")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        if not path:
            path = os.path.join(MODEL_DIR, datetime.now().strftime(conf['general']['datetime_format']) + '.pth')
        else:
            path = os.path.join(MODEL_DIR, path)

        # Save model state dict and scaler
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_,
            'input_size': 4,
            'num_classes': 3
        }, path)
        logging.info(f"Model saved to {path}")


def main():
    configure_logging()

    data_proc = DataProcessor()
    tr = Training()

    df = data_proc.prepare_data(max_rows=conf['train']['data_sample'])
    tr.run_training(df, test_size=conf['train']['test_size'])


if __name__ == "__main__":
    main()