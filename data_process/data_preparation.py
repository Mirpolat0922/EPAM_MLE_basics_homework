# Importing required libraries
import numpy as np
import pandas as pd
import logging
import os
import sys
import json
from dotenv import load_dotenv

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from utils import singleton, get_project_dir, configure_logging

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

load_dotenv()

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.getenv('CONF_PATH')

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining paths...")
DATA_DIR = get_project_dir(conf['general']['data_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
INFERENCE_PATH = os.path.join(DATA_DIR, conf['inference']['inp_table_name'])

# Singleton class for generating XOR data set
@singleton
class IrisDatasetProcessor():
    def __init__(self):
        self.df = None

    # Method to prepare iris dataset
    def create(self, train_path: os.path, inference_path: os.path, test_size: float = 0.2):

        logger.info("Loading Iris dataset...")
        self.df = self._load_iris_dataset()

        logger.info("Splitting data into train and inference sets...")
        train_df, inference_df = self._split_data(self.df, test_size)

        logger.info("Saving train data...")
        self.save(train_df, train_path, is_labeled=True)

        logger.info("Saving inference data...")
        self.save(inference_df, inference_path, is_labeled=False)

        return train_df, inference_df

    # Method to load iris data
    def _load_iris_dataset(self):
        logger.info("Loading Iris dataset...")
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df["target"] = iris.target
        return df

    # Method to split dataset to train and test.
    def _split_data(self, df: pd.DataFrame, test_size: float = 0.2):
        logger.info("Splitting dataset...")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=conf['general']['random_state'])
        return train_df, test_df

    # Method to save data
    def save(self, df: pd.DataFrame, out_path: os.path, is_labeled: bool = True):
        logger.info(f"Saving data to {out_path}...")
        if not is_labeled:
            df = df.drop('target', axis=1)  # Remove labels for inference
        df.to_csv(out_path, index=False)

# Main execution
if __name__ == "__main__":
    configure_logging()
    logger.info("Starting script...")
    processor = IrisDatasetProcessor()
    processor.create(train_path=TRAIN_PATH, inference_path=INFERENCE_PATH, test_size=0.2)
    logger.info("Script completed successfully.")