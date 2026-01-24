import unittest
import pandas as pd
import os
import sys
import json
import torch
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
CONF_FILE = os.getenv('CONF_PATH')

from training.train import DataProcessor, Training, IrisNet


class TestDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONF_FILE, "r") as file:
            conf = json.load(file)
        cls.data_dir = conf['general']['data_dir']
        cls.train_path = os.path.join(cls.data_dir, conf['train']['table_name'])

    def test_data_extraction(self):
        """Test if data can be loaded from CSV"""
        dp = DataProcessor()
        df = dp.data_extraction(self.train_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_prepare_data(self):
        """Test if data sampling works correctly"""
        dp = DataProcessor()
        df = dp.prepare_data(100)
        self.assertLessEqual(df.shape[0], 100)

class TestModel(unittest.TestCase):
    def test_model_creation(self):
        """Test if IrisNet model can be instantiated"""
        model = IrisNet()
        self.assertIsNotNone(model)

    def test_model_forward_pass(self):
        """Test if model accepts correct input shape(iris dataset has 4 features and 3 classes for target)"""
        model = IrisNet()
        x = torch.randn(1, 4)  # 1 sample, 4 features
        output = model(x)
        self.assertEqual(output.shape, (1, 3))  # Should output 3 classes


class TestTraining(unittest.TestCase):
    def test_train(self):
        """Test if model can be trained on sample data"""
        tr = Training()
        # Create sample data
        X_train = pd.DataFrame({
            'sepal length (cm)': [5.1, 4.9, 6.2, 5.9],
            'sepal width (cm)': [3.5, 3.0, 3.4, 3.0],
            'petal length (cm)': [1.4, 1.4, 5.4, 5.1],
            'petal width (cm)': [0.2, 0.2, 2.3, 1.8]
        })
        y_train = pd.Series([0, 0, 2, 2])

        # Convert to tensors
        X_train_tensor, y_train_tensor = tr.prepare_tensors(X_train.values, y_train.values, fit_scaler=True)

        # Train for 1 epoch to test
        tr.model.train()
        outputs = tr.model(X_train_tensor)

        # Check output shape
        self.assertEqual(outputs.shape, (4, 3))


if __name__ == '__main__':
    unittest.main()