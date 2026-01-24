# Iris Flower Classification with PyTorch

## Project Overview
This project implements a deep learning solution for classifying Iris flowers into three species (Setosa, Versicolor, and Virginica) using PyTorch. The project demonstrates a complete machine learning pipeline from data preparation to model training and inference, all containerized with Docker for reproducibility and easy deployment.

## Dataset
The project uses the famous **Iris flower dataset** from the UCI Machine Learning Repository. The dataset contains:
- **150 samples** of iris flowers
- **4 features**: sepal length, sepal width, petal length, petal width (all in cm)
- **3 classes**: Setosa (0), Versicolor (1), Virginica (2)

## Model Architecture
A simple feedforward neural network implemented in PyTorch:
- **Input Layer**: 4 features
- **Hidden Layer 1**: 8 neurons with ReLU activation
- **Hidden Layer 2**: 4 neurons with ReLU activation
- **Output Layer**: 3 neurons (one for each class)
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (learning rate: 0.01)
- **Training**: 100 epochs

## Project Structure
```
MLE_basic_example
├── data/                          # Data files for training and inference(it can be generated with data_preparation.py script)
│   ├── iris_train_data.csv
│   └── iris_inference_data.csv
├── data_process/                  # Data processing scripts
│   ├── data_preparation.py
│   └── __init__.py           
├── inference/                     # Script and Dockerfile used for inference
│   ├── Dockerfile
│   ├── run.py
│   └── __init__.py
├── models/                        # Folder where trained models are stored(it is generated with train.py script)
│   └── *.pth
├── results/                       # Prediction outputs (it is generated with run.py script)
│   └── *.csv
├── training/                      # Script and Dockerfile used for training
│   ├── Dockerfile
│   ├── train.py
│   └── __init__.py
├── utils.py                       # Utility functions and classes that are used in scripts
├── settings.json                  # All configurable parameters and settings
├── requirements.txt               # Python dependencies
├── unittests.py                  # Unit tests for data processing and model
├── .gitignore                    # Git ignore file
├── .env                          # Environment variables (CONF_PATH)(you should create it)
└── README.md                     # README file that gives overall information about the project
```

## Prerequisites

Before running this project, ensure you have the following installed:

- **Docker Desktop** - For containerized training and inference
- **Python 3.10+** - For local execution

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Mirpolat0922/EPAM_MLE_basics_homework.git
cd MLE_basic_example
```

### 2. Set Up Environment Variables
Create a `.env` file in the root directory:
```bash
echo "CONF_PATH=settings.json" > .env
```

### 3. Install Dependencies (for local execution)
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Data Preparation

Generate training and inference datasets from the Iris dataset:
```bash
python data_process/data_preparation.py
```

**What this does:**
- Loads the Iris dataset from sklearn
- Splits it into 80% training (120 samples) and 20% inference (30 samples)
- Saves `iris_train_data.csv` (with labels) and `iris_inference_data.csv` (without labels)

### Step 2: Model Training

#### Option A: Train with Docker

1. **Build the training Docker image:**
```bash
docker build -f ./training/Dockerfile --build-arg settings_name=settings.json -t iris_training .
```

This will automatically:
- Install all dependencies
- Train the model for 100 epochs
- Save the model inside the container

2. **Extract the trained model from the container:**
```bash
# Run container interactively
docker run -it iris_training /bin/bash

# Inside container, check the models
ls /app/models/

# Exit and copy from another terminal
docker cp <container_id>:/app/models/<model_name>.pth ./models/
```

Replace `<container_id>` with your running Docker container ID and `<model_name>.pth` with your model's name.

#### Option B: Train Locally
```bash
python training/train.py
```

### Step 3: Run Inference

#### Option A: Inference with Docker

1. **Build the inference Docker image:**
```bash
docker build -f ./inference/Dockerfile \
  --build-arg model_name=latest_model.pth \
  --build-arg settings_name=settings.json \
  -t iris_inference .
```

2. **Run the inference container:**
```bash
docker run \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/input \
  -v $(pwd)/results:/app/output \
  iris_inference
```

**Or run interactively:**
```bash
docker run -it iris_inference /bin/bash
# Inside container:
python3 inference/run.py
```

#### Option B: Inference Locally
```bash
python inference/run.py
```

### Step 4: View Results

Check the `results/` folder for prediction outputs with timestamp:
```bash
cat results/25.01.2026_15.45.csv
```

## Running Tests

Run unit tests to verify functionality:
```bash
python -m unittest unittests.py
```

**Tests included:**
- Data extraction and loading
- Data sampling functionality
- Model instantiation
- Forward pass with correct tensor shapes
- Training capability with sample data

## MLFlow Tracking

The project includes MLFlow for experiment tracking:

1. **Start MLFlow UI:**
```bash
mlflow ui
```

2. **Access the UI:**
Open your browser and navigate to `http://localhost:5000`

3. **View Experiments:**
- Training metrics (loss, accuracy)
- Model parameters
- Run comparisons

**Note:** If you have problems with MLFlow installation, comment out these lines in `train.py` and `requirements.txt`:
```python
# import mlflow
# mlflow.autolog()
```

## Model Performance

Typical results on Iris dataset:
- **Training Accuracy**: 100%
- **Test Accuracy**: ~100%
- **Training Time**: 2-3 seconds
- **Inference Time**: <1 second for 30 samples