# Heart Disease Prediction System Using Data Mining

This project implements a comprehensive machine learning pipeline for predicting heart disease using various classification algorithms. It includes data preprocessing, model training, evaluation, and deployment stages.

## Features

- **Data Preprocessing**: Cleans and transforms raw datasets for machine learning models.
- **Multiple Algorithms**: Implements and compares Logistic Regression, Decision Tree, Random Forest, SVM, and Neural Network.
- **Cross-Validation**: Uses K-Fold cross-validation (k=10) to ensure robust model evaluation.
- **Cost-Sensitive Learning**: Implements cost functions to handle imbalanced datasets and prioritize certain types of errors.
- **Deployment**: Dockerized deployment with FastAPI, including RESTful APIs for predictions.
- **Monitoring**: Basic logging and metrics tracking.

## Project Structure

```
heart-disease-ml-pipeline/
├── Data/
│   ├── Raw Datasets: Original data files.
│   ├── Processed Datasets: Cleaned and prepared data.
│   └── Cleveland Data: Main dataset used for most experiments.
├── models/
│   ├── Configs/: Configuration files for models (e.g., neural network layers).
│   ├── cost_sensitive_learning.py: Cost-sensitive classification implementation.
│   ├── load_and_prepare_data.py: Data loading and preprocessing logic.
│   ├── model_training.py: Core training and evaluation logic.
│   ├── run_experiments.py: Script to run all experiments.
│   ├── train_neural_network.py: Specific training script for Neural Network.
│   └── train_random_forest_cs.py: Specific script for Cost-Sensitive Random Forest.
├── costs/: Custom cost matrices for different scenarios.
├── Dockerfile: Container configuration for deployment.
├── requirements.txt: Python dependencies.
└── README.md: This file.
```

## Getting Started

### Prerequisites

- Python 3.6+
- Docker (for deployment)

### Installation & Setup

1. **Clone the repository** (or download the files).
2. **Navigate** to the project directory:
   ```bash
   cd heart-disease-ml-pipeline
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Run Experiments

To train all models and evaluate them using cross-validation:

```bash
python models/run_experiments.py
```

This will output performance metrics for all algorithms on the Cleveland dataset.