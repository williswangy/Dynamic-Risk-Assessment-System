import os
import json
import pickle
import logging
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression

###################Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = config['output_folder_path']
model_path = config['output_model_path']
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Define the features to keep



def load_dataset(filepath):
    """Load dataset from csv file"""
    logging.info(f"Loading dataset from {filepath}")
    return pd.read_csv(filepath)


def filter_features(dataset):
    """Filter the dataset for necessary features"""
    FEATURES = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees', 'exited']
    logging.info(f"Filtering features {FEATURES}")
    return dataset[FEATURES]


def segregate_dataset(dataset):
    """Segregate the dataset into feature dataframe and target series"""
    logging.info(f"Segregating dataset into features and target...")
    X = dataset.drop(columns='exited')
    y = dataset['exited']
    return X, y


def train_model(X, y, model_config):
    """Train the model using the training data"""
    logging.info(f"Training model with configuration: {model_config}")
    model = LogisticRegression(**model_config)
    model.fit(X, y)
    return model


def save_model(model, filepath):
    """Save the trained model to disk"""
    logging.info(f"Saving model to {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    logging.info(f"Starting model training...")
    dataset = load_dataset(os.path.join(dataset_csv_path, "finaldata.csv"))
    dataset = filter_features(dataset)
    X, y = segregate_dataset(dataset)

    # Get the model config from `config.json`
    model_config = config['model_config']
    model = train_model(X, y, model_config)

    save_model(model, os.path.join(model_path, 'trainedmodel.pkl'))

    logging.info(f"Model training completed.")


if __name__ == '__main__':
    main()
