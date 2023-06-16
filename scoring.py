import os
import json
import pickle
import pandas as pd
import logging
from sklearn import metrics
from training import segregate_dataset,filter_features
import sys

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

#################Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])

# import test dataset
filepath = os.path.join(test_data_path, 'testdata.csv')
testdata = pd.read_csv(filepath)

def load_model(model_path: str):
    """Loads the trained model from disk"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logging.info("Model loaded successfully")
        return model
    except Exception as e:
        logging.error(f"Error occurred while loading model: {e}")
        raise

#################Function for model scoring
def score_model():
    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file

    # load trained model
    modelpath = os.path.join(model_path, 'trainedmodel.pkl')
    model = load_model(modelpath)

    # segregate test dataset
    X, y = segregate_dataset(filter_features(testdata))

    # evaluate model on test set
    logging.info("Evaluating model on test set...")
    yhat = model.predict(X)
    score = metrics.f1_score(y, yhat)
    logging.info(f"Computed F1 score: {score}")

    # save as latest score
    scorespath = os.path.join(model_path, 'latestscore.txt')
    with open(scorespath, 'w') as f:
        f.write(str(score))
    logging.info(f"F1 score computed and saved to {scorespath}")

    return score


if __name__ == '__main__':
    score_model()
