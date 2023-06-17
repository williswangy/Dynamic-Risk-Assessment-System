import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import (model_predictions, dataframe_summary, missing_data,
                        execution_time, outdated_packages_list)
import logging
import sys
from training import segregate_dataset,filter_features
from sklearn.metrics import confusion_matrix



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def plot_confusion_matrix():
    """
    Calculate a confusion matrix using the test data and the deployed model
    plot the confusion matrix using seaborn heatmap to the workspace
    """
    logging.info("Loading and preparing testdata.csv")
    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    logging.info("Predicting test data")
    y_pred = model_predictions(test_df)
    _,y_true = segregate_dataset(filter_features(test_df))

    logging.info("Calculating confusion matrix")
    cm = confusion_matrix(y_true, y_pred)

    logging.info("Plotting and saving confusion matrix")
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title("Model Confusion Matrix")
    plt.savefig(os.path.join(model_path, 'confusionmatrix.png'))



##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    pass





if __name__ == '__main__':
    plot_confusion_matrix()
    score_model()
