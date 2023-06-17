import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
from io import StringIO
import subprocess
import logging
import sys

from training import segregate_dataset, filter_features

##################Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

##################Function to get model predictions
def model_predictions(dataset=None):
    logging.info("Running model_predictions function")

    if dataset is None:
        datasetpath = os.path.join(test_data_path, 'testdata.csv')
        dataset = pd.read_csv(datasetpath)
        logging.info("Dataset loaded successfully")

    modelpath = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    with open(modelpath, 'rb') as f:
        model = pickle.load(f)
        logging.info("Model loaded successfully")

    X, y = segregate_dataset(filter_features(dataset))
    logging.info("Dataset segregated successfully")

    yhat = model.predict(X)
    logging.info("Predictions made successfully")
    logging.info("Predictions examined: {}".format(yhat))

    return yhat


##################Function to get summary statistics
def dataframe_summary():
    logging.info("Running dataframe_summary function")

    datasetpath = os.path.join(dataset_csv_path, 'finaldata.csv')
    dataset = pd.read_csv(datasetpath)
    logging.info("Dataset loaded successfully")

    numeric_col_index = np.where(dataset.dtypes != object)[0]
    numeric_col = dataset.columns[numeric_col_index].tolist()

    statistics = {}
    for col in numeric_col:
        statistics[col] = {
            'mean': dataset[col].mean(),
            'median': dataset[col].median(),
            'std': dataset[col].std()
        }

    logging.info("Statistics calculated successfully")
    logging.info("Summary statistics examined: {}".format(statistics))

    return statistics



##################Function to get missing data
def missing_data():
    logging.info("Running missing_data function")

    datasetpath = os.path.join(dataset_csv_path, 'finaldata.csv')
    dataset = pd.read_csv(datasetpath)
    logging.info("Dataset loaded successfully")

    missing_data = dataset.isna().sum(axis=0)
    missing_data /= len(dataset) * 100

    logging.info("Missing data calculated successfully")
    logging.info("Missing data examined: {}".format(missing_data.tolist()))

    return missing_data.tolist()


##################Function to get timings
def execution_time():
    logging.info("Running execution_time function")
    timing_measures = {}

    scripts = ['ingestion.py', 'training.py']

    for script in scripts:
        start_time = timeit.default_timer()
        os.system('python {}'.format(script))
        end_time = timeit.default_timer()

        duration_step = end_time - start_time
        timing_measures[script] = duration_step

        logging.info("{} executed successfully".format(script))

    logging.info("Execution time examined: {}".format(timing_measures))

    return timing_measures


def execute_cmd(cmd):
    logging.info("Running execute_cmd function")

    a = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    b = StringIO(a.communicate()[0].decode('utf-8'))
    df = pd.read_csv(b, sep="\s+")
    df.drop(index=[0], axis=0, inplace=True)
    df = df.set_index('Package')

    logging.info("Command executed successfully")

    return df


##################Function to check dependencies
def outdated_packages_list():
    logging.info("Running outdated_packages_list function")

    cmd = ['pip', 'list', '--outdated']
    df = execute_cmd(cmd)
    df.drop(['Version', 'Type'], axis=1, inplace=True)

    cmd = ['pip', 'list']
    df1 = execute_cmd(cmd)
    df1 = df1.rename(columns={'Version': 'Latest'})

    requirements = pd.read_csv('requirements.txt', sep='==', header=None, names=['Package', 'Version'], engine='python')
    requirements = requirements.set_index('Package')

    dependencies = requirements.join(df1)
    for p in df.index:
        if p in dependencies.index:
            dependencies.at[p, 'Latest'] = df.at[p, 'Latest']

    dependencies.dropna(inplace=True)

    logging.info("Outdated packages list generated successfully")
    logging.info("Outdated packages list examined: {}".format(dependencies))

    return dependencies


if __name__ == '__main__':
    logging.info("Program started")

    model_predictions()
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()

    logging.info("Program completed")
