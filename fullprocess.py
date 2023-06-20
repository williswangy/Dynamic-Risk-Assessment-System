import json
import os
import ast
from sklearn import metrics
import logging
import pandas as pd

import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion

# Initialize logging
logging.basicConfig(filename='journal.txt',
                    level=logging.INFO,
                    filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y/%m/%d %I:%M:%S %p')


def load_config():
    # Load config.json and get environment variables
    with open('config.json', 'r') as f:
        config = json.load(f)

    return config


def check_new_data(input_folder_path, prod_deployment_path):
    # Check and read new data
    filepath = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
    with open(filepath, 'r') as f:
        ingestedfiles = [line.strip() for line in f.readlines()[1:]]
    files = [file for file in os.listdir(input_folder_path) if file not in ingestedfiles]

    return files


def ingest_new_data(files, input_folder_path, output_folder_path):
    # Ingest new data
    if files:
        logging.info("ingesting new files")
        ingestion.merge_multiple_dataframe(input_folder_path, output_folder_path)
        return True
    else:
        logging.info("No new files - ending process")
        return False


def check_model_drift(move_to_next_step, output_folder_path, prod_deployment_path):
    # Check for model drift
    if move_to_next_step:
        scorespath = os.path.join(prod_deployment_path, 'latestscore.txt')
        with open(scorespath, 'r') as f:
            latest_score = ast.literal_eval(f.read())

        filepath = os.path.join(output_folder_path, 'finaldata.csv')
        dataset = pd.read_csv(filepath)
        new_yhat = diagnostics.model_predictions(dataset)
        new_y = pd.read_csv(filepath)['exited']
        new_score = metrics.f1_score(new_y, new_yhat)
        logging.info(f'latest score: {latest_score}, new score: {new_score}')

        if new_score >= latest_score:
            logging.info('No model drift - ending process')
            return False

    return True


def retrain_model(move_to_next_step):
    # Train new model if there's model drift
    if move_to_next_step:
        logging.info('training new model')
        training.train_model()
        scoring.score_model()


def redeploy_model(move_to_next_step):
    # Redeploy model if new one has been trained
    if move_to_next_step:
        logging.info('deploying new model')
        deployment.store_model_into_pickle()


def run_diagnostics_and_reporting(move_to_next_step):
    # Run diagnostics.py and reporting.py for the re-deployed model
    if move_to_next_step:
        logging.info('producing reporting and calling apis for statistics')
        reporting.score_model()
        os.system('python apicalls.py')


def main():
    logging.info("Launching automated monitoring")

    config = load_config()
    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']
    prod_deployment_path = config['prod_deployment_path']
    model_path = config['output_model_path']

    new_files = check_new_data(input_folder_path, prod_deployment_path)
    move_to_next_step = ingest_new_data(new_files, input_folder_path, output_folder_path)

    if move_to_next_step:
        move_to_next_step = check_model_drift(move_to_next_step, output_folder_path, prod_deployment_path)
        retrain_model(move_to_next_step)
        redeploy_model(move_to_next_step)
        run_diagnostics_and_reporting(move_to_next_step)


if __name__ == '__main__':
    main()
