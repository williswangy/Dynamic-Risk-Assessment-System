import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging
import sys

#############Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def ensure_directory_exists(dir_path: str):
    """
    Function to ensure that a directory exists.
    If the directory does not exist, create it.

    :param dir_path: str, directory path to check/create
    :return: None
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def merge_multiple_dataframe(input_folder_path: str, output_folder_path: str):
    """
    Function for data ingestion.
    Check for datasets in input_folder_path, combine them together,
    drops duplicates and write metadata ingestedfiles.txt and ingested data
    to finaldata.csv in output_folder_path.

    :param input_folder_path: str, path where the input csv files are stored
    :param output_folder_path: str, path where the output files will be stored
    :return: None
    """
    dfs = []  # list to store DataFrames
    file_names = []

    logging.info(f"Reading files from {input_folder_path}")
    for file in os.listdir(input_folder_path):
        if file.endswith('.csv'):  # Ensure we're working with a .csv file
            file_path = os.path.join(input_folder_path, file)
            try:
                df_tmp = pd.read_csv(file_path)
                dfs.append(df_tmp)  # append DataFrame to list
            except pd.errors.EmptyDataError:
                logging.warning(f"No columns to parse from file {file_path}")
                continue  # Skip this file

            file = os.path.join(*file_path.split(os.path.sep)[-3:])
            file_names.append(file)

    df = pd.concat(dfs, ignore_index=True)  # concatenate all DataFrames in the list

    logging.info("Dropping duplicates")
    df = df.drop_duplicates().reset_index(drop=1)

    logging.info("Saving ingested metadata")
    ensure_directory_exists(output_folder_path)  # Ensure the output directory exists before writing to it
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), "w") as file:
        file.write(f"Ingestion date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        file.write("\n".join(file_names))

    logging.info("Saving ingested data")
    df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)


if __name__ == '__main__':
    logging.info("Running ingestion.py")
    merge_multiple_dataframe(input_folder_path, output_folder_path)
