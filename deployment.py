import os
import json
import shutil
import logging
import sys

##################Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


####################function for deployment
def store_model_into_pickle():
    os.makedirs(prod_deployment_path, exist_ok=True)
    try:
        # copy the latest pickle file and its latestscore.txt file into the deployment directory
        for file in ['latestscore.txt', 'trainedmodel.pkl']:
            src = os.path.join(model_path, file)
            dst = os.path.join(prod_deployment_path, file)
            if os.path.exists(src):
                shutil.copy(src, dst)
                logging.info(f"Copied {src} to {dst}")
            else:
                logging.error(f"{src} not found.")

        # copy the ingestfiles.txt file into the deployment directory
        src = os.path.join(dataset_csv_path, 'ingestedfiles.txt')
        dst = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
        if os.path.exists(src):
            shutil.copy(src, dst)
            logging.info(f"Copied {src} to {dst}")
        else:
            logging.error(f"{src} not found.")
    except Exception as e:
        logging.error(f"Error during deployment: {str(e)}")


if __name__ == '__main__':
    store_model_into_pickle()
