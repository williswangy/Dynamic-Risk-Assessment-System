from flask import Flask, session, jsonify, request
import pandas as pd
import json
import os
from diagnostics import model_predictions
from scoring import score_model
import diagnostics



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():
    #call the prediction function you created in Step 3
    if request.method == 'POST':
        file = request.files['filename']
        dataset = pd.read_csv(file)
        return model_predictions(dataset)
    if request.method == 'GET':
        file = request.args.get('filename')
        dataset = pd.read_csv(file)
        return {'predictions': str(model_predictions(dataset))}

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def get_score():
    #check the score of the deployed model
    return {'F1 score': score_model()}

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def get_stats():
    # fetch the summary data
    summary_data = diagnostics.dataframe_summary()

    # filter out the stats for the relevant columns
    return {'key statistics': {c: summary_data[c] for c in ['lastmonth_activity','lastyear_activity','number_of_employees'] if c in summary_data}}


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def get_diagnostics():
    # calculate missing data
    missing_data_percentages = diagnostics.missing_data()
    missing_data_dict = {
        'lastmonth_activity': missing_data_percentages[0],
        'lastyear_activity': missing_data_percentages[1],
        'number_of_employees': missing_data_percentages[2],
        'exited': missing_data_percentages[3]
    }

    # calculate execution times
    timing = diagnostics.execution_time()
    execution_time_dict = {
        'ingestion step': timing['ingestion.py'],
        'training step': timing['training.py']
    }

    # check outdated packages
    outdated_packages_df = diagnostics.outdated_packages_list()
    dependency_check = []
    for row in outdated_packages_df.iterrows():
        dependency_check.append({
            'Module': row[0],
            'Version': row[1]['Version'],
            'Vlatest': row[1]['Latest']
        })

    return {
        'execution time': execution_time_dict,
        'missing data': missing_data_dict,
        'dependency check': dependency_check
    }


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
