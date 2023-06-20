import matplotlib.pyplot as plt
import seaborn as sns
import json
from diagnostics import model_predictions
import os
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Table, TableStyle
from reportlab.lib.colors import lavender, red, green
import diagnostics
import logging
import sys
from training import segregate_dataset, filter_features
from sklearn.metrics import confusion_matrix

###############Load config.json and get path variables
with open('config.json', 'r') as f:
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
    _, y_true = segregate_dataset(filter_features(test_df))

    logging.info("Calculating confusion matrix")
    cm = confusion_matrix(y_true, y_pred)

    logging.info("Plotting and saving confusion matrix")
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title("Model Confusion Matrix")
    plt.savefig(os.path.join(model_path, 'confusionmatrix2.png'))


def _get_statistics_df():
    """
    Get data statistics and missing percentage of each column
    in pandas dataframe to draw table in the PDF report

    Returns:
        pd.DataFrame: Train data summary
    """
    stats = diagnostics.dataframe_summary()
    missing = diagnostics.missing_data()

    data = {'Column Name': [k for k in missing.keys()]}
    data['Missing %'] = [missing[column] for column in data['Column Name']]

    temp_col = list(stats.keys())[0]
    for stat in stats[temp_col].keys():
        data[stat] = [
            round(
                stats[column][stat],
                2) if stats.get(
                column,
                None) else '-' for column in data['Column Name']]

    return pd.DataFrame(data)


def generate_pdf_report():
    """
    Generate PDF report that includes ingested data information, model scores
    on test data and diagnostics of execution times and packages
    """
    pdf = canvas.Canvas(
        os.path.join(
            model_path,
            'summary_report.pdf'),
        pagesize=A4)

    pdf.setTitle("Model Summary Report")

    pdf.setFontSize(24)
    pdf.setFillColorRGB(31 / 256, 56 / 256, 100 / 256)
    pdf.drawCentredString(300, 800, "Model Summary Report")

    # Ingest data section
    pdf.setFontSize(18)
    pdf.setFillColorRGB(47 / 256, 84 / 256, 150 / 256)
    pdf.drawString(25, 750, "Ingested Data")

    pdf.setFontSize(14)
    pdf.setFillColorRGB(46 / 256, 116 / 256, 181 / 256)
    pdf.drawString(35, 725, "List of files used:")

    # Ingested files
    with open(os.path.join(dataset_csv_path, "ingestedfiles.txt")) as file:
        pdf.setFontSize(12)
        text = pdf.beginText(40, 705)
        text.setFillColor('black')

        for line in file.readlines():
            text.textLine(line.strip('\n'))

        pdf.drawText(text)

    # Data statistics and missing percentage
    data = _get_statistics_df()
    data_df = pd.DataFrame(data)
    data_table = data_df.values.tolist()
    data_table.insert(0, list(data_df.columns))

    # Draw summary table
    stats_table = Table(data_table)
    stats_table.setStyle([
        ('GRID', (0, 0), (-1, -1), 1, 'black'),
        ('BACKGROUND', (0, 0), (-1, 0), lavender)
    ])

    pdf.setFontSize(14)
    pdf.setFillColorRGB(46 / 256, 116 / 256, 181 / 256)
    pdf.drawString(35, 645, "Statistics Summary")

    stats_table.wrapOn(pdf, 40, 520)
    stats_table.drawOn(pdf, 40, 520)

    # Trained model section
    pdf.setFontSize(18)
    pdf.setFillColorRGB(47 / 256, 84 / 256, 150 / 256)
    pdf.drawString(25, 490, "Trained Model Scoring on Test Data")

    pdf.setFontSize(12)
    pdf.setFillColorRGB(128 / 256, 128 / 256, 128 / 256)
    pdf.drawString(25, 480, "testdata.csv")

    # Model score
    with open(os.path.join(model_path, "latestscore.txt")) as file:
        pdf.setFontSize(12)
        pdf.setFillColor('black')
        pdf.drawString(40, 460, file.read())

    # Model confusion matrix
    pdf.drawInlineImage(
        os.path.join(
            model_path,
            'confusionmatrix2.png'),
        40,
        150,
        width=300,
        height=300)

    # New page
    pdf.showPage()

    # Diagnostics section
    pdf.setFontSize(18)
    pdf.setFillColorRGB(47 / 256, 84 / 256, 150 / 256)
    pdf.drawString(25, 780, "Diagnostics")

    # Execution time
    timings = diagnostics.execution_time()

    pdf.setFontSize(14)
    pdf.setFillColorRGB(46 / 256, 116 / 256, 181 / 256)
    pdf.drawString(35, 755, "Execution times:")

    pdf.setFontSize(12)
    text = pdf.beginText(40, 735)
    text.setFillColor('black')

    for k, v in timings.items():
        text.textLine(f"{k} = {round(v, 4)}")

    pdf.drawText(text)

    dependencies = diagnostics.outdated_packages_list()
    # Convert the DataFrame to a list of lists
    data = [dependencies.columns.tolist()] + dependencies.values.tolist()

    table_style = TableStyle()
    table_style.add('GRID', (0, 0), (-1, -1), 1, 'black')
    table_style.add('BACKGROUND', (0, 0), (-1, 0), lavender)

    for row, values in enumerate(data[1:], start=1):
        if values[0] != values[1]:  # compare the 'Version' and 'Latest' values
            table_style.add('TEXTCOLOR', (1, row), (1, row), red)
            table_style.add('TEXTCOLOR', (2, row), (2, row), green)

    depend_table = Table(data)
    depend_table.setStyle(table_style)

    pdf.setFontSize(14)
    pdf.setFillColorRGB(46 / 256, 116 / 256, 181 / 256)
    pdf.drawString(300, 690, "Outdated Dependencies")  # Change here

    depend_table.wrapOn(pdf, 300, 325)  # And here
    depend_table.drawOn(pdf, 300, 325)  # And here

    pdf.save()


##############Function for reporting
def score_model():
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace
    plot_confusion_matrix()
    generate_pdf_report()


if __name__ == '__main__':
    score_model()
