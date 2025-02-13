# Import Libraries
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import init_paths
from src.utils.json_utils import save_dict_as_json

from global_config import DIRECTORY_TO_SAVE_ROOT, DATABASES_PATH

def accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--experiment_name', type=str, default = 'DL_BCI_fairness')
parser.add_argument('--model_to_train', type=str, default = 'EEGNetv4_SM')
parser.add_argument('--dataset_name', type=str,default = 'Cho2017')
parser.add_argument('--session', type=int, default = 1)
parser.add_argument('--timestamp', type=str, default = '20241004_144153')
args = parser.parse_args()

# Charge the configuration of the script
experiment_name = args.experiment_name
model_to_train = args.model_to_train
dataset_name = args.dataset_name
session = args.session
timestamp = args.timestamp
script_name = 'ROC_curves_by_test_subject'
# Path to load and save the results
path_to_load = os.path.join(DIRECTORY_TO_SAVE_ROOT, f'{experiment_name}', f'{model_to_train}', f'{dataset_name}_{session}', f'{timestamp}')
path_to_save = os.path.join(DIRECTORY_TO_SAVE_ROOT, f'{experiment_name}', f'{model_to_train}', f'{dataset_name}_{session}', f'{timestamp}', 'plots', script_name)
os.makedirs(path_to_save, exist_ok=True)

# Load the configuration of the script
script_progress = pd.read_csv(os.path.join(path_to_load, 'script_progress.csv'))

columns = script_progress.columns.to_list()
columns.remove('subject_id')
columns.remove('completed')

# Initialize variables
actual_subject = script_progress['subject_id'].values[0]
end_row = script_progress.iloc()[-1]
fpr_subject = []
tpr_subject = []
auc_subject = []

fig_subjects, ax_subjects = plt.subplots(figsize=(10, 10))

data_actual_subject = {}
i=0

# Iterate over each row in the script progress dataframe
for index, row in script_progress.iterrows():
    new_subject = row['subject_id']

    if index == end_row.name:
        ax_subjects.plot([0, 1], [0, 1], 'k--') # Diagonal line 
        ax_subjects.set_xlabel('False Positive Rate')
        ax_subjects.set_ylabel('True Positive Rate')
        ax_subjects.set_title(f'ROC Curves for subject {actual_subject}')
        ax_subjects.legend(loc="lower right")
        ax_subjects.grid()
        # Add to the plot the AUC value
        ax_subjects.text(0.6, 0.2, f'AUC = {np.mean(auc_subject):.2f}', fontsize=12, color='red')
        # Save the plot
        path_to_save_subject = os.path.join(path_to_save, f'subject_{actual_subject}')
        os.makedirs(path_to_save_subject, exist_ok=True)
        fig_subjects.savefig(os.path.join(path_to_save_subject, f'ROC_Curves_{actual_subject}.png')) 
        plt.close()

        # Save the data
        data_actual_subject = save_dict_as_json(path_to_save_subject,
                                                data_actual_subject, 
                                                f'ROC_Curves_{actual_subject}.json')
    elif new_subject == actual_subject:
        # Generate the subject name
        ss = f"subject_{row['subject_id']}"

        # Generate a folder name based on the values in each column
        folder_name = ""
        for column in columns:
            folder_name += f"_{column}_{row[column]}"
        folder_name = folder_name[1:]
        
        # Print the progress of loading the training history for a specific subject and folder
        print(f"Loading the outputs for the subject {row['subject_id']} and {folder_name}")

        path_to_load_files = os.path.join(path_to_load, ss, folder_name)
        file = "test_outputs.csv"

        test_outputs = pd.read_csv(os.path.join(path_to_load_files, file))
        y_true = test_outputs['true_labels'].values
        proba_class_1 = test_outputs['proba_class_1'].values
        
        # ROC CURVE
        fpr, tpr, thresholds = roc_curve(y_true, proba_class_1)
        auc_value = auc(fpr, tpr)
        fpr_subject.append(fpr)
        tpr_subject.append(tpr)
        auc_subject.append(auc_value)

        data_actual_subject[folder_name] = {'auc': auc_value, 'fpr': fpr, 'tpr': tpr, 'thresholds':thresholds}
        # Plot the ROC curve
        ax_subjects.plot(fpr, tpr, color="blue", alpha=0.25) 

    else:
        ax_subjects.plot([0, 1], [0, 1], 'k--') # Diagonal line 
        ax_subjects.set_xlabel('False Positive Rate')
        ax_subjects.set_ylabel('True Positive Rate')
        ax_subjects.set_title(f'ROC Curves for subject {actual_subject}')
        ax_subjects.legend(loc="lower right")
        ax_subjects.grid()
        # Add to the plot the AUC value
        ax_subjects.text(0.6, 0.2, f'AUC = {np.mean(auc_subject):.2f}', fontsize=12, color='red')
        # Save the plot
        path_to_save_subject = os.path.join(path_to_save, f'subject_{actual_subject}')
        os.makedirs(path_to_save_subject, exist_ok=True)
        fig_subjects.savefig(os.path.join(path_to_save_subject, f'ROC_Curves_{actual_subject}.png')) 
        plt.close()

        # Save the data
        data_actual_subject = save_dict_as_json(path_to_save_subject,
                                                data_actual_subject, 
                                                f'ROC_Curves_{actual_subject}.json')

        # Initialize variables
        actual_subject = new_subject
        fpr_subject = []
        tpr_subject = []
        auc_subject = []
        data_actual_subject = {}
        fig_subjects, ax_subjects = plt.subplots(figsize=(10, 10))

        # Generate the subject name
        ss = f"subject_{row['subject_id']}"

        # Generate a folder name based on the values in each column
        folder_name = ""
        for column in columns:
            folder_name += f"_{column}_{row[column]}"
        folder_name = folder_name[1:]
        
        # Print the progress of loading the training history for a specific subject and folder
        print(f"Loading the outputs for the subject {row['subject_id']} and {folder_name}")

        path_to_load_files = os.path.join(path_to_load, ss, folder_name)
        file = "test_outputs.csv"

        test_outputs = pd.read_csv(os.path.join(path_to_load_files, file))
        y_true = test_outputs['true_labels'].values
        proba_class_1 = test_outputs['proba_class_1'].values
        
        # ROC CURVE
        fpr, tpr, thresholds = roc_curve(y_true, proba_class_1)
        auc_value = auc(fpr, tpr)
        fpr_subject.append(fpr)
        tpr_subject.append(tpr)
        auc_subject.append(auc_value)

        data_actual_subject[folder_name] = {'auc': auc_value, 'fpr': fpr, 'tpr': tpr, 'thresholds':thresholds}
        # Plot the ROC curve
        ax_subjects.plot(fpr, tpr, color="blue", alpha=0.25) 






