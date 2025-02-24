# Import Libraries
import os
import argparse
import pandas as pd
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import init_paths

from src.utils.json_utils import load_json_file
from sklearn.metrics import roc_auc_score
DIRECTORY_TO_SAVE_ROOT = os.path.join(os.getcwd(), "RESULTS")
# from global_config import DIRECTORY_TO_SAVE_ROOT

def accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


parser = argparse.ArgumentParser(description='')
parser.add_argument('--experiment_name', type=str, default = 'DL_BCI_fairness')
parser.add_argument('--model_to_train', type=str, default = 'EEGNetv4_SM')
parser.add_argument('--dataset_name', type=str,default = 'Cho2017')
parser.add_argument('--session', type=int, default = 1)
parser.add_argument('--timestamp', type=str, default = '20250222_111051')
args = parser.parse_args()

# Charge the configuration of the script
experiment_name = args.experiment_name
model_to_train = args.model_to_train
dataset_name = args.dataset_name
session = args.session
timestamp = args.timestamp

# Path to load and save the results
path_to_load = os.path.join(DIRECTORY_TO_SAVE_ROOT, f'{experiment_name}', f'{model_to_train}', f'{dataset_name}_{session}', f'{timestamp}')
path_to_save = os.path.join(DIRECTORY_TO_SAVE_ROOT, f'{experiment_name}', f'{model_to_train}', f'{dataset_name}_{session}', f'{timestamp}', 'different_analysis', 'compute_metrics_from_outputs')
os.makedirs(path_to_save, exist_ok=True)

# Load the configuration of the script
script_config = load_json_file(os.path.join(path_to_load, 'config.json'))
script_progress = pd.read_csv(os.path.join(path_to_load, 'script_progress.csv'))
# Print the configuration
print("Configuration of the script:")
print(script_config)
print("Script progress:")
print(script_progress)

columns = script_progress.columns.to_list()
columns.remove('subject_id')
columns.remove('completed')

split_seed_flag = True if 'split_seed' in columns else False

# Initialize variables
actual_subject = script_progress['subject_id'].values[0]
data_actual_subject = {}
i=0

metrics = pd.DataFrame(columns= ['subject_id'] + columns + ['subset', 'accuracy', 'auc'])

# Iterate over each row in the script progress dataframe
for index, row in script_progress.iterrows():
    # Generate the subject name
    ss = f"subject_{row['subject_id']}"
    # Generate a folder name based on the values in each column
    folder_name = ""
    for column in columns:
        folder_name += f"_{column}_{row[column]}"
    folder_name = folder_name[1:]
    
    # Print the progress of loading the training history for a specific subject and folder
    print(f"Loading the training history of subject {row['subject_id']} and {folder_name}")

    path_to_load_files = os.path.join(path_to_load, ss, folder_name)
    files = [f for f in os.listdir(path_to_load_files) if f.endswith('.csv')]
    # Sort
    files = sorted(files)
    for file in files:
        file_output = pd.read_csv(os.path.join(path_to_load_files, file))
        subset = file.split('_')[0]

        y_pred = file_output['predicted_labels'].values
        y_true = file_output['true_labels'].values
        proba_class_1 = file_output['proba_class_1'].values
        if split_seed_flag:
            # Compute the metrics
            df_aux = pd.DataFrame({'subject_id': [row['subject_id']], 
                                'split_seed': [row['split_seed']],
                                'model_init_seed': [row['model_init_seed']],
                                'subset': [subset],
                                'accuracy': accuracy(y_pred, y_true),
                                'auc': roc_auc_score(y_true, proba_class_1) if len(np.unique(y_true)) > 1 else np.nan})
        else:
            # Compute the metrics
            df_aux = pd.DataFrame({'subject_id': [row['subject_id']], 
                                'model_init_seed': [row['model_init_seed']],
                                'subset': [subset],
                                'accuracy': accuracy(y_pred, y_true),
                                'auc': roc_auc_score(y_true, proba_class_1) if len(np.unique(y_true)) > 1 else np.nan})

        metrics = pd.concat([metrics, df_aux], ignore_index=True)
metrics.to_csv(os.path.join(path_to_save, 'metrics.csv'), index=False)