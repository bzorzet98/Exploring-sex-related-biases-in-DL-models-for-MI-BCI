# Import Libraries
import os
import argparse
import pandas as pd
import numpy as np

from src.utils.json_utils import load_json_file
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import init_paths
from global_config import DIRECTORY_TO_SAVE_ROOT

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

# Path to load and save the results
path_to_load = os.path.join(DIRECTORY_TO_SAVE_ROOT, f'{experiment_name}', f'{model_to_train}', f'{dataset_name}_{session}', f'{timestamp}')
path_to_save = os.path.join(DIRECTORY_TO_SAVE_ROOT, f'{experiment_name}', f'{model_to_train}', f'{dataset_name}_{session}', f'{timestamp}', 'different_analysis', 'compute_duration_and_best_epochs')
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

# Initialize variables
actual_subject = script_progress['subject_id'].values[0]
data_actual_subject = {}
i=0

metrics = pd.DataFrame(columns= ['subject_id'] + columns + ['duration', 'best_epoch'])

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

        # Print the progress of loading the training history for a specific subject and folder
    print(f"Loading the training history of subject {row['subject_id']} and {folder_name}")
    
    # Set the paths to load the training history and save the plots
    path_to_load_training_history = os.path.join(path_to_load, ss, folder_name)
    path_to_save_plots = os.path.join(path_to_save, ss)

    ########################################## Load information of the training ##########################################
    # Load the training history from a JSON file
    training_history = load_json_file(os.path.join(path_to_load_training_history, 'training_history.json'))
    
    # Extract the metrics from the training history
    epochs = [epoch for epoch in range(0,len(training_history))]
    duration = [epoch['dur'] for epoch in training_history]
    complete_duration = np.sum(duration)
    best_epoch = epochs[-1]
    if 'valid_loss' in training_history[0]:
        best_val_epochs = [epoch['valid_loss_best'] for epoch in training_history]
        # Obtain only the index of the true values
        best_epoch = np.argwhere(best_val_epochs).flatten()[-1]

    # Compute the metrics
    df_aux = pd.DataFrame({'subject_id': [row['subject_id']], 
                            'split_seed': [row['split_seed']],
                            'model_init_seed': [row['model_init_seed']],
                            'duration': [complete_duration],
                            'best_epoch': [best_epoch]})
    metrics = pd.concat([metrics, df_aux], ignore_index=True)

metrics.to_csv(os.path.join(path_to_save, 'durations_and_best_epochs.csv'), index=False)