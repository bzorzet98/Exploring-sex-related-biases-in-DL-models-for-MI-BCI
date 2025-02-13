# Import Libraries
import argparse
import os
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import init_paths

from src.utils.json_utils import load_json_file
from src.preprocessed_dataset import PreprocessedDataset
from src.utils.auxiliary_functions import convert_labels_to_int

from global_config import MI_EVENTS_DICT, DIRECTORY_TO_SAVE_ROOT

from pyriemann.estimation import Covariances
from pyriemann.classification import class_distinctiveness

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Load database for train the model.')

# Create an ArgumentParser object
parser = argparse.ArgumentParser()
parser.add_argument('--script_config', type=str, default = 'distinctiveness_Dreyer2023_A_without_preprocessing_segregated_by_runs')

args = parser.parse_args()
script_config_ = args.script_config

# Current path
current_folder_path = os.path.dirname(os.path.abspath(__file__))

# Load configurations
script_config = load_json_file(os.path.join(current_folder_path, 'configs', f'{script_config_}.json'))
experiment_name = script_config["script_config"]['experiment_name']
dataset_name = script_config["script_config"]['database_name']
session = script_config["script_config"]['database_session']
segregated_by_runs = script_config["script_config"]['segregate_by_runs'] if "segregate_by_runs" in script_config["script_config"].keys() else False

# Path to save
script_name = os.path.basename(__file__)[:-3]
path_to_save = os.path.join(DIRECTORY_TO_SAVE_ROOT, f'{experiment_name}', script_name)
os.makedirs(path_to_save, exist_ok = True)  
# Import the dataset selected
dataset_args = script_config['databases']
#dataset = import_class(dataset_name)(**dataset_args)
dataset = PreprocessedDataset(db_name = dataset_name, config = dataset_args['config'], channels = dataset_args['channels'])
subjects_id = script_config['databases']['subjects'] if script_config['databases']['subjects'] else dataset.subject_list 
nsubjects = len(subjects_id)
sessions = dataset.sessions
if session not in sessions:
    raise ValueError(f'The session {session} is not in the sessions of the dataset {dataset_name}')

# Create a Dataframe to save the results
df = pd.DataFrame(columns=['subject_id', 'class_distinctiveness', 'run'])

for subject_id  in subjects_id:
    print(f"Compute the distinctiveness coefficient for the subject {subject_id} of the dataset {dataset_name} in the session {session}")
    if segregated_by_runs:
        data = dataset.get_data(subjects = [subject_id], sessions = [session], return_as_dict = True) # Here we have to implement the method get_data with the corrects arguments
        subject_key = list(data.keys())[0]
        session_key = list(data[subject_key].keys())[0]
        for key, value in data[subject_key][session_key].items():
            X, y = value['data_eeg'], value['labels_eeg']
            y = convert_labels_to_int(y, dict_labels=MI_EVENTS_DICT)
            covariance_matrix = Covariances(estimator="lwf").transform(X)
            df = pd.concat([df, pd.DataFrame({'subject_id':[subject_id], 'run':[key], 'class_distinctiveness':[class_distinctiveness(covariance_matrix, y, exponent=2)]})], axis=0)
    else:
        X, y, _ = dataset.get_data(subjects = [subject_id], sessions = [session], return_as_dict = False) # Here we have to implement the method get_data with the corrects arguments
        y = convert_labels_to_int(y, dict_labels=MI_EVENTS_DICT)
        
        covariance_matrix = Covariances(estimator="lwf").transform(X)    
        df = pd.concat([df, pd.DataFrame({'subject_id':[subject_id], 'run':["whole_data"], 'class_distinctiveness':[class_distinctiveness(covariance_matrix, y, exponent=2)]})], axis=0)
    
# Save the metadata with the distinctiveness coefficient
if segregated_by_runs:
    df.to_csv(os.path.join(path_to_save, f'{dataset_name}_{session}_class_distinctiveness_by_run.csv'), index = False)
else:
    df.to_csv(os.path.join(path_to_save, f'{dataset_name}_{session}_class_distinctiveness.csv'), index = False)