""" THis script trains models for Leave One Subject Out (LOSO) cross-validation. 
The training set and validation set are balanced by sex. 
The script trains the model for each subject in the dataset. We can define the number of subjects for validation.
Also, we can define if we want to trainning for the same subject in different training sets and validation sets, as well as
differents model initialization seeds of the model.
"""

# Import Libraries
import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.pipeline import Pipeline
import mne
mne.set_log_level('ERROR')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import init_paths

from src.utils.json_utils import load_json_file, save_dict_as_json
from src.utils.auxiliary_functions import convert_labels_to_int, zip_files_in_directory
from src.utils.imports import import_class
from src.cross_validators.splitter import train_val_balanceBySex
from src.preprocessed_dataset import PreprocessedDataset


from sklearn.metrics import accuracy_score

from global_config import MI_EVENTS_DICT,DATABASES_PATH, DIRECTORY_TO_SAVE_ROOT

start_time = datetime.now()

# Create an ArgumentParser object
parser = argparse.ArgumentParser()
parser.add_argument('--script_config', type=str, default = 'CSP+LDA_config_fairness_Cho2017')
parser.add_argument('--save_scripts', type=bool, default = False)

args = parser.parse_args()
script_config_ = args.script_config
save_scripts = args.save_scripts

# Current path
current_folder_path = os.path.dirname(os.path.abspath(__file__))

# Load configurations
script_config = load_json_file(os.path.join(current_folder_path, 'configs', f'{script_config_}.json'))
experiment_name = script_config["script_config"]['experiment_name']
model_to_train = script_config["script_config"]['model_to_train']
dataset_name = script_config["script_config"]['database_name']
session = script_config["script_config"]['database_session']
# Here check if timestamp is setted in the config file, if not, set the current timestamp
timestamp = script_config["script_config"]['timestamp_to_continue_trainning'] if script_config["script_config"]['timestamp_to_continue_trainning'] else datetime.now().strftime("%Y%m%d_%H%M%S")

# Path to save
path_to_save = os.path.join(DIRECTORY_TO_SAVE_ROOT, f'{experiment_name}', f'{model_to_train}',
                            f'{dataset_name}_{session}', f'{timestamp}')

# Import the dataset selected
dataset_args = script_config['databases']
dataset = PreprocessedDataset(db_name = dataset_name, config = dataset_args['config'], channels = dataset_args['channels'])
subjects_id = script_config['databases']['subjects'] if script_config['databases']['subjects'] else dataset.subject_list 
nsubjects = len(subjects_id)
sessions = dataset.sessions
if session not in sessions:
    raise ValueError(f'The session {session} is not in the sessions of the dataset {dataset_name}')

# Load metadata
metadata = pd.read_csv(os.path.join(DATABASES_PATH, 'databases_information', dataset_name, f'{dataset_name}_database_information.csv')) # This code must be updated after we complete the code of datasets

# Splits seeds, model init seeds and number of splits 
master_seed = script_config['training']['seed']
splits_seeds = script_config['training']['sets_seeds'] if script_config['training']['sets_seeds'] else [master_seed]
models_init_seed = script_config['training']['model_init_seed'] if script_config['training']['model_init_seed'] else [master_seed]
nsplits= len(splits_seeds)
nmodel_init_seeds = len(models_init_seed)

# Check if we have to continue the training
continue_train = os.path.exists(path_to_save)
os.makedirs(path_to_save, exist_ok=True)
if (continue_train) and (os.path.exists(os.path.join(path_to_save, 'script_progress.csv'))):
    script_progress = pd.read_csv(os.path.join(path_to_save, 'script_progress.csv'))
else:
    script_progress = pd.DataFrame({'subject_id':[ss for ss in subjects_id for _ in range(nsplits*nmodel_init_seeds)],
                                    'split_seed':[i for i in splits_seeds for _ in range(nmodel_init_seeds)]*nsubjects,   
                                    'model_init_seed': models_init_seed * (nsplits*nsubjects),
                                    'completed': [False] * (nsplits*nmodel_init_seeds*nsubjects)})
    script_progress.to_csv(os.path.join(path_to_save, 'script_progress.csv'), index = False)

# Calculate the number of subjects for training and validation
n_subjects = len(metadata)
n_males = len(metadata.query('sex == "M"'))
n_females = len(metadata.query('sex == "F"'))
n_subjects_val = script_config['training']['n_subjects_val']
min_set = min(n_males,n_females) -1  # We subtract 1 because we need at least one subject of the lower amount of subjects 
n_subjects_train = 2*min_set - n_subjects_val

# Obtain the device where send the data and model
device = "cpu"
script_config['training']['device'] = str(device)
# Iterate over the subjects
script_progress__ = script_progress.query('completed == False')

for subject_id, split_seed, model_init_seed in script_progress__[['subject_id', 'split_seed', 'model_init_seed']].values.tolist():
    print(f"Training subject {subject_id} of the dataset {dataset_name} in the session {session}, with split seed {split_seed} and model init seed {model_init_seed}")
    path_to_save_results = os.path.join(path_to_save, f'subject_{subject_id}', f'split_seed_{split_seed}_model_init_seed_{model_init_seed}')
    os.makedirs(path_to_save_results, exist_ok=True)

    # Load test data
    X_test, y_test, metadata_test = dataset.get_data(subjects = [subject_id], sessions=[session], return_as_dict = False) # Here we have to implement the method get_data with the corrects arguments
    y_test = convert_labels_to_int(y_test, dict_labels=MI_EVENTS_DICT)
    
    ############################ TRAINING AND VALIDATION SUBJECTS ########################################
    metadata_wtest = metadata[metadata['subject_id']!=subject_id] # metadata of the subjects that are not the test subject
    train_subjects, val_subjects, ignored_subjects = train_val_balanceBySex(metadata_wtest, n_subjects_train, n_subjects_val, seed = split_seed)

    X_train, y_train, metadata_train = dataset.get_data(subjects = train_subjects ,sessions = [session], return_as_dict = False) 
    y_train = convert_labels_to_int(y_train, dict_labels=MI_EVENTS_DICT)

    X_val, y_val, metadata_val = dataset.get_data(subjects = val_subjects ,sessions = [session], return_as_dict = False) 
    y_val = convert_labels_to_int(y_val, dict_labels=MI_EVENTS_DICT)

    ############################################## DEFINE THE FEATURE EXTRACTOR AND CLASSIFIER ########################################################
    
    # Define the feature extractor
    feature_extractor_class = import_class(script_config['feature_extractor']['class_name'],script_config['feature_extractor']['module_name'])
    classifier_class = import_class(script_config['classifier']['class_name'],script_config['classifier']['module_name'])
    np.random.seed(model_init_seed)

    # Initilize the pipeline
    CSP = feature_extractor_class(**script_config['feature_extractor']['params'])
    LDA = classifier_class(**script_config['classifier']['params'])

    # Create the pipeline
    pipeline = Pipeline([
        ('CSP', CSP),
        ('LDA', LDA)
    ])

    ############################################## TRAIN THE PIPELINE ########################################################
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    ############################################## SAVE THE WHOLE PIPELINE ####################################################
    
    joblib.dump(pipeline, os.path.join(path_to_save_results, 'pipeline_classifier.pkl'))

    print(f"The whole pipeline was saved in {os.path.join(path_to_save_results, 'pipeline_classifier.pkl')}")

    ############################################## COMPUTE THE OUTPUTS ########################################################
    # Compute the outputs of the classifier
    ypred_train = pipeline.predict(X_train)
    ypred_proba_train = pipeline.predict_proba(X_train)

    ypred_val = pipeline.predict(X_val)
    ypred_proba_val = pipeline.predict_proba(X_val)

    ypred_test = pipeline.predict(X_test)
    ypred_proba_test = pipeline.predict_proba(X_test)

    acc_train = accuracy_score(y_train, ypred_train)
    acc_val = accuracy_score(y_val, ypred_val)
    acc_test = accuracy_score(y_test, ypred_test)

    # Print the results
    print(f'Dataset: {dataset_name}. Session: {session}. Subject: {subject_id}')
    print(f'Train Accuracy: {acc_train}')
    print(f'Validation Accuracy: {acc_val}')
    print(f'Test Accuracy: {acc_test}')

    ############################################## SAVE THE OUTPUTS ########################################################
    info_data_training = {
        'dataset': dataset_name,
        'session': f"session_{session}",
        'test_subjects': [subject_id],
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'ignored_subjects': ignored_subjects,}
    
    info_metrics_training = {
        'train_accuracy': acc_train,
        'val_accuracy': acc_val,
        'test_accuracy': acc_test,
    }
    
    train_outputs = pd.DataFrame({"index": metadata_train['index'].tolist(), 
                                "true_labels": y_train, 
                                "predicted_labels": ypred_train,})
    val_outputs = pd.DataFrame({"index": metadata_val['index'].tolist(),
                                "true_labels": y_val,
                                "predicted_labels": ypred_val,})
    test_outputs = pd.DataFrame({"index": metadata_test['index'].tolist(),
                                "true_labels": y_test,
                                "predicted_labels": ypred_test,})
    
    for classes in range(ypred_proba_test.shape[1]):
        train_outputs[f'proba_class_{classes}'] = ypred_proba_train[:,classes]
        val_outputs[f'proba_class_{classes}'] = ypred_proba_val[:,classes]
        test_outputs[f'proba_class_{classes}'] = ypred_proba_test[:,classes]

    # Save the dictionary
    save_dict_as_json(path_to_save_results, info_data_training, file_name='info_data_training.json')
    save_dict_as_json(path_to_save_results, info_metrics_training, file_name='info_metrics_training.json')
    train_outputs.to_csv(os.path.join(path_to_save_results, 'train_outputs.csv'), index = False)
    val_outputs.to_csv(os.path.join(path_to_save_results, 'val_outputs.csv'), index = False)
    test_outputs.to_csv(os.path.join(path_to_save_results, 'test_outputs.csv'), index = False)
    # Change the script_progress based on subject_id, model_init_seed and set_seed
    script_progress.loc[(script_progress['subject_id'] == subject_id) & (script_progress['split_seed'] == split_seed) & (script_progress['model_init_seed'] == model_init_seed), 'completed'] = True
    script_progress.to_csv(os.path.join(path_to_save, 'script_progress.csv'), index = False)

#Add the name of the script
script_config['name_script'] = os.path.basename(__file__).split('.')[0]
# SAVE THE SCRIPT CONFIG
save_dict_as_json(path_to_save, script_config, file_name='config.json')
if save_scripts:
    zip_files_in_directory(source_directory=os.path.dirname(current_folder_path), output_directory=path_to_save, prefix="script_files")

end_time = datetime.now()  
print(f"Total time: {end_time - start_time}")