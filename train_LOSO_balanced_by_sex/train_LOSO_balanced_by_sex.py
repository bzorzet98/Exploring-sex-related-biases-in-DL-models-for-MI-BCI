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
import torch 
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import init_paths

from src.utils.json_utils import load_json_file, save_dict_as_json
from src.torch_utils import obtain_cuda_device
from src.utils.auxiliary_functions import convert_labels_to_int, build_skorch_kwargs, zip_files_in_directory
from src.utils.imports import import_class
from src.cross_validators.splitter import train_val_balanceBySex
from src.networks import *
from src.preprocessed_dataset import PreprocessedDataset
from src.skorch_modules.custom_checkpoints import SaveFirstEpoch, SaveLastEpoch, SaveEveryNEpochs
from src.skorch_modules.customdatasets import *

from skorch import NeuralNet
from skorch.callbacks import EpochScoring, Checkpoint

from skorch.dataset import ValidSplit
from sklearn.model_selection import PredefinedSplit

from sklearn.metrics import accuracy_score

from global_config import MI_EVENTS_DICT,DATABASES_PATH, DIRECTORY_TO_SAVE_ROOT

start_time = datetime.now()

def custom_accuracy_scoring(net, X, y_true):
    """ This function is used to compute the accuracy in the training set. How we use NeuralNet and not NeuralNetClassifier
    we have to reimplement the accuracy function to handdle the output of the predict_proba function, becaus predict from NeuralNet returns 
    the same that predict_proba."""
    y_pred = net.predict_proba(X)
    y_pred = np.argmax(y_pred, axis=1)
    return accuracy_score(y_true, y_pred)

# Create an ArgumentParser object
parser = argparse.ArgumentParser()
parser.add_argument('--script_config', type=str, default = 'EEGNetv4_Cho2017_paper_config')
parser.add_argument('--cuda', type=int, default = 0)
parser.add_argument('--save_scripts', type=bool, default = False)
args = parser.parse_args()
script_config_ = args.script_config
cuda = args.cuda
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
#dataset = import_class(dataset_name)(**dataset_args)
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
device = obtain_cuda_device(cuda)
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
    X_train, y_train, metadata_train = dataset.get_data(subjects = train_subjects + val_subjects,sessions = [session], return_as_dict = False) # Here we have to implement the method get_data with the corrects arguments
    y_train = convert_labels_to_int(y_train, dict_labels=MI_EVENTS_DICT)
    val_idx = metadata_train['subject'].isin(val_subjects).to_list()
    train_idx = metadata_train['subject'].isin(train_subjects).to_list()
    metadata_train, metadata_val  = metadata_train[train_idx], metadata_train[val_idx]
    val_fold = np.full(len(X_train), -1) 
    val_fold[val_idx]  = 0
    cv = ValidSplit(cv=PredefinedSplit(val_fold))

    ############################################## CALLBACKS ########################################################
    callbacks = [
        ('acc_train', EpochScoring(scoring=custom_accuracy_scoring, lower_is_better=False, on_train=True, name='train_accuracy',use_caching=True)), # This callback is used to compute the accuracy in the training set
        ('acc_val',EpochScoring(scoring=custom_accuracy_scoring,lower_is_better=False,on_train=False,name='val_accuracy',use_caching=True)),# This callback is used to compute the accuracy in the validation set
        ('chekpoint_first', SaveFirstEpoch( dirname=path_to_save_results,
                                            f_params='params_epoch_{last_epoch[epoch]}.pt',
                                            f_optimizer=None,
                                            f_history=None,
                                            f_criterion=None,
                                            fn_prefix = "",
                                            monitor = None)),#THis callback is used to save the model in the first epoch
        ('checkpoint_epochs',SaveEveryNEpochs(  every=10, 
                                                dirname=path_to_save_results,
                                                f_params='params_epoch_{last_epoch[epoch]}.pt',
                                                f_optimizer=None,
                                                f_history=None,
                                                f_criterion=None,
                                                fn_prefix = "",
                                                monitor = None)),#THis callback is used to save the model every 10 epochs
        ('chekpoint',Checkpoint( dirname=path_to_save_results,
                                monitor='valid_loss_best',
                                f_params='best_model.pt',
                                f_optimizer=None,   
                                f_history=None,
                                f_criterion=None,
                                fn_prefix = "",
                                load_best=True)),#THis callback is used to save the best model
        ('chekpoint_end',SaveLastEpoch(dirname=path_to_save_results,
                    f_params='params_epoch_{last_epoch[epoch]}.pt',
                    f_optimizer=None,
                    f_history='training_history.json',
                    f_criterion=None,
                    fn_prefix = "",
                    monitor = None)),#This callback is used to save the model in the last epoch
                    ]


    ############################################## DEFINE NEURALNET ########################################################
    # Import the classes for the skorch_module, model, criterion and optimizer
    dataset_class = import_class(script_config['torch_dataset']['class_name'],script_config['torch_dataset']['module_name'])
    model = import_class(script_config['model']['class_name'],script_config['model']['module_name'])
    criterion = import_class(script_config['criterion']['class_name'],script_config['criterion']['module_name'])
    optimizer = import_class(script_config['optimizer']['class_name'],script_config['optimizer']['module_name'])
    # Convert the model, criterion and optimizer parameters to the correct format for skorch
    skorch_kwargs = build_skorch_kwargs(model_params=script_config['model']['params'], 
                                        criterion_params=script_config['criterion']['params'], 
                                        optimizer_params=script_config['optimizer']['params'])
    
    """Here we decide to use NeuralNet instead NeuralNetClassifier because if we want to define our own custom_dataset
    to handle the cases where the networks need more than one input, we have to use NeuralNet instead NeuralNetClassifier because
    NeuralNetClassifier.fit(custom_dataset) doesn't work, but NeuralNet.fit(custom_dataset) works. """
    torch.manual_seed(model_init_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(model_init_seed)
        torch.cuda.manual_seed_all(model_init_seed) 

    net = NeuralNet(
        module = model,
        criterion=criterion,    
        optimizer=optimizer,
        max_epochs = script_config['training']['max_epochs'],
        device = device,
        batch_size = script_config['training']['batch_size'],
        train_split=cv,
        iterator_train__shuffle = True,
        verbose = True,
        callbacks = callbacks,
        warm_start=True,
        **skorch_kwargs
        )

    ############################################## DEFINE THE DATASET ########################################################
    # Change the type of the data, because skorch doesn't check that the model parameters and the input data have the same type, we need to be sure that the data is in the correct type
    X_train, y_train, X_test, y_test = X_train.astype(np.float32), y_train.astype(np.int64), X_test.astype(np.float32), y_test.astype(np.int64)

    # Define the dataset
    if script_config['torch_dataset']['params']:
        test_dataset = dataset_class(X_test, y_test,**script_config['torch_dataset']['params'])
        train_dataset = dataset_class(X_train, y_train, **script_config['torch_dataset']['params'])
    else:
        test_dataset = dataset_class(X_test, y_test)
        train_dataset = dataset_class(X_train, y_train)

    

    ############################################## TRAIN THE MODEL ########################################################
    # Set torch seed
    # torch.manual_seed(model_init_seed)
    net.initialize() # For initialize the model with the correct parameters
    # Fit the model
    net.fit(train_dataset)

    ############################################## COMPUTE THE OUTPUTS ########################################################
    # Compute the outputs of the network
    ypred_proba = net.predict_proba(train_dataset)
    ypred = np.argmax(ypred_proba, axis=1)
    ypred_train , ypred_val = ypred[train_idx], ypred[val_idx]
    ypred_proba_train, ypred_proba_val = ypred_proba[train_idx], ypred_proba[val_idx]

    acc_train = accuracy_score(y_train[train_idx], ypred_train)
    acc_val = accuracy_score(y_train[val_idx], ypred_val)

    ypred_proba_test = net.predict_proba(test_dataset)
    ypred_test = np.argmax(ypred_proba_test, axis=1)
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
                                "true_labels": y_train[train_idx], 
                                "predicted_labels": ypred_train,})
    val_outputs = pd.DataFrame({"index": metadata_val['index'].tolist(),
                                "true_labels": y_train[val_idx],
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