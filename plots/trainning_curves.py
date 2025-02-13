""" This script is used to plot the training and validation curves of the training process of the models.
"""

# Import Libraries
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import init_paths
from src.utils.json_utils import load_json_file

from global_config import DIRECTORY_TO_SAVE_ROOT

def plot_train_val_metric(train_metric,  title, metric, path_to_save_metric = None, val_metric=None, ax = None, train_color='C0',val_color = 'C1'):
    """
    Plots the training and validation metric over epochs.

    Parameters:
    train_metric (list): List of training metric values.
    title (str): Title of the plot.
    metric (str): Name of the metric being plotted.
    path_to_save_metric (str, optional): Path to save the plot as an image file. Defaults to None.
    val_metric (list, optional): List of validation metric values. Defaults to None.
    ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, a new figure and axes will be created. Defaults to None.
    train_color (str, optional): Color of the training metric line. Defaults to 'C0'.
    val_color (str, optional): Color of the validation metric line. Defaults to 'C1'.

    Returns:
    matplotlib.axes.Axes: The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(train_metric, label='Train', color= train_color)
    if val_metric:
        ax.plot(val_metric, label='Val', color = val_color)
    ax.set_title(f'{title}_{metric}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric)
    if metric == 'loss':
        ymax = max(max(train_metric), max(val_metric)) if val_metric else max(train_metric)
    else:
        ymax=1
    ax.set_ylim(0, ymax)
    ax.legend()
    ax.grid(True)
    if path_to_save_metric:
        if not os.path.exists(os.path.dirname(path_to_save_metric)):
            os.makedirs(os.path.dirname(path_to_save_metric))
        plt.savefig(path_to_save_metric, dpi=300)
    return ax

def plot_train_val_metric_mean_std(train_metric_mean,train_metric_std, title, metric, path_to_save_metric = None, val_metric_mean=None, val_metric_std=None,
                                    train_color='C0',val_color = 'C1'):
    """
    Plots the training and validation metrics with mean and standard deviation.

    Parameters:
    train_metric_mean (array-like): Array of mean values for the training metric.
    train_metric_std (array-like): Array of standard deviation values for the training metric.
    title (str): Title of the plot.
    metric (str): Name of the metric.
    path_to_save_metric (str, optional): Path to save the plot. Defaults to None.
    val_metric_mean (array-like, optional): Array of mean values for the validation metric. Defaults to None.
    val_metric_std (array-like, optional): Array of standard deviation values for the validation metric. Defaults to None.
    train_color (str, optional): Color for the training metric plot. Defaults to 'C0'.
    val_color (str, optional): Color for the validation metric plot. Defaults to 'C1'.
    """
                                   
    epoch_array = np.arange(0, len(train_metric_mean))
    plt.figure(figsize=(10, 6))
    plt.plot(train_metric_mean, label=f'train {metric} (mean)', color= train_color)
    plt.fill_between(epoch_array, train_metric_mean - train_metric_std, train_metric_mean + train_metric_std, alpha=0.3, color = train_color)
    if val_metric_mean.all() != None:
        plt.plot(val_metric_mean, label=f'validation {metric} (mean)', color = val_color)
        plt.fill_between(epoch_array, val_metric_mean - val_metric_std, val_metric_mean + val_metric_std, alpha=0.3, color = val_color)
    plt.title(f'{title}:{metric}') 
    plt.xlabel('epochs')
    plt.ylabel(metric)
    if 'loss' in metric:
        ymax = max(max(train_metric_mean + train_metric_std), max(val_metric_mean + val_metric_std)) if val_metric_mean.all()!= None else max(train_metric_mean + train_metric_std)
    else:
        ymax=1
    plt.ylim(0, ymax)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  
    if path_to_save_metric:
        if not os.path.exists(os.path.dirname(path_to_save_metric)):
            os.makedirs(os.path.dirname(path_to_save_metric))
        plt.savefig(path_to_save_metric, dpi=300)
    plt.close()


parser = argparse.ArgumentParser(description='Plot the training and validation curves of the training process of the models.')
parser.add_argument('--experiment_name', type=str, default = 'BCI_GNN')
parser.add_argument('--model_to_train', type=str, default = 'SimpleGraphNetwork')
parser.add_argument('--dataset_name', type=str,default = 'Cho2017')
parser.add_argument('--session', type=int, default = 1)
parser.add_argument('--timestamp', type=str, default = '20241206_173905')
args = parser.parse_args()

# Charge the configuration of the script
experiment_name = args.experiment_name
model_to_train = args.model_to_train
dataset_name = args.dataset_name
session = args.session
timestamp = args.timestamp

# Path to load and save the results
path_to_load = os.path.join(DIRECTORY_TO_SAVE_ROOT, f'{experiment_name}', f'{model_to_train}', f'{dataset_name}_{session}', f'{timestamp}')
path_to_save = os.path.join(DIRECTORY_TO_SAVE_ROOT, f'{experiment_name}', f'{model_to_train}', f'{dataset_name}_{session}', f'{timestamp}', 'plots', 'training_curves')
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
fig_all_loss_plots, ax_all_loss_plots = plt.subplots(1, 1, figsize=(10, 6))
fig_all_acc_plots, ax_all_acc_plots = plt.subplots(1, 1, figsize=(10, 6))



equal_len_flag = True    # Generate a folder name based on the values in each column
folder_name = ""
for column in columns:
    folder_name += f"_{column}_{script_progress[column].values[0]}"
folder_name = folder_name[1:]
path_to_load_training_history = os.path.join(path_to_load, f"subject_{actual_subject}", folder_name)
training_history = load_json_file(os.path.join(path_to_load_training_history, 'training_history.json'))
last_len = len([epoch['train_loss'] for epoch in training_history])

# Iterate over each row in the script progress dataframe
for index, row in script_progress.iterrows():

    # Check if the subject has changed
    if actual_subject != row['subject_id']:
        # Save the plots of every single curve training
        # eliminate the repeated legends
        plt.close()
        handles, labels = ax_all_loss_plots.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_all_loss_plots.legend(by_label.values(), by_label.keys())
        handles, labels = ax_all_acc_plots.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_all_acc_plots.legend(by_label.values(), by_label.keys())
        # Set y lim (0,1)
        ax_all_loss_plots.set_ylim(0,1)
        ax_all_acc_plots.set_ylim(0,1)

        ax_all_loss_plots.title.set_text(f'subject_{actual_subject} loss')
        ax_all_acc_plots.title.set_text(f'subject_{actual_subject} accuracy')

        fig_all_loss_plots.savefig(os.path.join(path_to_save_plots,f'subject_{actual_subject}_all_loss_plots.png'), dpi=300)
        fig_all_acc_plots.savefig(os.path.join(path_to_save_plots,f'subject_{actual_subject}_all_acc_plots.png'), dpi=300)
        
        # Check that
        if data_actual_subject and equal_len_flag:
            # Calculate mean and standard deviation for training loss and accuracy
            train_losses = np.array([data_actual_subject[it]['train_loss'] for it in data_actual_subject])
            train_loss_mean = np.mean(train_losses, axis=0)
            train_loss_std = np.std(train_losses, axis=0)
            
            train_accuracy = np.array([data_actual_subject[it]['train_accuracy'] for it in data_actual_subject])
            train_acc_mean = np.mean(train_accuracy, axis=0)
            train_acc_std = np.std(train_accuracy, axis=0)

            # Calculate mean and standard deviation for validation loss and accuracy
            val_loss_mean = None
            val_loss_std = None
            val_acc_mean = None
            val_acc_std = None

            if data_actual_subject[0]['valid_loss']:
                val_losses = np.array([data_actual_subject[it]['valid_loss'] for it in data_actual_subject])
                val_loss_mean = np.mean(val_losses, axis=0)
                val_loss_std = np.std(val_losses, axis=0)
            if data_actual_subject[0]['valid_accuracy']:
                val_acc = np.array([data_actual_subject[it]['valid_accuracy'] for it in data_actual_subject])
                val_acc_mean = np.mean(val_acc, axis=0)
                val_acc_std = np.std(val_acc, axis=0)

            # Plot mean and standard deviation for training and validation curves
            title = f'training curves {actual_subject}'
            plot_train_val_metric_mean_std(train_loss_mean,train_loss_std, title = title, metric = 'loss', 
                                            path_to_save_metric = os.path.join(path_to_save_plots,f'subject_{actual_subject}_loss_mean.png'), 
                                            val_metric_mean=val_loss_mean, val_metric_std=val_loss_std)
            plot_train_val_metric_mean_std(train_acc_mean,train_acc_std, title = title, metric = 'accuracy', 
                                            path_to_save_metric = os.path.join(path_to_save_plots,f'subject_{actual_subject}_accuracy_mean.png'), 
                                            val_metric_mean=val_acc_mean, val_metric_std=val_acc_std)

            plt.close()
        # Reset the variables for the next subject
        actual_subject = row['subject_id']
        data_actual_subject = {}
        i = 0
        fig_all_loss_plots, ax_all_loss_plots = plt.subplots(1, 1, figsize=(10, 6))
        fig_all_acc_plots, ax_all_acc_plots = plt.subplots(1, 1, figsize=(10, 6))
        

        equal_len_flag = True
        folder_name = ""
        for column in columns:
            folder_name += f"_{column}_{row[column]}"
        folder_name = folder_name[1:]
        path_to_load_training_history = os.path.join(path_to_load, ss, folder_name)
        training_history = load_json_file(os.path.join(path_to_load_training_history, 'training_history.json'))
        last_len = len([epoch['train_loss'] for epoch in training_history])

    # Generate the subject name
    ss = f"subject_{row['subject_id']}"
    # Generate a folder name based on the values in each column
    folder_name = ""
    for column in columns:
        folder_name += f"_{column}_{row[column]}"
    folder_name = folder_name[1:]
    
    # Print the progress of loading the training history for a specific subject and folder
    print(f"Loading the training history of subject {row['subject_id']} and {folder_name}")
    
    # Set the paths to load the training history and save the plots
    path_to_load_training_history = os.path.join(path_to_load, ss, folder_name)
    path_to_save_plots = os.path.join(path_to_save, ss)

    ########################################## Load information of the training ##########################################
    # Load the training history from a JSON file
    training_history = load_json_file(os.path.join(path_to_load_training_history, 'training_history.json'))
    
    # Extract the metrics from the training history
    train_loss = [epoch['train_loss'] for epoch in training_history]
    train_accuracy = [epoch['train_accuracy'] for epoch in training_history]
    valid_loss = None
    valid_accuracy = None
    if 'valid_loss' in training_history[0]:
        valid_loss = [epoch['valid_loss'] for epoch in training_history]
    if 'val_accuracy' in training_history[0]:
        valid_accuracy = [epoch['val_accuracy'] for epoch in training_history]

    # Generate a single plot of the training and validation curves 
    title = f"subject_{row['subject_id']}_{folder_name}"
    _ = plot_train_val_metric(  train_loss,  title, metric = 'loss', 
                                path_to_save_metric = os.path.join(path_to_save_plots,'loss',f'{title}_loss.png'), 
                                val_metric=valid_loss)
    _ = plot_train_val_metric(train_accuracy,  title, metric = 'accuracy', 
                            path_to_save_metric = os.path.join(path_to_save_plots,'accuracy',f'{title}_accuracy.png'), 
                            val_metric=valid_accuracy)
    
    # Plot the mean of the training and validation of the subject in the same figure
    ax_all_loss_plots = plot_train_val_metric(train_loss,  title, metric = 'loss', 
                                        path_to_save_metric = None, 
                                        val_metric=valid_loss, ax = ax_all_loss_plots)
    
    ax_all_acc_plots = plot_train_val_metric(train_accuracy,  title, metric = 'accuracy', 
                                        path_to_save_metric = None, 
                                        val_metric=valid_accuracy, ax = ax_all_acc_plots)
    
    plt.close()
    # Store the training and validation metrics for the subject
    data_actual_subject.update({i:{'train_loss': train_loss, 
                                'train_accuracy': train_accuracy, 
                                'valid_loss': valid_loss, 
                                'valid_accuracy': valid_accuracy}})
    
    # Check that the length of the training loss is the same inside test subject
    equal_len_flag = True if (last_len == len(train_loss)) & (equal_len_flag) else False
    last_len = len(train_loss) 

    i+=1

# Save the plots of the last subjects curve training
# eliminate the repeated legends
handles, labels = ax_all_loss_plots.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax_all_loss_plots.legend(by_label.values(), by_label.keys())
handles, labels = ax_all_acc_plots.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax_all_acc_plots.legend(by_label.values(), by_label.keys())
fig_all_loss_plots.savefig(os.path.join(path_to_save_plots,f'subject_{actual_subject}_all_loss_plots.png'), dpi=300)
fig_all_acc_plots.savefig(os.path.join(path_to_save_plots,f'subject_{actual_subject}_all_acc_plots.png'), dpi=300)
plt.close()

# Check if data_actual_subject is not empty
if data_actual_subject and equal_len_flag:
    # Calculate mean and standard deviation for training loss and accuracy
    train_losses = np.array([data_actual_subject[i]['train_loss'] for i in data_actual_subject])
    train_loss_mean = np.mean(train_losses, axis=0)
    train_loss_std = np.std(train_losses, axis=0)
    
    train_accuracy = np.array([data_actual_subject[i]['train_accuracy'] for i in data_actual_subject])
    train_acc_mean = np.mean(train_accuracy, axis=0)
    train_acc_std = np.std(train_accuracy, axis=0)

    # Calculate mean and standard deviation for validation loss and accuracy
    val_loss_mean = None
    val_loss_std = None
    val_acc_mean = None
    val_acc_std = None

    if data_actual_subject[0]['valid_loss']:
        val_losses = np.array([data_actual_subject[i]['valid_loss'] for i in data_actual_subject])
        val_loss_mean = np.mean(val_losses, axis=0)
        val_loss_std = np.std(val_losses, axis=0)
    if data_actual_subject[0]['valid_accuracy']:
        val_acc = np.array([data_actual_subject[i]['valid_accuracy'] for i in data_actual_subject])
        val_acc_mean = np.mean(val_acc, axis=0)
        val_acc_std = np.std(val_acc, axis=0)

    # Plot mean and standard deviation for training and validation curves
    title = f'training curves {actual_subject}'
    plot_train_val_metric_mean_std(train_loss_mean,train_loss_std, title = title, metric = 'loss', 
                                    path_to_save_metric = os.path.join(path_to_save_plots,f'subject_{actual_subject}_loss_mean.png'), 
                                    val_metric_mean=val_loss_mean, val_metric_std=val_loss_std)
    plot_train_val_metric_mean_std(train_acc_mean,train_acc_std, title = title, metric = 'accuracy', 
                                    path_to_save_metric = os.path.join(path_to_save_plots,f'subject_{actual_subject}_accuracy_mean.png'), 
                                    val_metric_mean=val_acc_mean, val_metric_std=val_acc_std)

plt.close()