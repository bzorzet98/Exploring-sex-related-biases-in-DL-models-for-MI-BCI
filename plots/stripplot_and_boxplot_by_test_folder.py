# Import Libraries
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import init_paths
from global_config import DIRECTORY_TO_SAVE_ROOT

def plot_boxplot(data,y,x = None, ax=None, hue=None, palette="viridis", custom_palette=None, order=None,
                   xlabel=None, ylabel=None,  title=None, ylim = (0.1,1.), 
                   set_hline = True, rotation = 'horizontal', 
                   dodge=False,saturation=1.0,fill=True):
    """
    Creates a boxplot of accuracy scores.

    Parameters:
        - data: DataFrame containing the data
        - x: Column name for the x-axis (subject)
        - y: Column name for the y-axis (accuracy_score_mean)
        - ax: Matplotlib Axes object (optional)
        - hue: Column name for coloring the points (optional)
        - palette: Color palette for the plot (default is "viridis")
        - custom_palette: Custom palette for coloring based on hue (optional)
        - order: Order of plotting (optional)

    Returns:
        - ax: Matplotlib Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))

    if hue:
        if custom_palette:
            sns.boxplot(x=x, y=y, hue=hue, data=data, palette=custom_palette, dodge=dodge, ax=ax, order=order, saturation=saturation)
        else:
            sns.boxplot(x=x, y=y, hue=hue, data=data, palette=palette, dodge=dodge, ax=ax, order=order, saturation=saturation, fill=fill)
        ax.legend(title=hue, bbox_to_anchor=(1, 1), loc='upper left')
    else:
        sns.boxplot(x=x, y=y, data=data, dodge=dodge, palette=palette, ax=ax, order=order, saturation=saturation, fill=fill)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    # Set ylim to 0-1.1
    if ylim:
        ax.set_ylim(ylim)
    # Set hline to 0.5
    if set_hline:
        ax.axhline(0.5, color='black', linestyle='--', linewidth = 1.)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    
    return ax

def plot_stripplot(data,  y,x= None, ax=None, hue=None, palette="viridis", custom_palette=None, order=None,
                   xlabel=None, ylabel=None, alpha=0.5, title=None, jitter=0.1,  ylim = (0.1,1.), set_hline = True,
                   rotation = 'horizontal', dodge=False):
    """
    Creates a stripplot of accuracy scores.

    Parameters:
        - data: DataFrame containing the data
        - x: Column name for the x-axis (subject)
        - y: Column name for the y-axis (accuracy_score_mean)
        - ax: Matplotlib Axes object (optional)
        - hue: Column name for coloring the points (optional)
        - palette: Color palette for the plot (default is "viridis")
        - custom_palette: Custom palette for coloring based on hue (optional)
        - order: Order of plotting (optional)

    Returns:
        - ax: Matplotlib Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))

    if hue:
        if custom_palette:
            sns.stripplot(x=x, y=y, hue=hue, data=data, palette=custom_palette, dodge=dodge, jitter=jitter, ax=ax, order=order, alpha=alpha)
        else:
            sns.stripplot(x=x, y=y, hue=hue, data=data, palette=palette, dodge=dodge, jitter=jitter, ax=ax, order=order, alpha=alpha)
        ax.legend(title=hue, bbox_to_anchor=(1, 1), loc='upper left')
    else:
        sns.stripplot(x=x, y=y, data=data, dodge=dodge, jitter=jitter, ax=ax, order=order, alpha=alpha)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    # Set ylim to 0-1.1
    if ylim:
        ax.set_ylim(ylim)
    # Set hline to 0.5
    if set_hline:
        ax.axhline(0.5, color='black', linestyle='--', linewidth = 1.)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    
    return ax

from global_config import DIRECTORY_TO_SAVE_ROOT, DATABASES_PATH

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
path_to_load = os.path.join(DIRECTORY_TO_SAVE_ROOT, f'{experiment_name}', f'{model_to_train}', f'{dataset_name}_{session}', f'{timestamp}','different_analysis', 'compute_metrics_from_outputs')
path_to_save = os.path.join(DIRECTORY_TO_SAVE_ROOT, f'{experiment_name}', f'{model_to_train}', f'{dataset_name}_{session}', f'{timestamp}', 'plots', 'stripplot_and_boxplot_by_test_folder')
os.makedirs(path_to_save, exist_ok=True)

# Load the metrics.csv file
metrics = pd.read_csv(os.path.join(path_to_load, 'metrics.csv'))
metrics_names = ['accuracy', 'auc']
subsets = metrics['subset'].unique()

custom_colors = sns.color_palette('Set1')
custom_palette = custom_colors
# Check if metadata is available
metadata_available = False
if os.path.exists(os.path.join(DATABASES_PATH, 'databases_information', dataset_name, f'{dataset_name}_database_information.csv')):
    metadata = pd.read_csv(os.path.join(DATABASES_PATH, 'databases_information', dataset_name, f'{dataset_name}_database_information.csv'))
    
    metrics = pd.merge(metrics, metadata[['subject_id','sex','age']], left_on='subject_id', right_on='subject_id', how='left')

    custom_colors[0], custom_colors[1] = custom_colors[1], custom_colors[0]  # Swap the colors for 'Males' and 'Females'
    custom_palette = {'M': custom_colors[0], 'F': custom_colors[1]}

    metadata_available = True
    
for metric_name in metrics_names:
    for subset in subsets:
        # Check if the column metrics_name is nan in the subset
        if metrics[metrics['subset'] == subset][metric_name].isnull().all():
            continue
        else:
            df_aux = metrics[(metrics['subset'] == subset)]
            title = f'Boxplot and stripplot of {metric_name} in {subset}'
            
            # Keep only the females subject_id
            order = df_aux[df_aux['sex'] == 'F']['subject_id'].values.tolist() +  df_aux[df_aux['sex'] == 'M']['subject_id'].values.tolist()

            fig, ax = plt.subplots(figsize=(20, 8))
            
            if metadata_available:
                ax = plot_boxplot(df_aux, x="subject_id", y=metric_name, 
                                    hue="sex", custom_palette=custom_palette, xlabel=None, ylabel=None, 
                                    title=title,order=order, ax = ax, ylim = (0.3,1.1))
                    
                ax = plot_stripplot(df_aux, x="subject_id", y=metric_name, 
                                    hue="sex", custom_palette=custom_palette, xlabel=None, ylabel=None, 
                                    alpha= 0.1, jitter=0.1 ,title=title,order=order, ax = ax,ylim =  (0.3,1.1))
                
            # Save the figure
            fig.savefig(os.path.join(path_to_save, f'{metric_name}_{subset}.png'), dpi = 300)
            plt.close(fig)