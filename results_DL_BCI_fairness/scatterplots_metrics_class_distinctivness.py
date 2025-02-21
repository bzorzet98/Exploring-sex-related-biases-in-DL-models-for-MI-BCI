# Import Libraries
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import init_paths
from global_config import DIRECTORY_TO_SAVE_ROOT, DATABASES_PATH
import numpy as np
from scipy import stats

# Charge the configuration of the script
experiment_name = 'DL_BCI_fairness'
datasets_name = ['Cho2017', 'Lee2019_MI', 'Lee2019_MI']
sessions = [1,1,2]
model_1 = 'EEGNetv4_SM'
timestamps_1 = ['20241004_144153', '20241105_185644', '20241113_125922' ] # Original timestamps
# timestamps_1 = ['X', 'X', 'X'] # New timestamps , if you want to use new timestamps, replace 'X' with the new timestamp and uncomment this line
model_2 = 'CSP+LDA'
timestamps_2 = ['20250218_162518', '20250218_172951', '20250218_181318'] # Original timestamps
# timestamps_2 = ['X', 'X', 'X'] # New timestamps , if you want to use new timestamps, replace 'X' with the new timestamp and uncomment this line
row_titles = ['Cho 2017', 'Lee 2019 session 1', 'Lee 2019 session 2']
# Name of script
script_name = os.path.basename(__file__).split('.')[0]
path_to_save = os.path.join(DIRECTORY_TO_SAVE_ROOT, experiment_name, 'results_DL_BCI_fairness', script_name)
os.makedirs(path_to_save, exist_ok=True)
# Obtain the metrics names, are the columns that are not subjects_id
sufixes = ['_DL', '_CSPLDA']
metrics_names = ['accuracy', 'auc']

# Custom color palette
custom_colors = sns.color_palette('Set1')
custom_colors[0], custom_colors[1] = custom_colors[1], custom_colors[0]  # Swap the colors for 'Males' and 'Females'
custom_palette = {'M': custom_colors[0], 'F': custom_colors[1]}

subset = 'test'

plt.rcParams.update({
    "font.family": "Times New Roman",  # Set font to Times New Roman
    "axes.titlesize": 9,             # Title size
    "axes.labelsize": 9,              # Labels size
    "xtick.labelsize": 9,             # X-axis tick size
    "ytick.labelsize": 10,             # Y-axis tick size
    "legend.fontsize": 9              # Legend size
})

marker_size = 6
line_width = 1.5
title_pad =  5
figsize = (10,7)


for metric_name in metrics_names:
    row_axes = 0
    if metric_name == "auc":
        figsize = (7,5)
        plt.rcParams.update({
            "font.family": "Times New Roman",  # Set font to Times New Roman
            "axes.titlesize": 9,             # Title size
            "axes.labelsize": 9,              # Labels size
            "xtick.labelsize": 9,             # X-axis tick size
            "ytick.labelsize": 10,             # Y-axis tick size
            "legend.fontsize": 9              # Legend size
        })

    # Create the figure first figure
    fig, axes = plt.subplots(3, 3, figsize=figsize)  
    fig_log, axes_log = plt.subplots(3, 3, figsize=figsize)  
    plt.subplots_adjust(left=0.1, right=0.5, top=0.5, bottom=0.1)

    column_titles = [f'Deep Learning Model', 'CSP + LDA', 'Difference']
    row_titles = ['Cho 2017', 'Lee 2019 session 1', 'Lee 2019 session 2']

    for dataset_name,row_title, session, timestamp_1, timestamp_2 in zip(datasets_name, row_titles, sessions, timestamps_1, timestamps_2):
        
        # Path to load and save the results
        path_to_load_1 = os.path.join(DIRECTORY_TO_SAVE_ROOT, f'{experiment_name}', f'{model_1}', f'{dataset_name}_{session}', f'{timestamp_1}','different_analysis', 'compute_metrics_from_outputs')
        path_to_load_2 = os.path.join(DIRECTORY_TO_SAVE_ROOT, f'{experiment_name}', f'{model_2}', f'{dataset_name}_{session}', f'{timestamp_2}','different_analysis', 'compute_metrics_from_outputs')
        
        # Load the metrics.csv file
        metrics_DL = pd.read_csv(os.path.join(path_to_load_1, 'metrics.csv'))
        metrics_CSPLDA = pd.read_csv(os.path.join(path_to_load_2, 'metrics.csv'))

        # Merge the dataframes to compare the metrics
        metrics = pd.merge(metrics_DL, metrics_CSPLDA, on=['subject_id','split_seed', 'subset'], suffixes=('_DL', '_CSPLDA'))
        # Read the metadata
        class_distinctiveness = pd.read_csv(os.path.join(DIRECTORY_TO_SAVE_ROOT, f'{experiment_name}', 'distinctiveness_coefficent_per_subject', f'{dataset_name}_{session}_class_distinctiveness.csv'))
        metadata =  pd.read_csv(os.path.join(DATABASES_PATH, 'databases_information', dataset_name, f'{dataset_name}_database_information.csv'))
        metadata = pd.merge(metadata, class_distinctiveness, left_on='subject_id', right_on='subject_id', how='left')
        metrics = pd.merge(metrics, metadata[['subject_id','sex','age','class_distinctiveness']], left_on='subject_id', right_on='subject_id', how='left')

        ########################################### Difference between models ########################################################
        metrics[f'diff_{metric_name}'] = metrics[f'{metric_name}_DL'] - metrics[f'{metric_name}_CSPLDA']
        
        column_it = 0
        if metric_name == 'accuracy':
            ylabels = ['deep learning accuracy', 'CSP+LDA accuracy', 'Difference accuracy']
        else:
            ylabels = ['deep learning AUC', 'CSP+LDA AUC', 'Difference AUC']

        columns_list = [f'{metric_name}_DL', f'{metric_name}_CSPLDA', f'diff_{metric_name}']
        for column_name in columns_list:

            df_metrics_by_subject_id_aux = metrics.groupby(['subset', 'subject_id', 'sex', 'age', 'class_distinctiveness'])[column_name].agg(["mean","std"]).reset_index()
            df_metrics_by_subject_id_aux = df_metrics_by_subject_id_aux[df_metrics_by_subject_id_aux['subset']==subset].sort_values('subject_id')
            df_metrics_by_subject_id_aux = df_metrics_by_subject_id_aux[df_metrics_by_subject_id_aux['subset'] == subset]

            # Compute the logritm of the class_distinctiveness
            df_metrics_by_subject_id_aux['log_class_distinctiveness'] = np.log(df_metrics_by_subject_id_aux['class_distinctiveness'])

            # Plot the regression plot original scale
            axes[row_axes, column_it] = sns.regplot(x='class_distinctiveness', y='mean', data=df_metrics_by_subject_id_aux,
                                                        label=None, 
                                                        ci=None, 
                                                        scatter_kws={'color': 'black'},
                                                        line_kws={'color': 'black',
                                                        'linewidth': line_width},
                                                        ax = axes[row_axes, column_it], 
                                                        scatter = False,
                                                        logx=True)
            axes[row_axes, column_it] = sns.regplot(x='class_distinctiveness', y='mean', 
                                                data=df_metrics_by_subject_id_aux[df_metrics_by_subject_id_aux['sex']=='M'], 
                                                label="males" if column_it == 2 else None, 
                                                ci=None, 
                                                scatter_kws={'color': custom_palette['M'],
                                                            's': marker_size},
                                                line_kws={'color': custom_palette['M'],
                                                        'linewidth': line_width},
                                                ax = axes[row_axes, column_it],
                                                logx=True)
            axes[row_axes, column_it] = sns.regplot(x='class_distinctiveness', y='mean', 
                                                data=df_metrics_by_subject_id_aux[df_metrics_by_subject_id_aux['sex']=='F'], 
                                                label=f"females" if column_it == 2 else None, 
                                                ci=None, 
                                                scatter_kws={'color': custom_palette['F'],
                                                            's': marker_size},
                                                line_kws={'color': custom_palette['F'],
                                                        'linewidth': line_width},
                                                ax = axes[row_axes, column_it],
                                                logx=True)

            axes[row_axes, column_it].set_title(f'')  
            # axes[row_axes, column_it].set_xlabel('class distinctiveness')  
            # axes[row_axes, column_it].set_ylabel(ylabels[column_it])  
            axes[row_axes, column_it].tick_params(axis='both', which='major')  
            axes[row_axes, column_it].grid(True)

            if column_it != 2:
                axes[row_axes, column_it].set_ylim([0., 1.05])

            # Plot the regression plot log scale
            axes_log[row_axes, column_it] = sns.regplot(x='log_class_distinctiveness', y='mean', data=df_metrics_by_subject_id_aux,
                                                        label=None, 
                                                        ci=None, 
                                                        scatter_kws={'color': 'black'},
                                                        line_kws={'color': 'black',
                                                        'linewidth': line_width},
                                                        ax = axes_log[row_axes, column_it], 
                                                        scatter = False,
                                                        )
            axes_log[row_axes, column_it] = sns.regplot(x='log_class_distinctiveness', y='mean', 
                                                data=df_metrics_by_subject_id_aux[df_metrics_by_subject_id_aux['sex']=='M'], 
                                                label="males" if column_it == 2 else None, 
                                                ci=None, 
                                                scatter_kws={'color': custom_palette['M'],
                                                            's': marker_size},
                                                line_kws={'color': custom_palette['M'],
                                                        'linewidth': line_width},
                                                ax = axes_log[row_axes, column_it],
                                                )
            axes_log[row_axes, column_it] = sns.regplot(x='log_class_distinctiveness', y='mean', 
                                                data=df_metrics_by_subject_id_aux[df_metrics_by_subject_id_aux['sex']=='F'], 
                                                label=f"females" if column_it == 2 else None, 
                                                ci=None, 
                                                scatter_kws={'color': custom_palette['F'],
                                                            's': marker_size},
                                                line_kws={'color': custom_palette['F'],
                                                        'linewidth': line_width},
                                                ax = axes_log[row_axes, column_it],
                                                )


            axes_log[row_axes, column_it].tick_params(axis='both', which='major')  
            axes_log[row_axes, column_it].grid(True)

            if column_it != 2:
                axes_log[row_axes, column_it].set_ylim([0., 1.05])

            column_it += 1

        for ax in axes_log[:, :].flatten():
            ax.set_xlabel('')
            ax.set_ylabel('')   
            
        for ax, col in zip(axes_log[0], column_titles):
            ax.set_title(col, fontsize=10)

        for ax, row in zip(axes_log[:, 0], row_titles):
            ax.set_ylabel(row, fontsize=10, fontweight='bold', rotation=90, labelpad=20)
        
        for ax in axes_log[-1, :]:
            ax.set_xlabel('class distinctiveness')

        # axes[row_axes, 1].set_title(row_titles[row_axes], pad=title_pad, loc='center')
        # axes_log[row_axes, 1].set_title(row_titles[row_axes], pad=title_pad, loc='center')

        row_axes += 1

    

    axes[0, -1].legend()
    axes_log[0, -1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(path_to_save, 
                             f'scatter_plots_class_distinctiveness_{metric_name}_test.png'), 
                             format='png',dpi=400, bbox_inches="tight")
    fig.savefig(os.path.join(path_to_save, 
                             f'scatter_plots_class_distinctiveness_{metric_name}_test.svg'), 
                             format='svg',dpi=400, bbox_inches="tight")

    fig_log.tight_layout()
    fig_log.savefig(os.path.join(path_to_save, 
                                 f'scatter_plots_class_distinctiveness_{metric_name}_log_test.png'), 
                                 format='png',dpi=400, bbox_inches="tight")
    fig_log.savefig(os.path.join(path_to_save,
                                 f'scatter_plots_class_distinctiveness_{metric_name}_log_test.svg'), 
                                 format='svg',dpi=400, bbox_inches="tight")