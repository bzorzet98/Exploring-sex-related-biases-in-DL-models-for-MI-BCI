# Import Libraries
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import init_paths
from global_config import DIRECTORY_TO_SAVE_ROOT, DATABASES_PATH, SEX_DICT
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
dataset_titles = ['Cho 2017', 'Lee 2019 session 1', 'Lee 2019 session 2']
# Name of script
script_name = os.path.basename(__file__).split('.')[0]
path_to_save_tables = os.path.join(DIRECTORY_TO_SAVE_ROOT, experiment_name, 'results_DL_BCI_fairness', script_name)
os.makedirs(path_to_save_tables, exist_ok=True)

# Obtain the metrics names, are the columns that are not subjects_id
sufixes = ['_DL', '_CSPLDA']
metrics_names = ['accuracy', 'auc']
models = ['EEGNet', 'CSP+LDA']

subset = 'test'

row_axes = 0

# Loop for metrics
for metric_name in metrics_names:    
    # Compute the correlation between the class distinctiveness and the metrics
    df_partial_correlation = pd.DataFrame(columns=['dataset','model', 'data', 'r', 'p_value'])
    for dataset_name,dataset_title, session, timestamp_1, timestamp_2 in zip(datasets_name, dataset_titles, sessions, timestamps_1, timestamps_2):
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

        for model in models:    
            if model == 'EEGNet':
                column_name = f'{metric_name}_DL'
            else:
                column_name = f'{metric_name}_CSPLDA'
            df_metrics_by_subject_id_aux = metrics.groupby(['subset', 'subject_id', 'sex', 'age', 'class_distinctiveness'])[column_name].agg(["mean","std"]).reset_index()
            df_metrics_by_subject_id_aux = df_metrics_by_subject_id_aux[df_metrics_by_subject_id_aux['subset']==subset].sort_values('subject_id')
            df_metrics_by_subject_id_aux = df_metrics_by_subject_id_aux[df_metrics_by_subject_id_aux['subset'] == subset]

            # Compute the partial correlations log scale
            df_metrics_by_subject_id_aux['log_class_distinctiveness'] = np.log(df_metrics_by_subject_id_aux['class_distinctiveness'])
           
            # Map with sex_dict if the column sex is not a number
            df_metrics_by_subject_id_aux['sex_number'] = df_metrics_by_subject_id_aux['sex'].map(SEX_DICT)
            partial_corr = pg.partial_corr(data=df_metrics_by_subject_id_aux,
                                x='log_class_distinctiveness',
                                y='mean',
                                covar='sex_number')
            r = partial_corr['r'].values[0]
            p_value = partial_corr['p-val'].values[0]
            df_partial_correlation = pd.concat([df_partial_correlation, 
                                                pd.DataFrame({'dataset':[dataset_title],
                                                              'model':[model],
                                                            'data':['all_data'], 
                                                            'r': [r], 
                                                            'p_value': [p_value], 
                                                            })], axis=0)
        # Save the results
        df_partial_correlation.to_csv(os.path.join(path_to_save_tables, f'{metric_name}_correlation_with_log_class_distinctiveness.csv'), index=False)
         