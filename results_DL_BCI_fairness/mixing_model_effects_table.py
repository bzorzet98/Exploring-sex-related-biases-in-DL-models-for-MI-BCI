# Import Libraries
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
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
timestamps_1 = ['20241004_144153', '20241105_185644', '20241113_125922' ]
model_2 = 'CSP+LDA'
timestamps_2 = ['20241210_161906', '20241210_172944', '20241210_181717']
row_names = ['Cho 2017', 'Lee 2019 session 1', 'Lee 2019 session 2']

# Name of script
script_name = os.path.basename(__file__).split('.')[0]
path_to_save = os.path.join(DIRECTORY_TO_SAVE_ROOT, experiment_name, 'results_DL_BCI_fairness', script_name)
os.makedirs(path_to_save, exist_ok=True)
subsets = 'test'
# Metrics Name
metrics_name = ['accuracy', 'auc']

for metric_name in metrics_name:
    df_table_DL = pd.DataFrame(columns=['dataset',  'variables', 'coefficient',	'standar error','z', 'p value', '[0.025-0.975]'])
    df_table_CSPLDA = pd.DataFrame(columns=['dataset',  'variables', 'coefficient',	'standar error','z', 'p value', '[0.025-0.975]'])
    df_table_DL_log = pd.DataFrame(columns=['dataset',  'variables', 'coefficient',	'standar error','z', 'p value', '[0.025-0.975]'])
    df_table_CSPLDA_log = pd.DataFrame(columns=['dataset',  'variables', 'coefficient',	'standar error','z', 'p value', '[0.025-0.975]'])
    for dataset_name,row_name, session, timestamp_1, timestamp_2 in zip(datasets_name, row_names, sessions, timestamps_1, timestamps_2):
        # Path to load and save the results
        path_to_load_DL = os.path.join(DIRECTORY_TO_SAVE_ROOT, f'{experiment_name}', f'{model_1}', f'{dataset_name}_{session}', f'{timestamp_1}','different_analysis', 'compute_metrics_from_outputs')
        path_to_load_CSPLDA = os.path.join(DIRECTORY_TO_SAVE_ROOT, f'{experiment_name}', f'{model_2}', f'{dataset_name}_{session}', f'{timestamp_2}','different_analysis', 'compute_metrics_from_outputs')
        
        # Load the metrics.csv file
        metrics_DL = pd.read_csv(os.path.join(path_to_load_DL, 'metrics.csv'))
        metrics_CSPLDA = pd.read_csv(os.path.join(path_to_load_CSPLDA, 'metrics.csv'))

        # Load the metadata
        class_distinctiveness = pd.read_csv(os.path.join(DIRECTORY_TO_SAVE_ROOT, f'{experiment_name}', 'distinctiveness_coefficent_per_subject', f'{dataset_name}_{session}_class_distinctiveness.csv'))
        class_distinctiveness['log_class_distinctiveness'] = np.log(class_distinctiveness['class_distinctiveness'])
        metadata = pd.read_csv(os.path.join(DATABASES_PATH, 'databases_information', dataset_name, f'{dataset_name}_database_information.csv'))
        metadata['sex_number'] = metadata['sex'].map(SEX_DICT)
        metadata = pd.merge(metadata, class_distinctiveness, left_on='subject_id', right_on='subject_id', how='left')
        metrics_DL = pd.merge(metrics_DL, metadata[['subject_id','sex_number','age','log_class_distinctiveness', 'class_distinctiveness']], left_on='subject_id', right_on='subject_id', how='left')
        metrics_CSPLDA = pd.merge(metrics_CSPLDA, metadata[['subject_id','sex_number','age','log_class_distinctiveness', 'class_distinctiveness']], left_on='subject_id', right_on='subject_id', how='left')


        metrics_DL_test = metrics_DL[(metrics_DL['subset'] == 'test')]
        metrics_CSPLDA_test = metrics_CSPLDA[(metrics_CSPLDA['subset'] == 'test')]

        # Mixed models
        md_DL = sm.MixedLM(metrics_DL_test[metric_name], 
                        metrics_DL_test[['class_distinctiveness', 'sex_number', 'age']], 
                        groups=metrics_DL_test['subject_id'])
        md_CSPLDA= sm.MixedLM(metrics_CSPLDA_test[metric_name], 
                        metrics_CSPLDA_test[['class_distinctiveness', 'sex_number', 'age']], 
                        groups=metrics_CSPLDA_test['subject_id'])
    
        mdf_DL = md_DL.fit()
        mdf_CSPLDA = md_CSPLDA.fit()
        df_DL = mdf_DL.summary().tables[1]
        df_CSPLDA = mdf_CSPLDA.summary().tables[1]
        df_table_DL = pd.concat([df_table_DL, pd.DataFrame({'dataset': [row_name,row_name,row_name,row_name],  
                                                            'variables': list(df_DL.index), 
                                                            'coefficient': df_DL['Coef.'].values, 
                                                            'standar error': df_DL['Std.Err.'].values, 
                                                            'z':df_DL['z'].values,
                                                            'p value': df_DL['P>|z|'].values, 
                                                            '[0.025-0.975]': [f"{df_DL['[0.025'].values[0]} -- {df_DL['0.975]'].values[0]}", 
                                                                              f"{df_DL['[0.025'].values[1]}--{df_DL['0.975]'].values[1]}",
                                                                              f"{df_DL['[0.025'].values[2]}--{df_DL['0.975]'].values[2]}",
                                                                              f"{df_DL['[0.025'].values[3]}--{df_DL['0.975]'].values[3]}"]})], axis=0)
        df_table_CSPLDA = pd.concat([df_table_CSPLDA, pd.DataFrame({'dataset': [row_name,row_name,row_name,row_name],  
                                                            'variables': list(df_CSPLDA.index), 
                                                            'coefficient': df_CSPLDA['Coef.'].values, 
                                                            'standar error': df_CSPLDA['Std.Err.'].values, 
                                                            'z':df_CSPLDA['z'].values,
                                                            'p value': df_CSPLDA['P>|z|'].values, 
                                                            '[0.025-0.975]': [f"{df_CSPLDA['[0.025'].values[0]} -- {df_CSPLDA['0.975]'].values[0]}", 
                                                                              f"{df_CSPLDA['[0.025'].values[1]}--{df_CSPLDA['0.975]'].values[1]}",
                                                                              f"{df_CSPLDA['[0.025'].values[2]}--{df_CSPLDA['0.975]'].values[2]}",
                                                                              f"{df_CSPLDA['[0.025'].values[3]}--{df_CSPLDA['0.975]'].values[3]}"]})], axis=0)    
        
        # Mixed models
        md_DL = sm.MixedLM(metrics_DL_test[metric_name], 
                        metrics_DL_test[['log_class_distinctiveness', 'sex_number', 'age']], 
                        groups=metrics_DL_test['subject_id'])
        md_CSPLDA= sm.MixedLM(metrics_CSPLDA_test[metric_name], 
                        metrics_CSPLDA_test[['log_class_distinctiveness', 'sex_number', 'age']], 
                        groups=metrics_CSPLDA_test['subject_id'])
    
        mdf_DL = md_DL.fit()
        mdf_CSPLDA = md_CSPLDA.fit()
        df_DL = mdf_DL.summary().tables[1]
        df_CSPLDA = mdf_CSPLDA.summary().tables[1]
        df_table_DL_log = pd.concat([df_table_DL_log, pd.DataFrame({'dataset': [row_name,row_name,row_name,row_name],  
                                                            'variables': list(df_DL.index), 
                                                            'coefficient': df_DL['Coef.'].values, 
                                                            'standar error': df_DL['Std.Err.'].values, 
                                                            'z':df_DL['z'].values,
                                                            'p value': df_DL['P>|z|'].values, 
                                                            '[0.025-0.975]': [f"{df_DL['[0.025'].values[0]} -- {df_DL['0.975]'].values[0]}", 
                                                                              f"{df_DL['[0.025'].values[1]}--{df_DL['0.975]'].values[1]}",
                                                                              f"{df_DL['[0.025'].values[2]}--{df_DL['0.975]'].values[2]}",
                                                                              f"{df_DL['[0.025'].values[3]}--{df_DL['0.975]'].values[3]}"]})], axis=0)
        df_table_CSPLDA_log = pd.concat([df_table_CSPLDA_log, pd.DataFrame({'dataset': [row_name,row_name,row_name,row_name],  
                                                            'variables': list(df_CSPLDA.index), 
                                                            'coefficient': df_CSPLDA['Coef.'].values, 
                                                            'standar error': df_CSPLDA['Std.Err.'].values, 
                                                            'z':df_CSPLDA['z'].values,
                                                            'p value': df_CSPLDA['P>|z|'].values, 
                                                            '[0.025-0.975]': [f"{df_CSPLDA['[0.025'].values[0]} -- {df_CSPLDA['0.975]'].values[0]}", 
                                                                              f"{df_CSPLDA['[0.025'].values[1]}--{df_CSPLDA['0.975]'].values[1]}",
                                                                              f"{df_CSPLDA['[0.025'].values[2]}--{df_CSPLDA['0.975]'].values[2]}",
                                                                              f"{df_CSPLDA['[0.025'].values[3]}--{df_CSPLDA['0.975]'].values[3]}"]})], axis=0)
    
    df_table_DL.to_csv(os.path.join(path_to_save, f'{metric_name}_mix_models_DL.csv'),index=False)
    df_table_CSPLDA.to_csv(os.path.join(path_to_save, f'{metric_name}_mix_models_CSPLDA.csv'),index=False)    
    df_table_DL_log.to_csv(os.path.join(path_to_save, f'{metric_name}_mix_models_DL_log.csv'),index=False)
    df_table_CSPLDA_log.to_csv(os.path.join(path_to_save, f'{metric_name}_mix_models_CSPLDA_log.csv'),index=False)
