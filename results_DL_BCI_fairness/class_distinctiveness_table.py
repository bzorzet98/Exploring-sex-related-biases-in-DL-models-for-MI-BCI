# Import Libraries
import os
import argparse
import pandas as pd
import numpy as np
import scipy.stats as stats
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import init_paths
from global_config import DIRECTORY_TO_SAVE_ROOT, DATABASES_PATH


DIRECTORY_TO_SAVE_ROOT = os.path.join(os.getcwd(), "RESULTS")

# Charge the configuration of the script
experiment_name = 'DL_BCI_fairness'
datasets_name = ['Cho2017', 'Lee2019_MI', 'Lee2019_MI']
sessions = [1,1,2]
row_names = ['Cho 2017', 'Lee 2019 session 1', 'Lee 2019 session 2']

# Name of script
script_name = os.path.basename(__file__).split('.')[0]
path_to_save = os.path.join(DIRECTORY_TO_SAVE_ROOT, experiment_name, 'results_DL_BCI_fairness', script_name)
os.makedirs(path_to_save, exist_ok=True)

subsets = 'test'

# Metrics Name
metrics_name = ['accuracy', 'auc']
# Hyphotesis test
alpha = 0.05
hyphotesis_test = 'Mann-Whitney-U'

df_table = pd.DataFrame(columns=['dataset',  'female', 'male', 'p-value'])
df_table_log = pd.DataFrame(columns=['dataset',  'female', 'male', 'p-value'])

for dataset_name,row_name, session in zip(datasets_name, row_names, sessions):
    # Check if metadata is available
    if os.path.exists(os.path.join(DIRECTORY_TO_SAVE_ROOT, f'{experiment_name}', 'distinctiveness_coefficent_per_subject')):
        class_distinctiveness = pd.read_csv(os.path.join(DIRECTORY_TO_SAVE_ROOT, f'{experiment_name}', 'distinctiveness_coefficent_per_subject', f'{dataset_name}_{session}_class_distinctiveness.csv'))
        class_distinctiveness['log_class_distinctiveness'] = np.log(class_distinctiveness['class_distinctiveness'])
        metadata =  pd.read_csv(os.path.join(DATABASES_PATH, 'databases_information', dataset_name, f'{dataset_name}_database_information.csv'))
        metadata = pd.merge(metadata, class_distinctiveness, left_on='subject_id', right_on='subject_id', how='left')
        
        female_data = metadata[metadata['sex'] == 'F']['class_distinctiveness'].values
        male_data = metadata[metadata['sex'] == 'M']['class_distinctiveness'].values

        female_average = female_data.mean()
        female_std = female_data.std()

        male_average = male_data.mean()
        male_std = male_data.std()
        
        # Do the hyphotesis test
        stat_test, p_value_test = stats.mannwhitneyu(female_data, male_data, alternative='two-sided')

        df_table = pd.concat([df_table, 
                              pd.DataFrame({'dataset': [row_name],
                                            'female': [f"{female_average:.3f}\u00b1 {female_std:.3f}"],
                                            'male': [f"{male_average:.3f}\u00b1 {male_std:.3f}"],
                                            'p-value': [f"{p_value_test:.4f}"]})], axis=0)
        
        female_data = metadata[metadata['sex'] == 'F']['log_class_distinctiveness'].values
        male_data = metadata[metadata['sex'] == 'M']['log_class_distinctiveness'].values

        female_average = female_data.mean()
        female_std = female_data.std()

        male_average = male_data.mean()
        male_std = male_data.std()
        
        # Do the hyphotesis test
        stat_test, p_value_test = stats.mannwhitneyu(female_data, male_data, alternative='two-sided')

        df_table_log = pd.concat([df_table_log, 
                              pd.DataFrame({'dataset': [row_name],
                                            'female': [f"{female_average:.3f}\u00b1 {female_std:.3f}"],
                                            'male': [f"{male_average:.3f}\u00b1 {male_std:.3f}"],
                                            'p-value': [f"{p_value_test:.4f}"]})], axis=0)

df_table.to_csv(os.path.join(path_to_save, 'class_distinctiveness_table.csv'), index=False)
df_table_log.to_csv(os.path.join(path_to_save, 'log_class_distinctiveness_table.csv'), index=False)