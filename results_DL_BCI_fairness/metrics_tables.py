# Import Libraries
import os
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import init_paths
from global_config import DIRECTORY_TO_SAVE_ROOT, DATABASES_PATH
from scipy import stats

DIRECTORY_TO_SAVE_ROOT = os.path.join(os.getcwd(), "RESULTS")
# Charge the configuration of the script
experiment_name = 'DL_BCI_fairness'
datasets_name = ['Cho2017', 'Lee2019_MI', 'Lee2019_MI']
sessions = [1,1,2]
model_1 = 'EEGNetv4_SM'
timestamps_1 = ['20250222_111051', '20250222_140710', '20250222_160239' ] # Original timestamps
# timestamps_1 = ['X', 'X', 'X'] # New timestamps , if you want to use new timestamps, replace 'X' with the new timestamp and uncomment this line
model_2 = 'CSP+LDA'
timestamps_2 = ['20250222_180052', '20250222_190405', '20250222_194642'] # Original timestamps
# timestamps_2 = ['X', 'X', 'X'] # New timestamps , if you want to use new timestamps, replace 'X' with the new timestamp and uncomment this line
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


for metric_name in metrics_name:
    df_table = pd.DataFrame(columns=['dataset', 'model', f'global {metric_name}', 'female', 'male', 'p-value'])
    for dataset_name,row_name, session, timestamp_1, timestamp_2 in zip(datasets_name, row_names, sessions, timestamps_1, timestamps_2):
        # Path to load and save the results
        path_to_load_DL = os.path.join(DIRECTORY_TO_SAVE_ROOT, f'{experiment_name}', f'{model_1}', f'{dataset_name}_{session}', f'{timestamp_1}','different_analysis', 'compute_metrics_from_outputs')
        path_to_load_CSPLDA = os.path.join(DIRECTORY_TO_SAVE_ROOT, f'{experiment_name}', f'{model_2}', f'{dataset_name}_{session}', f'{timestamp_2}','different_analysis', 'compute_metrics_from_outputs')
        
        # Load the metrics.csv file
        metrics_DL = pd.read_csv(os.path.join(path_to_load_DL, 'metrics.csv'))
        metrics_CSPLDA = pd.read_csv(os.path.join(path_to_load_CSPLDA, 'metrics.csv'))

        # Load the metadata
        metadata = pd.read_csv(os.path.join(DATABASES_PATH, 'databases_information', dataset_name, f'{dataset_name}_database_information.csv'))
        metrics_DL = pd.merge(metrics_DL, metadata[['subject_id','sex','age']], left_on='subject_id', right_on='subject_id', how='left')
        metrics_CSPLDA = pd.merge(metrics_CSPLDA, metadata[['subject_id','sex','age']], left_on='subject_id', right_on='subject_id', how='left')

        # Obtain the average, std of metrics
        DL_average = metrics_DL[metrics_DL['subset'] == subsets][metric_name].mean()
        DL_std = metrics_DL[metrics_DL['subset'] == subsets][metric_name].std()
        CSPLDA_average = metrics_CSPLDA[metrics_CSPLDA['subset'] == subsets][metric_name].mean()
        CSPLDA_std = metrics_CSPLDA[metrics_CSPLDA['subset'] == subsets][metric_name].std()
        
        # Obtain the average, std of metrics by sex
        DL_female_average = metrics_DL[(metrics_DL['subset'] == subsets) & (metrics_DL['sex'] == 'F')][metric_name].mean()
        DL_female_std = metrics_DL[(metrics_DL['subset'] == subsets) & (metrics_DL['sex'] == 'F')][metric_name].std()
        DL_male_average = metrics_DL[(metrics_DL['subset'] == subsets) & (metrics_DL['sex'] == 'M')][metric_name].mean()
        DL_male_std = metrics_DL[(metrics_DL['subset'] == subsets) & (metrics_DL['sex'] == 'M')][metric_name].std()
        CSPLDA_female_average = metrics_CSPLDA[(metrics_CSPLDA['subset'] == subsets) & (metrics_CSPLDA['sex'] == 'F')][metric_name].mean()
        CSPLDA_female_std = metrics_CSPLDA[(metrics_CSPLDA['subset'] == subsets) & (metrics_CSPLDA['sex'] == 'F')][metric_name].std()
        CSPLDA_male_average = metrics_CSPLDA[(metrics_CSPLDA['subset'] == subsets) & (metrics_CSPLDA['sex'] == 'M')][metric_name].mean()
        CSPLDA_male_std = metrics_CSPLDA[(metrics_CSPLDA['subset'] == subsets) & (metrics_CSPLDA['sex'] == 'M')][metric_name].std()
        
        # Agregate the data by subject_id and subset
        DL_grouped = metrics_DL.groupby(['subject_id', 'subset','sex'])[[metric_name]].mean().reset_index()
        CSPLDA_grouped = metrics_CSPLDA.groupby(['subject_id', 'subset','sex'])[[metric_name]].mean().reset_index()
        # Hypothesis test
        if hyphotesis_test == 'Mann-Whitney-U':
            female_metrics = DL_grouped[(DL_grouped['subset'] == subsets) &  (DL_grouped['sex'] == 'F')][metric_name].values
            male_metrics = DL_grouped[(DL_grouped['subset'] == subsets) &  (DL_grouped['sex'] == 'M')][metric_name].values
            DL_stats, DL_pvalue = stats.mannwhitneyu(female_metrics,male_metrics,alternative='two-sided')
            
            female_metrics = CSPLDA_grouped[(CSPLDA_grouped['subset'] == subsets) &  (CSPLDA_grouped['sex'] == 'F')][metric_name].values
            male_metrics = CSPLDA_grouped[(CSPLDA_grouped['subset'] == subsets) &  (CSPLDA_grouped['sex'] == 'M')][metric_name].values
            CSPLDA_stats, CSPLDA_pvalue = stats.mannwhitneyu(female_metrics,male_metrics,alternative='two-sided')

        df_table = pd.concat([df_table, 
                            pd.DataFrame({'dataset': [row_name, row_name],
                                          'model': ['EEGNet', 'CSP+LDA'],  
                                          f'global {metric_name}': [f'{DL_average:.3f} \u00b1 {DL_std:.3f}', 
                                                                    f'{CSPLDA_average:.3f} \u00b1 {CSPLDA_std:.3f}'],
                                        'female': [f'{DL_female_average:.3f} \u00b1 {DL_female_std:.3f}', 
                                                    f'{CSPLDA_female_average:.3f} \u00b1 {CSPLDA_female_std:.3f}'],
                                        'male': [f'{DL_male_average:.3f} \u00b1 {DL_male_std:.3f}', 
                                                    f'{CSPLDA_male_average:.3f} \u00b1 {CSPLDA_male_std:.3f}'],
                                        'difference ': [f'{DL_female_average - DL_male_average:.3f}', f'{CSPLDA_female_average - CSPLDA_male_average:.3f}'],
                                        'p-value': [f'{DL_pvalue:.4f}', f'{CSPLDA_pvalue:.4f}']})],axis=0)
    
    df_table.to_csv(os.path.join(path_to_save, f'{metric_name}_table.csv'), index = False)

