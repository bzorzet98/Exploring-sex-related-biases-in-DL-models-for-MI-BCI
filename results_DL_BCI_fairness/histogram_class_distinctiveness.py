# Import Libraries
import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import scipy.stats as stats
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

# Hyphotesis test
alpha = 0.05
hyphotesis_test = 'Mann-Whitney-U'

df_table = pd.DataFrame(columns=['dataset',  'female', 'male', 'p-value'])

# plt.rcParams.update({
#     "font.family": "Times New Roman",  # Set font to Times New Roman
#     "axes.titlesize": 10,             # Title size
#     "axes.labelsize": 9,              # Labels size
#     "xtick.labelsize": 9,             # X-axis tick size
#     "ytick.labelsize": 9,             # Y-axis tick size
#     "legend.fontsize": 9              # Legend size
# })

fig, ax = plt.subplots(3, 2, figsize=(7,5))


it_row = 0
df_test = pd.DataFrame(columns=['dataset', 'stat_test', 'p_value_test'])
for dataset_name,row_name, session in zip(datasets_name, row_names, sessions):
    # Check if metadata is available
    if os.path.exists(os.path.join(DIRECTORY_TO_SAVE_ROOT, f'{experiment_name}', 'distinctiveness_coefficent_per_subject')):

        class_distinctiveness = pd.read_csv(os.path.join(DIRECTORY_TO_SAVE_ROOT, f'{experiment_name}', 'distinctiveness_coefficent_per_subject', f'{dataset_name}_{session}_class_distinctiveness.csv'))
        class_distinctiveness['log_class_distinctiveness'] = np.log(class_distinctiveness['class_distinctiveness'])
        metadata =  pd.read_csv(os.path.join(DATABASES_PATH, 'databases_information', dataset_name, f'{dataset_name}_database_information.csv'))
        metadata = pd.merge(metadata, class_distinctiveness, left_on='subject_id', right_on='subject_id', how='left')
        
        # PLot a histogram
        ax[it_row,0] = sns.histplot(data=metadata, x='class_distinctiveness',ax=ax[it_row,0])
        # ax[it_row,0].set_title(row_name)
        # ax[it_row,0].set_xlabel('Class Distinctiveness')
        # ax[it_row,0].set_ylabel('Frequency')


        sns.histplot(data=metadata, x='log_class_distinctiveness',ax=ax[it_row,1])  
        # ax[it_row,1].set_title(row_name)
        # ax[it_row,1].set_xlabel('Log Class Distinctiveness')
        # ax[it_row,1].set_ylabel('Frequency')
        it_row += 1

        # Normality test
        stat_test, p_value_test = stats.shapiro(metadata['log_class_distinctiveness'])
        stat_test_male, p_value_test_male = stats.shapiro(metadata[metadata['sex']=='M']['log_class_distinctiveness'])
        stat_test_female, p_value_test_female = stats.shapiro(metadata[metadata['sex']=='F']['log_class_distinctiveness'])
        
        df_test = pd.concat([df_test,
                             pd.DataFrame(
                                 {'dataset': [row_name, row_name, row_name],
                                   'data':['whole data','male','female'],
                                     'stat_test': [stat_test,stat_test_male,stat_test_female],
                                       'p_value_test':[p_value_test,p_value_test_male,p_value_test_female]})], axis=0)
        
for ax_ in ax.flat:
    ax_.set(xlabel='', ylabel='')

row_names = ['Cho 2017', 'Lee 2019 session 1', 'Lee 2019 session 2']
for ax_, row_name in zip(ax[:,0], row_names):
  ax_.set_ylabel(row_name, fontweight='bold', fontsize=12, rotation=90, labelpad=20)

column_x_label = ['class distinctiveness', 'log class distinctiveness']
for ax_, col in zip(ax[-1, :], column_x_label):
    ax_.set_xlabel(col,  fontsize=11)

fig.tight_layout()
plt.savefig(os.path.join(path_to_save, 'class_distinctiveness_histogram.png'),dpi=300)
plt.close()

df_test.to_csv(os.path.join(path_to_save, 'class_distinctiveness_normality_test.csv'),index=False)
