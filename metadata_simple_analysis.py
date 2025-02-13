""" This scripts only process the metadata of the dataset,
with only the specifics columns name of the datasets"""

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import init_paths
from global_config import DIRECTORY_TO_SAVE_ROOT, DATABASES_PATH

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', type=str,default = 'Lee2019_MI')
args = parser.parse_args()

# Charge the configuration of the script
dataset_name = args.dataset_name

# Path to save the results
path_to_save = os.path.join(DIRECTORY_TO_SAVE_ROOT, 'metadata_simple_analysis', f'{dataset_name}')
os.makedirs(path_to_save, exist_ok=True)

# Load metadata
if os.path.exists(os.path.join(DATABASES_PATH, 'databases_information', dataset_name, f'{dataset_name}_database_information.csv')):
    metadata = pd.read_csv(os.path.join(DATABASES_PATH, 'databases_information', dataset_name, f'{dataset_name}_database_information.csv'))

    # Statistics by age
    age_stats = metadata.groupby('sex')['age'].describe()
    age_stats.to_csv(os.path.join(path_to_save,'descriptive_statistics_by_sex.csv'), index=True)

    # Boxplot of the age distribution
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='sex', y='age', data=metadata)
    plt.title('Age Distribution by Sex')
    plt.xlabel('Sex')
    plt.ylabel('Age')
    plt.grid()
    # Save the plot
    plt.savefig(os.path.join(path_to_save,'age_distribution_by_sex.png'))

    # Histogram of the age distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=metadata, x='age', hue='sex', bins=10, kde=True, stat='density', common_norm=False)
    plt.title('Age Distribution Histogram by Sex')
    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.grid()
    plt.savefig(os.path.join(path_to_save,'age_distribution_histogram_by_sex.png'))

    # Conduct t-test
    female_ages = metadata[metadata['sex'] == 'F']['age']
    male_ages = metadata[metadata['sex'] == 'M']['age']

    t_stat, p_value = stats.ttest_ind(female_ages, male_ages, equal_var=False)  # Welch's t-test

    print(f'T-statistic: {t_stat}, P-value: {p_value}')

    # Save the results of the t-test
    alpha = 0.05
    t_test_results = {
        't_statistic': t_stat,
        'p_value': p_value,
        'alpha': alpha,
        'H0': 'Reject H0' if p_value < alpha else 'Accept H0',  # H0: There is no significant difference
        'interpretation': 'There is a significant difference between the ages' if p_value < alpha else 'There is no significant difference between the ages'
    }
    # Save the results of the t-test
    t_test_results = pd.DataFrame([t_test_results])
    t_test_results.to_csv(os.path.join(path_to_save,'t_test_age_comparison.csv'), index=False)
else:
    print(f'The metadata of the dataset {dataset_name} is not available')