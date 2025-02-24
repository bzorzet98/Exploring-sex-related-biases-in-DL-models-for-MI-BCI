""" This scripts only process the metadata of the dataset, with only the specifics columns name of the datasets"""
# Import Libraries
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import init_paths
from global_config import DIRECTORY_TO_SAVE_ROOT, DATABASES_PATH

# Charge the configuration of the script
experiment_name = 'DL_BCI_fairness'
datasets_name = ['Cho2017', 'Lee2019_MI']
row_names = ['Cho 2017', 'Lee 2019']

# Name of script
script_name = os.path.basename(__file__).split('.')[0]
path_to_save = os.path.join(DIRECTORY_TO_SAVE_ROOT, experiment_name, 'results_DL_BCI_fairness', script_name)
os.makedirs(path_to_save, exist_ok=True)

# plt.rcParams.update({
#     "font.family": "Times New Roman",  # Set font to Times New Roman
#     "axes.titlesize": 10,             # Title size
#     "axes.labelsize": 9,              # Labels size
#     "xtick.labelsize": 9,             # X-axis tick size
#     "ytick.labelsize": 9,             # Y-axis tick size
#     "legend.fontsize": 9              # Legend size
# })

custom_colors = sns.color_palette('Set1')
custom_colors[0], custom_colors[1] = custom_colors[1], custom_colors[0]  # Swap the colors for 'Males' and 'Females'
custom_palette = {'M': custom_colors[0], 'F': custom_colors[1]}

fig, ax = plt.subplots(2, 1, figsize=(5,7))

it_row = 0

for dataset_name,row_name in zip(datasets_name, row_names):
    metadata = pd.read_csv(os.path.join(DATABASES_PATH, 'databases_information', dataset_name, f'{dataset_name}_database_information.csv'))
    if it_row == 0:
        legend_true = True
    else:   
        legend_true = False
    ax[it_row] = sns.histplot(data=metadata, x='age', hue = "sex", ax=ax[it_row], 
                              bins=10, kde=True, stat='density', 
                              common_norm=False, legend=legend_true,
                              palette=custom_palette)
    ax[it_row].set_ylabel(row_name, labelpad = 20, fontsize=10, fontweight='bold', rotation=90)
    ax[it_row].grid()
    it_row += 1

ax[0].set_xlabel('')
ax[-1].set_xlabel('Age')
plt.subplots_adjust(hspace=0.2)
fig.savefig(os.path.join(path_to_save,'age_distribution_histogram_by_sex.png'), dpi=300, bbox_inches="tight")
