import os

# You must change this path to the path where you have the EEG databases
DATABASES_PATH = os.path.join("/","media", "bzorzet", "EEG_DATABASES")
DIRECTORY_TO_SAVE_ROOT = os.path.join("/","media", "bzorzet", "BCI-Deep-Learning-BZ")


########################### AVAILABLES DATASETS #############################################   
DATASETS_AVAILABLE = ['Cho2017', 'Lee2019_MI']
DEFAULT_DATASET = 'Cho2017'

# Variables to define the events dicts 
MI_EVENTS_DICT = {"left_hand": 0, "right_hand": 1}
SEX_DICT = {'F' : 0, 'M' : 1}