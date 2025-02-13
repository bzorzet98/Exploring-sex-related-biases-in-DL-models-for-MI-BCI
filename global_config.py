import os
# If you want to change the default paths, you have to edit these lines. 

DATABASES_PATH = os.path.join(os.getcwd(), "EEG_DATABASES")

# You must change this path to the path where you want to save the results
DIRECTORY_TO_SAVE_ROOT = os.path.join(os.getcwd(), "RESULTS")


########################### AVAILABLES DATASETS #############################################   
DATASETS_AVAILABLE = ['Cho2017', 'Lee2019_MI']
DEFAULT_DATASET = 'Cho2017'

# Variables to define the events dicts 
MI_EVENTS_DICT = {"left_hand": 0, "right_hand": 1}
SEX_DICT = {'F' : 0, 'M' : 1}