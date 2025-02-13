import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import init_paths

from src.preprocessing import PreprocessEEG
from src.utils import save_dict_as_json,load_json_file
from src.utils.imports import import_class
from global_config import DATABASES_PATH

parser = argparse.ArgumentParser(description='Load database for preprocessing.')

# Add arguments for the two flags
parser.add_argument('--script_config', type=str,
                    default = "classical_preprocessing_Cho2017")
parser.add_argument('--save_scripts', type=bool, default = False)

args = parser.parse_args()

####################################### IMPORT THE CONFIGURATION FILE #############################################
script_config_ = args.script_config
save_scripts = args.save_scripts

# Current path
current_folder_path = os.path.dirname(os.path.abspath(__file__))

# Load configurations
script_config = load_json_file(os.path.join(current_folder_path, 'configs', f'{script_config_}.json'))

####################################### CONFIGURE THE DATABASES TO USES #############################################
script_name = os.path.basename(__file__)[:-3]

# Import the dataset
database_class = import_class(script_config['database']['class_name'],script_config['database']['module_name'])
database = database_class(**script_config['database']['args']) if script_config['database']['args'] is not None else database_class()
database_name = script_config['database']['class_name']
# Select the subjects and sessions
subjects = database.subject_list
sessions = None
if hasattr(database, 'sessions'):
    sessions = database.sessions
else:
    sessions = None

####################################### CONFIGURE THE PREPROCESSING DICTIONARY #############################################
preprocessing_dict = script_config['preprocessing_pipeline']

####################################### CREATE PATH TO SAVE THE PREPROCESSED DATA #############################################
path_to_save = os.path.join(DATABASES_PATH, "preprocessed_databases", 
                            script_config["preprocessing_name"], script_config['database']['class_name'])
os.makedirs(path_to_save, exist_ok = True)
save_dict_as_json(path_to_save = path_to_save, dictionary = preprocessing_dict, file_name='configuration.json')

####################################### LOAD THE INFORMATION OF THE DATABASE #############################################
if os.path.exists(os.path.join(DATABASES_PATH, "databases_information", database_name, f'{database_name}.json')):
    info_db = load_json_file(os.path.join(DATABASES_PATH, "databases_information", database_name, f'{database_name}.json'))
else:
    info_db = None
####################################### CREATE THE EEG PREPROCESSOR #############################################
preprocessor = PreprocessEEG(dataset=database, info_db=info_db)
# Import the functions to use in the pipeline
for i in range(len(preprocessing_dict)):
    preprocessing_dict[i]['function'] =  import_class(preprocessing_dict[i]['class_name'], preprocessing_dict[i]['module_name'])

preprocessor.add_pipeline(preprocessing_dict)

####################################### PREPROCESS THE DATA #############################################
preprocessor.preprocess(path_to_save = path_to_save, save_as_npy=True)
information = preprocessor.get_preprocessed_information()

save_dict_as_json(path_to_save = path_to_save, dictionary = information, file_name='information_db.json')
