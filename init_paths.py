# init_paths.py
import sys
import os

current_path = os.path.dirname(os.path.abspath(__file__))

if os.getenv('CONDA_DEFAULT_ENV') == 'eeg_environment':
    sys.path.append(os.path.join(current_path, 'src' , 'cross_validators'))
    sys.path.append(os.path.join(current_path, 'src' , 'preprocessing'))
    sys.path.append(os.path.join(current_path, 'src' , 'preprocessed_dataset'))
    sys.path.append(os.path.join(current_path, 'src' , 'utils'))
elif os.getenv('CONDA_DEFAULT_ENV') == 'train_environment':
    sys.path.append(os.path.join(current_path, 'src' , 'cross_validators'))
    sys.path.append(os.path.join(current_path, 'src' , 'networks'))
    sys.path.append(os.path.join(current_path, 'src' , 'preprocessed_dataset'))
    sys.path.append(os.path.join(current_path, 'src' , 'skorch_modules'))
    sys.path.append(os.path.join(current_path, 'src' , 'torch_utils'))
    sys.path.append(os.path.join(current_path, 'src' , 'utils'))
elif os.getenv('CONDA_DEFAULT_ENV') == 'analysis_environment':
    sys.path.append(os.path.join(current_path, 'src' , 'utils'))
