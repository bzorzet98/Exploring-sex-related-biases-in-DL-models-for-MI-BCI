import os 
import numpy as np
import pandas as pd

from global_config import DATABASES_PATH
from src.utils.json_utils import load_json_file

class PreprocessedDataset():
    def __init__(self, db_name, config, channels=None):
        
        self.db_name = db_name
        self.config = config
        self.code = db_name
        self.path_to_load = os.path.join(DATABASES_PATH, 'preprocessed_databases', config, db_name)
        
        self.sessions = None
        self.n_sessions = None
        self.sessions_per_subject = None
        self.subject_list = None
        self.sfreq = None
        self.montage = None
        self.channels = None
        self.channels_order = None
        self.event_id = None
        self.interval = None
        self.unit_factor = None
        self.__load_info_db()
        idx = None
        if channels:
            if all(ch in self.channels for ch in channels):
                # First we need to obtain the index of the channels
                idx = [self.channels.index(ch) for ch in channels]
        self.selected_channels =  idx
        
    def __load_info_db(self):
        path = os.path.join(self.path_to_load, 'information_db.json')
        info_db = load_json_file(path)
        
        self.subject_list = info_db['subject_list'] if 'subject_list' in info_db else list(range(1, info_db['nsubjects']+1)) 
        self.sessions = list(range(1, info_db['nsessions']+1))
        self.n_sessions = info_db['nsessions']
        self.sessions_per_subject = info_db['nsessions']
        self.nruns = info_db['nruns']
        self.sfreq = info_db['sfreq']
        self.channels = info_db['channels']
        self.channels_order = info_db['channels']
        self.events = info_db['event_id']
        self.interval = info_db['interval']
        self.unit_factor = info_db['unit_factor']
        
    def select_channels(self, channels):
        if all(ch in self.channels for ch in channels):
            idx = [self.channels.index(ch) for ch in channels]
            self.selected_channels = idx

    def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):
            """
            Get the path to the data for a given subject.

            Args:
                subject (int): The subject number to download data for.
                path (str): Not used in this implementation.
                force_update (bool): Not used in this implementation.
                update_path (str): Not used in this implementation.
                verbose (bool): Not used in this implementation.

            Returns:
                dict: A dict with the file paths to the downloaded data for the specified subject.
            """
            if subject not in self.subject_list:
                raise (ValueError("Invalid subject number"))

            """Only implement the load data from a local path"""
            subject_paths = {}
            for session in range(1, self.n_sessions+1):
                subject_paths[session] = {}
                for run in range(1, self.nruns+1):
                    path_eeg=os.path.join(self.path_to_load, f"subject_{subject}_session_{session}_run_{run}_eeg_preprocessed.npy")
                    path_label=os.path.join(self.path_to_load, f"subject_{subject}_session_{session}_run_{run}_labels.npy")
                    subject_paths[session][run] = {'eeg': path_eeg, 'labels': path_label}
            return subject_paths


    def _get_single_subject_data(self, subject, return_raw = False, return_epochs = False):
        """
        Return data for a single subject.

        Args:
            subject (int): The subject ID.

        Returns:
            dict: A dictionary containing the sessions and runs for the given subject.
        """
        sessions = {}
        file_path_dict = self.data_path(subject)
        
        if not return_raw and not return_epochs:
            # We return the whole data in a dictionary
            for session in range(1, self.n_sessions+1):
                sessions[f"session_{session}"] = {}
                for run in range(1, self.nruns+1):
                    eeg_data = np.load(file_path_dict[session][run]['eeg'])
                    # Keep only the information of the selected channels and keep the order of selected channels
                    if self.selected_channels:
                        eeg_data = eeg_data[:, self.selected_channels, :]
                    labels_eeg = np.load(file_path_dict[session][run]['labels'])
                    metadata = pd.DataFrame({'subject': [subject]*len(labels_eeg), 
                                             'session': [f"session_{session}"]*len(labels_eeg), 
                                             'run': [f"run_{run}"]*len(labels_eeg), 
                                             'labels': labels_eeg})
                    sessions[f"session_{session}"][f"run_{run}"] = {'data_eeg': eeg_data, 'labels_eeg': labels_eeg,
                                                                    'metadata': metadata}
            return sessions
        elif return_raw:
            # We need to implement in the future this part of the code
            pass
        elif return_epochs:
            pass
        else:
            raise ValueError('Invalid return type')

    def get_data(self, subjects = None, sessions=None, return_as_dict = False):
        """
        Return the data for a list of subjects.

        This method retrieves the data, labels, and metadata for a given list of subjects from a dataset.
        The returned data will be used as features for the model.

        Parameters:
        ----------
        dataset : Dataset, optional
            A dataset instance. If not provided, the default dataset of the Preprocessing instance will be used.
        subjects : List of int, optional
            A list of subject numbers. If not provided, all subjects in the dataset will be used.
        return_as_dict : bool, optional
            If True, the data will be returned as a nested dictionary. Each subject, session, and run will have its own dictionary entry.
            If False, the data will be returned as arrays.

        Returns:
        -------
        X : Union[np.ndarray]
            The data that will be used as features for the model.
        labels : np.ndarray
            The labels for training/evaluating the model.
        metadata : pd.DataFrame
            A dataframe containing the metadata. The dataframe will have the following columns:
            - subject: The subject index.
            - session: The session index.
            - run: The run index.
        """
        if subjects is None:
            subjects = self.subject_list
        if return_as_dict:
            data = {}
            for subject in subjects:
                data[f'subject_{subject}'] = self._get_single_subject_data(subject, return_raw = False, return_epochs = False)
            return data
        else:
            X = []
            labels = []
            metadata = []
            for subject in subjects:
                data = self._get_single_subject_data(subject, return_raw = False, return_epochs = False)
                if sessions is None:
                    sessions_key = data.keys()
                else:
                    sessions_key = [f"session_{session}" for session in sessions]
                for session in sessions_key:
                    for run in data[session].keys():
                        X.append(data[session][run]['data_eeg'])
                        labels.append(data[session][run]['labels_eeg'])
                        aux = data[session][run]['metadata']
                        aux['index'] = range(len(aux))
                        metadata.append(aux)        
            X = np.concatenate(X, axis=0)
            labels = np.concatenate(labels, axis=0)
            if len(metadata) == 1:
                metadata = metadata[0]
            else:
                metadata = pd.concat(metadata, ignore_index=True).reset_index(drop=True)
            return X, labels, metadata


