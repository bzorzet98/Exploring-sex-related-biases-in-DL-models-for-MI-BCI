import numpy as np

from moabb.datasets.base import BaseDataset
from moabb.datasets import utils
import mne
from mne import Epochs
from mne.io import BaseRaw, BaseRaw

import pandas as pd


from copy import deepcopy
from preprocessing_functions import *

def select_montage(db_name):
    db_standard_1005 = ['Shu2022', 'Cho2017', 'Lee2019_MI',]
    db_standard_1020 = ['Dreyer2023_A','Dreyer2023_B','Dreyer2023_C']
    # Here we need to complete with the other databases
    if db_name in db_standard_1005:
        montage = 'standard_1005'
    elif db_name in db_standard_1020:
        montage = 'standard_1020'
    else:
        montage = 'standard_1005'
    return montage

################################# PreprocessEEG class #############################################
class PreprocessEEG():
    def __init__(self, dataset = None, info_db = None):
        """
        Initializes the Preprocessing object.

        Parameters:
        - dataset (optional): The dataset to be preprocessed.

        Attributes:
        - pipelines: A list to store the preprocessing pipelines.
        - dataset: The dataset to be preprocessed.
        - pipelines_availables: A constant list of available preprocessing pipelines.
        """
        self.pipelines = []
        nsessions = None
        nsubjects = None  
        
        subject_list = []
        if self.is_valid(dataset):
            self.dataset = dataset
            subject_list = dataset.subject_list
            nsubjects = len(dataset.subject_list)
            
        self.pipelines_availables = AVAILABLE_PIPELINES
        if info_db is not None:
            self.data_preprocessed_information = info_db
            # Add the keys filters, event_id, interval_crop, unit_factor
            if 'filters' not in self.data_preprocessed_information.keys():
                self.data_preprocessed_information['filters'] = None
            if 'event_id' not in self.data_preprocessed_information.keys():
                self.data_preprocessed_information['event_id'] = None
            if 'interval_crop' not in self.data_preprocessed_information.keys():
                self.data_preprocessed_information['interval_crop'] = None
            if 'unit_factor' not in self.data_preprocessed_information.keys():
                self.data_preprocessed_information['unit_factor'] = None
        else:
            self.data_preprocessed_information = {  
                                                    'nsubjects': nsubjects,
                                                    'subject_list':subject_list,
                                                    'nsessions' : nsessions,
                                                    'nruns': 0,
                                                    'sfreq': None,
                                                    'nchannels': None,
                                                    'channels': None,
                                                    'filters':None,
                                                    'event_id': None,
                                                    'interval_crop': None,
                                                    'unit_factor': None}
                
        
    
    def add_pipeline(self, pipeline, reset_pipeline=True):
        """
        Adds a pipeline to the list of pipelines.

        Parameters:
        - pipeline (dict or list): The pipeline to be added. If it is a dictionary, it should have the keys 'function' and 'kwargs'.
                                    If it is a list, it should contain multiple pipelines.
        - reset_pipeline (bool): Whether to reset the list of pipelines before adding the new one(s). Default is True.
        """
        if reset_pipeline:
            self.pipelines = []
        # pipeline is a dictionary with the following keys: function, kwargs
        if isinstance(pipeline, dict):
            if 'function' in pipeline and 'kwargs' in pipeline:
                if pipeline['function'].__name__ in self.pipelines_availables:
                    self.pipelines.append(pipeline)
                    print(f"The pipeline {pipeline['function'].__name__} has been added to the list of pipelines")
                else:
                    raise ValueError(f"The function {pipeline['function'].__name__} is not available in the pipelines")
            else:
                raise ValueError("The pipeline dictionary must have the keys: function and kwargs")
        elif isinstance(pipeline, list):
            for p in pipeline:
                self.add_pipeline(p, reset_pipeline=False)
    
    def basic_pipeline(self,):
        """
        Returns a list of dictionaries representing the steps of a basic preprocessing pipeline.

        Each dictionary in the list contains the following keys:
        - 'function': The preprocessing function to be applied.
        - 'kwargs': A dictionary of keyword arguments to be passed to the preprocessing function.

        Returns:
        List: A list of dictionaries representing the steps of the preprocessing pipeline.
        """
        self.pipelines = []
        self.pipelines.append({'function': pick_channels, 'kwargs': {'channels': ['C3', 'Cz', 'C4']}})
        self.pipelines.append({'function': filter_raw, 'kwargs': {'l_freq': 0.5, 'h_freq':40.}})
        self.pipelines.append({'function': filter_notch_raw, 'kwargs': {'freqs': 50.}})
        self.pipelines.append({'function': re_referencing_raw, 'kwargs': {'ref_channels': 'average', 'ch_type': 'eeg'}})
        self.pipelines.append({'function': convert_to_epochs, 'kwargs': {'tmin': -0.5, 'tmax': 3.0, 'preload':True}})
        self.pipelines.append({'function': resample_epoch, 'kwargs': {'sfreq': 128}})
        self.pipelines.append({'function': crop_epoch, 'kwargs': {'tmin': 0.5, 'tmax': 2.5,}})
        self.pipelines.append({'function': convert_to_np_array, 'kwargs': {}})
        self.pipelines.append({'function': apply_unit_factor, 'kwargs': {'unit_factor': None}})
        return self.pipelines
        
    def _check_kwargs(self, pipeline, raw):
        if pipeline['function'].__name__ == 'convert_to_epochs':
            pipeline['kwargs'] = self._check_kwargs_convert_epochs(args_dict = pipeline['kwargs'], raw = raw)              
        elif pipeline['function'].__name__ == 'apply_unit_factor':
            pipeline['kwargs']['unit_factor'] = self._check_unit_factor()
            
        return pipeline
    
    def _update_preprocessed_information(self, pipeline, raw):
        if self.data_preprocessed_information['channels'] is None:
            self.data_preprocessed_information['channels'] = raw.ch_names
        if pipeline['function'].__name__ == 'pick_channels':
            if 'channels' in pipeline['kwargs']:
                self.data_preprocessed_information['channels'] = pipeline['kwargs']['channels']
            else:
                self.data_preprocessed_information['channels'] = raw.info.get_channels_type(picks=None)
        elif pipeline['function'].__name__ == 'filter_raw':
            self.data_preprocessed_information['filters'] = pipeline['kwargs']
        elif pipeline['function'].__name__ == 'resample_epoch':
            self.data_preprocessed_information['sfreq'] = pipeline['kwargs']['sfreq']
        elif pipeline['function'].__name__ == 'convert_to_epochs':
            self.data_preprocessed_information['event_id'] = pipeline['kwargs']['event_id']
        elif pipeline['function'].__name__ == 'crop_epoch':
            self.data_preprocessed_information['interval'] = (0., pipeline['kwargs']['tmax']-pipeline['kwargs']['tmin'])
            self.data_preprocessed_information['interval_crop'] = (pipeline['kwargs']['tmin'], pipeline['kwargs']['tmax'])
        elif pipeline['function'].__name__ == 'apply_unit_factor':
            self.data_preprocessed_information['unit_factor'] = pipeline['kwargs']['unit_factor']
        
    def _check_kwargs_convert_epochs(self, args_dict, raw):
        """
        Check and convert the keyword arguments for epoch conversion.

        Args:
            args_dict (dict): Dictionary containing the keyword arguments.
            raw (BaseRaw): Raw data.

        Returns:
            dict: Updated dictionary containing the converted keyword arguments.
        """
        baseline = None if 'baseline' not in args_dict else args_dict['baseline']
        tmin = args_dict['tmin'] + self.dataset.interval[0] if 'tmin' in args_dict else self.dataset.interval[0]
        tmax = args_dict['tmax'] + self.dataset.interval[0] if 'tmax' in args_dict else self.dataset.interval[1]
        if baseline is not None:
            baseline = (
                self.baseline[0] + self.dataset.interval[0],
                self.baseline[1] + self.dataset.interval[0],
            )
            args_dict['tmin'] = baseline[0] if baseline[0] < tmin else tmin
            args_dict['tmax'] = baseline[1] if baseline[1] > tmax else tmax
        args_dict['events'], args_dict['event_id'] = self.obtain_events(raw)
        return args_dict
    
    def _check_unit_factor(self, ):
        """
        Check the unit factor of the dataset.

        If the unit factor is not set, it is initialized to 1e-6.

        Returns:
            float: The unit factor of the dataset.
        """
        if self.dataset.unit_factor is None:
            self.dataset.unit_factor = 1e-6
        return self.dataset.unit_factor
    
    def obtain_events(self, data):
        """
        Obtains the events from the given data.

        Parameters:
        - data: The data from which to obtain the events. It can be either a BaseRaw or an Epochs object.

        Returns:
        - events: The obtained events.
        - event_id: The dictionary mapping event names to event IDs.
        """
        if not self.is_valid(self.dataset):
            raise ValueError("The dataset is not valid.")
        event_id = self.dataset.event_id
        if isinstance(data, BaseRaw):
            # find the events, first check stim_channels then annotations
            stim_channels = mne.utils._get_stim_channel(None, data.info, raise_error=False)
            if len(stim_channels) > 0:
                events = mne.find_events(data, shortest_event=0, verbose=False)
            else:
                events = np.unique(mne.events_from_annotations(data)[0], axis=0)
            # Manually pick the events and event_ids that have asociated events
            mask = np.zeros(len(events), dtype=bool)
            event_id_pick = {}
            for k, e in event_id.items():
                mask = np.logical_or(mask, events[:, 2] == e)
                if np.any(events[:, 2] == e):
                    event_id_pick[k] = e
            events_pick = events[mask]
            return events_pick, event_id_pick
        elif isinstance(data, Epochs):
            return data.events, event_id
            
    def _apply_pipeline(self, raw):  
        """
        Applies a series of preprocessing pipelines to the raw data.

        Parameters:
        raw (mne.io.BaseRaw): The raw data to be preprocessed.

        Returns:
        tuple: A tuple containing the preprocessed data, the corresponding labels, and metadata.
        """
        # Check that raw is an instance of mne.io.Raw
        if not isinstance(raw, BaseRaw):
            raise TypeError("raw is not an instance of mne.io.BaseRaw.")      
        x_ = deepcopy(raw)
        events, event_id = self.obtain_events(raw)
        for pipeline in self.pipelines:     
            pipeline = self._check_kwargs(pipeline = pipeline, raw = x_)
            self._update_preprocessed_information(pipeline = pipeline, raw = x_)
            x_ = pipeline['function'](x_, **pipeline['kwargs'])
            
        if not isinstance(x_, np.ndarray):
            x_ = convert_to_np_array(x_)
            x_ = apply_unit_factor(x_, self.check_unit_factor())
        if events is not None:
            inv_events = {k: v for v, k in event_id.items()}
            labels = np.array([inv_events[e] for e in events[:, -1]])
        else:
            labels = np.array(['None' for _ in range(len(x_))])
        
        metadata = pd.DataFrame(index=range(len(labels)))
        return x_, labels, metadata
    

    def preprocess(self, path_to_save, save_as_npy=True):
        import os
        dataset = self.dataset
        subjects = self.dataset.subject_list
        eeg_filename = 'eeg_preprocessed.npy'
        labels_filename = 'labels.npy'

        if save_as_npy:
            for subject in subjects:
                data = dataset.get_data([subject])
                for _ , sessions in data.items():
                    it_sess = 1
                    for _ , runs in sessions.items():
                        it_run = 1
                        for _ , raw in runs.items():
                            eeg_data, labels_eeg, met = self._apply_pipeline(raw)
                            
                            if isinstance(eeg_data, np.ndarray) and isinstance(labels_eeg, np.ndarray):
                                path = os.path.join(path_to_save, f"subject_{subject}_session_{it_sess}_run_{it_run}_" + eeg_filename)
                                np.save(path, eeg_data)
                                path = os.path.join(path_to_save, f"subject_{subject}_session_{it_sess}_run_{it_run}_" + labels_filename)
                                np.save(path, labels_eeg)
                                print(f'{subject}, session {it_sess}, run {it_run} preprocessed and saved in {path_to_save}')            

                            it_run+=1
                        it_sess+=1
        self.data_preprocessed_information['nruns'] = it_run-1
        self.data_preprocessed_information['nsessions'] = it_sess-1

    def get_data(self, dataset=None, subjects=None, return_as_dict = False):
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
            if dataset is None:
                dataset = self.dataset
                
            
            if not self.is_valid(dataset):
                message = f"Dataset {dataset.code} is not valid for paradigm"
                raise AssertionError(message)
            
            self.data_preprocessed_information['montage'] = select_montage(dataset.__class__.__name__)
            if subjects is None:
                subjects = self.dataset.subject_list
            if return_as_dict:
                data_ = {}
                for subject in subjects:
                    data = dataset.get_data([subject])
                    for subject_, sessions in data.items():
                        data_[f"subject_{subject_}"] = {}
                        it_sess = 0
                        for session, runs in sessions.items():
                            data_[f"subject_{subject_}"][f"session_{it_sess+1}"] = {}
                            it_run = 0
                            for run, raw in runs.items():
                                x, lbs, met = self._apply_pipeline(raw)
                                data_[f"subject_{subject_}"][f"session_{it_sess+1}"] [f"run_{it_run+1}"] = {
                                    'data_eeg':x,
                                    'labels_eeg':lbs,
                                    'metadata':met
                                }
                                if met.shape[0] != x.shape[0]:
                                    print('Error')
                                it_run+=1
                            it_sess+=1
                self.data_preprocessed_information['nruns'] = it_run-1
                self.data_preprocessed_information['nsessions'] = it_sess-1
                return data_
            else:
                X = np.array([])
                labels = []
                metadata = []
                for subject in subjects:
                    data = dataset.get_data([subject])
                    for subject_, sessions in data.items():
                        it_sess = 0
                        for session, runs in sessions.items():
                            it_run = 0
                            for run, raw in runs.items():
                                x, lbs, met = self._apply_pipeline(raw)
                                met["subject"] = subject_
                                met["session"] = session 
                                met["run"] = run 
                                metadata.append(met)
                                X = np.append(X, x, axis=0) if len(X) else x
                                labels = np.append(labels, lbs, axis=0)
                                it_run+=1
                            it_sess+=1
                self.data_preprocessed_information['nruns'] = it_run-1
                self.data_preprocessed_information['nsessions'] = it_sess-1
                metadata = pd.concat(metadata, ignore_index=True).reset_index(drop=True, inplace=True)
                return X, labels, metadata
        
    def is_valid(self, dataset):
            """
            Check if the dataset is a valid instance of moabb.datasets.base.BaseDataset.

            Parameters:
            dataset (moabb.datasets.base.BaseDataset): The dataset to be checked.

            Returns:
            bool: True if the dataset is a valid instance of BaseDataset, False otherwise.
            """
            ret = True
            if not isinstance(dataset, BaseDataset):
                ret = False            
            return ret

    def get_preprocessed_information(self,):
        return self.data_preprocessed_information
    
    def used_events(self, dataset):
        """
        Return the used events for the dataset.

        Parameters:
        dataset (moabb.datasets.base.BaseDataset): The dataset to be checked.

        Returns:
        dict: The dictionary of used events.
        """
        if not self.is_valid(dataset):
            raise ValueError("The dataset is not valid.")
        return dataset.event_id
