import torch.utils.data
import torch
import numpy as np
import os
from src.utils.json_utils import load_json_file

from global_config import DIRECTORY_TO_SAVE_ROOT

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, classification = True):
        # Check if X is a tensor or numpy array
        if isinstance(X,np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float)
        if isinstance(y,np.ndarray):
            if classification:
                self.y = torch.tensor(y, dtype=torch.long)  
            else:
                self.y = torch.tensor(y, dtype=torch.float)
        self.n = X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return self.n
    
    def get_data(self, idx = None):
        if idx is None:
            return  self.X, self.y
        else:
            return  self.X[idx], self.y[idx]
    
    
class EEGGraphDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, config_adjacency_matrix = None, 
                 name_adjacency_matrix = None, 
                 name_electrode_order = None,
                 channels_to_keep = None):
        # Check if X is a tensor or numpy array
        if isinstance(X,np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float)
        if isinstance(y,np.ndarray):
            self.y = torch.tensor(y, dtype=torch.long)  
        self.n = X.shape[0]
        self.edges_weight = None
        self.edges_index = None
        if name_adjacency_matrix is not None and channels_to_keep is not None:
            self.load_edges(config_adjacency_matrix, 
                            name_adjacency_matrix, 
                            name_electrode_order,
                            channels_to_keep)
    
    def load_edges(self,config_adjacency_matrix=None,
                    name_adjacency_matrix = None, 
                    name_electrode_order = None,
                    channels_to_keep = None):
        import networkx as nx
        path_to_load = os.path.join(DIRECTORY_TO_SAVE_ROOT, 
                                    "adjacency_matrix", 
                                    config_adjacency_matrix)
        # Load adjacency matrix npy file
        adjacency_matrix = np.load(os.path.join(path_to_load,name_adjacency_matrix))
        # Load electrodes order json file
        electrodes_order = load_json_file(os.path.join(path_to_load,name_electrode_order))
        # Reorganize adjacency matrix and electrodes order
        if channels_to_keep:
            reorganized_channels = [channel for channel in electrodes_order if channel in channels_to_keep]
            reorganized_indices = [electrodes_order.index(channel) for channel in reorganized_channels]
            adjacency_matrix = adjacency_matrix[reorganized_indices][:, reorganized_indices]

        G = nx.convert_matrix.from_numpy_array(adjacency_matrix)
        self.edges_index = torch.tensor(list(G.edges)).t().contiguous()
        self.edges_weight = torch.tensor(list(nx.get_edge_attributes(G, 'weight').values()))

    def __getitem__(self, idx):
        return {"input": self.X[idx], "edge_index": self.edges_index, "edge_weight": self.edges_weight}, self.y[idx]
    
    def __len__(self):
        return self.n
    
    def get_data(self, idx = None):
        if idx is None:
            return  (self.X, self.edges_index, self.edges_weight) , self.y
        else:
            return  (self.X[idx], self.edges_index, self.edges_weight), self.y[idx]
    
