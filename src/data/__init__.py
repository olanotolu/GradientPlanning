"""Data generation and loading modules."""

from torch.utils.data import Dataset
import numpy as np
import torch


class ExpertDataset(Dataset):
    """PyTorch dataset for expert trajectory data."""
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: Path to .npz file with keys: states, actions, next_states
        """
        data = np.load(data_path)
        self.states = torch.from_numpy(data['states']).float()
        self.actions = torch.from_numpy(data['actions']).float()
        self.next_states = torch.from_numpy(data['next_states']).float()
        
        assert len(self.states) == len(self.actions) == len(self.next_states)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.next_states[idx]

