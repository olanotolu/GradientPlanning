"""Data generation and loading modules."""

from torch.utils.data import Dataset
import numpy as np
import torch


class ExpertDataset(Dataset):
    """PyTorch dataset for expert trajectory data."""
    
    def __init__(self, data_path: str, use_images: bool = False):
        """
        Args:
            data_path: Path to .npz file with keys: states, actions, next_states, (optional: images, next_images)
            use_images: Whether to load and return images instead of states
        """
        data = np.load(data_path)
        self.actions = torch.from_numpy(data['actions']).float()
        self.use_images = use_images
        
        if use_images:
            if 'images' not in data or 'next_images' not in data:
                raise ValueError("Dataset does not contain images, but use_images=True")
            
            # Load images: (N, H, W, 3) -> (N, 3, H, W) and normalize to [0, 1]
            images = data['images']
            next_images = data['next_images']
            
            # Permute to CHW
            images = np.transpose(images, (0, 3, 1, 2))
            next_images = np.transpose(next_images, (0, 3, 1, 2))
            
            # Convert to float and normalize
            self.states = torch.from_numpy(images).float() / 255.0
            self.next_states = torch.from_numpy(next_images).float() / 255.0
        else:
            self.states = torch.from_numpy(data['states']).float()
            self.next_states = torch.from_numpy(data['next_states']).float()
        
        assert len(self.states) == len(self.actions) == len(self.next_states)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.next_states[idx]


class SequenceDataset(Dataset):
    """PyTorch dataset for trajectory sequences (for Transformer)."""
    
    def __init__(self, data_path: str, context_length: int = 16, use_images: bool = False):
        """
        Args:
            data_path: Path to .npz file
            context_length: Length of sequences to return
            use_images: Whether to use images
        """
        data = np.load(data_path)
        self.context_length = context_length
        self.use_images = use_images
        
        # Load raw data
        if use_images:
            if 'images' not in data:
                raise ValueError("Dataset does not contain images")
            states = data['images']
            next_states = data['next_images']
            # Permute to N, C, H, W
            states = np.transpose(states, (0, 3, 1, 2))
            next_states = np.transpose(next_states, (0, 3, 1, 2))
            
            states = torch.from_numpy(states).float() / 255.0
            next_states = torch.from_numpy(next_states).float() / 255.0
        else:
            states = torch.from_numpy(data['states']).float()
            next_states = torch.from_numpy(data['next_states']).float()
            
        actions = torch.from_numpy(data['actions']).float()
        
        # Try to load dones/terminals to identify trajectory boundaries
        if 'dones' in data:
            dones = data['dones']
        else:
            # Assume 100 steps per trajectory if not provided (legacy)
            # This is a heuristic and might be wrong if trajectories are shorter
            dones = np.zeros(len(states), dtype=bool)
            # We don't know for sure, but let's assume contiguous blocks
            # Safest is to treat whole dataset as one block if we don't know, 
            # but that's bad for transitions.
            # Ideally we rely on 'dones'.
            pass
            
        # Reconstruct trajectories
        self.trajectories = []
        
        current_states = []
        current_actions = []
        current_next_states = []
        
        for i in range(len(states)):
            current_states.append(states[i])
            current_actions.append(actions[i])
            current_next_states.append(next_states[i])
            
            # If done or max steps reached (we don't know max steps here easily if not saved)
            # But we can use 'dones' if available.
            # If 'dones' not available, we assume contiguous unless we see a large jump?
            # Or just assume 100.
            is_done = dones[i] if 'dones' in data else ((i + 1) % 100 == 0)
            
            if is_done or i == len(states) - 1:
                if len(current_states) > 0:
                    self.trajectories.append({
                        'states': torch.stack(current_states),
                        'actions': torch.stack(current_actions),
                        'next_states': torch.stack(current_next_states)
                    })
                current_states = []
                current_actions = []
                current_next_states = []
        
        # Create indices for sampling: (traj_idx, start_idx)
        self.indices = []
        for traj_idx, traj in enumerate(self.trajectories):
            T = len(traj['states'])
            if T >= context_length:
                for t in range(T - context_length + 1):
                    self.indices.append((traj_idx, t))
                
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        traj_idx, start_idx = self.indices[idx]
        traj = self.trajectories[traj_idx]
        
        end_idx = start_idx + self.context_length
        
        s_seq = traj['states'][start_idx:end_idx]
        a_seq = traj['actions'][start_idx:end_idx]
        ns_seq = traj['next_states'][start_idx:end_idx]
        
        return s_seq, a_seq, ns_seq
