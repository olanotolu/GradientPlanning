"""
Device utilities for PyTorch - supports CUDA, MPS (Apple Silicon), and CPU.
"""

import torch


def get_device() -> torch.device:
    """
    Get the best available device for PyTorch.
    Priority: CUDA > MPS (Apple Silicon) > CPU
    
    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    Handles CUDA, MPS, and CPU appropriately.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS doesn't have manual_seed_all, but manual_seed works


