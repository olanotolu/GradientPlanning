"""
Rollout utilities for world model and simulator.
"""

import torch
import numpy as np
from typing import List, Union


def rollout_model(
    model: torch.nn.Module,
    z0: Union[torch.Tensor, np.ndarray],
    a_sequence: Union[torch.Tensor, np.ndarray],
    horizon: int = None,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Rollout world model for a sequence of actions.
    
    Args:
        model: World model
        z0: Initial state [state_dim] or [batch, state_dim]
        a_sequence: Action sequence [horizon, action_dim] or [batch, horizon, action_dim]
        horizon: Optional horizon (inferred if not provided)
        
    Returns:
        State sequence [horizon+1, state_dim] or [batch, horizon+1, state_dim]
    """
    # Convert numpy to torch if needed
    if isinstance(z0, np.ndarray):
        z0 = torch.from_numpy(z0).float()
        return_numpy = True
    else:
        return_numpy = False
    
    if isinstance(a_sequence, np.ndarray):
        a_sequence = torch.from_numpy(a_sequence).float()
    
    # Rollout using model's rollout method
    z_sequence = model.rollout(z0, a_sequence, horizon=horizon)
    
    if return_numpy:
        z_sequence = z_sequence.detach().cpu().numpy()
    
    return z_sequence


def rollout_sim(
    env,
    z0: Union[torch.Tensor, np.ndarray],
    a_sequence: Union[torch.Tensor, np.ndarray],
    horizon: int = None,
) -> np.ndarray:
    """
    Rollout action sequence in real simulator.
    
    Args:
        env: Environment with step(z, a) -> z_next method
        z0: Initial state [state_dim]
        a_sequence: Action sequence [horizon, action_dim]
        horizon: Optional horizon (inferred if not provided)
        
    Returns:
        State sequence [horizon+1, state_dim] as numpy array
    """
    # Convert torch to numpy if needed
    if isinstance(z0, torch.Tensor):
        z0 = z0.detach().cpu().numpy()
    if isinstance(a_sequence, torch.Tensor):
        a_sequence = a_sequence.detach().cpu().numpy()
    
    z0 = np.array(z0, dtype=np.float32)
    a_sequence = np.array(a_sequence, dtype=np.float32)
    
    if horizon is None:
        horizon = len(a_sequence)
    
    # Initialize state sequence
    z_sequence = [z0.copy()]
    z = z0.copy()
    
    # Rollout
    for t in range(horizon):
        a = a_sequence[t]
        z = env.step(z, a)
        z_sequence.append(z.copy())
    
    return np.array(z_sequence, dtype=np.float32)

