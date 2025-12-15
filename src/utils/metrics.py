"""
Metrics for evaluating world model and planning performance.
"""

import torch
import numpy as np
from typing import Union


def world_model_error(
    model: torch.nn.Module,
    z_sequence: Union[torch.Tensor, np.ndarray],
    a_sequence: Union[torch.Tensor, np.ndarray],
    z_sim_sequence: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Compute world model prediction error on a trajectory.
    
    Args:
        model: World model
        z_sequence: Predicted state sequence [horizon+1, state_dim]
        a_sequence: Action sequence [horizon, action_dim]
        z_sim_sequence: Ground-truth simulator state sequence [horizon+1, state_dim]
        
    Returns:
        Mean squared error
    """
    # Convert to torch if needed
    if isinstance(z_sequence, np.ndarray):
        z_sequence = torch.from_numpy(z_sequence).float()
    if isinstance(a_sequence, np.ndarray):
        a_sequence = torch.from_numpy(a_sequence).float()
    if isinstance(z_sim_sequence, np.ndarray):
        z_sim_sequence = torch.from_numpy(z_sim_sequence).float()
    
    device = next(model.parameters()).device
    z_sequence = z_sequence.to(device)
    a_sequence = a_sequence.to(device)
    z_sim_sequence = z_sim_sequence.to(device)
    
    # Compute one-step prediction errors
    errors = []
    horizon = len(a_sequence)
    
    for t in range(horizon):
        z_t = z_sequence[t:t+1, :]  # [1, state_dim]
        a_t = a_sequence[t:t+1, :]  # [1, action_dim]
        z_next_sim = z_sim_sequence[t+1:t+2, :]  # [1, state_dim]
        
        # Predict next state
        z_next_pred = model(z_t, a_t)
        
        # Compute error
        error = torch.mean((z_next_pred - z_next_sim) ** 2).item()
        errors.append(error)
    
    return float(np.mean(errors))


def planning_success(
    z_final: Union[torch.Tensor, np.ndarray],
    z_goal: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.1,
) -> bool:
    """
    Check if planning succeeded (final state is within threshold of goal).
    
    Args:
        z_final: Final state [state_dim]
        z_goal: Goal state [state_dim]
        threshold: Success threshold (distance)
        
    Returns:
        True if distance < threshold
    """
    if isinstance(z_final, torch.Tensor):
        z_final = z_final.detach().cpu().numpy()
    if isinstance(z_goal, torch.Tensor):
        z_goal = z_goal.detach().cpu().numpy()
    
    z_final = np.array(z_final, dtype=np.float32)
    z_goal = np.array(z_goal, dtype=np.float32)
    
    # Extract position (first 2 dims)
    pos_final = z_final[:2]
    pos_goal = z_goal[:2]
    
    distance = np.linalg.norm(pos_final - pos_goal)
    return distance < threshold


def distance_to_goal(
    z_final: Union[torch.Tensor, np.ndarray],
    z_goal: Union[torch.Tensor, np.ndarray],
) -> float:
    """Compute distance from final state to goal."""
    if isinstance(z_final, torch.Tensor):
        z_final = z_final.detach().cpu().numpy()
    if isinstance(z_goal, torch.Tensor):
        z_goal = z_goal.detach().cpu().numpy()
    
    z_final = np.array(z_final, dtype=np.float32)
    z_goal = np.array(z_goal, dtype=np.float32)
    
    pos_final = z_final[:2]
    pos_goal = z_goal[:2]
    
    return float(np.linalg.norm(pos_final - pos_goal))

