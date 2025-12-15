"""
Cross-Entropy Method (CEM) Baseline Planner

Gradient-free planning using iterative sampling and elite selection.
"""

import torch
import numpy as np
from typing import Optional


def plan(
    world_model: torch.nn.Module,
    z0: torch.Tensor,
    z_goal: torch.Tensor,
    horizon: int = 25,
    n_iterations: int = 10,
    n_samples: int = 100,
    n_elites: int = 10,
    action_max: float = 0.1,
    init_mean: Optional[torch.Tensor] = None,
    init_std: float = 0.5,
) -> torch.Tensor:
    """
    Plan action sequence using Cross-Entropy Method.
    
    Args:
        world_model: World model f(z, a) -> z_next
        z0: Initial state [state_dim] or [batch, state_dim]
        z_goal: Goal state [state_dim] or [batch, state_dim]
        horizon: Planning horizon
        n_iterations: Number of CEM iterations
        n_samples: Number of candidate sequences to sample per iteration
        n_elites: Number of elite sequences to use for updating distribution
        action_max: Maximum action magnitude
        init_mean: Initial mean for action distribution [horizon, action_dim]
        init_std: Initial standard deviation for action distribution
        
    Returns:
        Best action sequence [horizon, action_dim]
    """
    device = next(world_model.parameters()).device
    
    # Ensure tensors are on correct device
    if z0.dim() == 1:
        z0 = z0.unsqueeze(0).to(device)
        z_goal = z_goal.unsqueeze(0).to(device)
        squeeze_output = True
    else:
        z0 = z0.to(device)
        z_goal = z_goal.to(device)
        squeeze_output = False
    
    state_dim = z0.shape[-1]
    action_dim = 2  # Assume 2D actions
    
    # Initialize distribution
    if init_mean is not None:
        mean = init_mean.clone().to(device)
        if mean.dim() == 2:
            mean = mean.unsqueeze(0)
    else:
        mean = torch.zeros(1, horizon, action_dim, device=device)
    
    std = torch.ones(1, horizon, action_dim, device=device) * init_std
    
    # CEM iterations
    for iteration in range(n_iterations):
        # Sample candidate action sequences
        # Sample from Gaussian: a ~ N(mean, std^2)
        noise = torch.randn(n_samples, horizon, action_dim, device=device)
        a_samples = mean + std * noise
        
        # Clip to action bounds
        a_samples = torch.clamp(a_samples, -action_max, action_max)
        
        # Evaluate each sample
        costs = []
        for i in range(n_samples):
            a_seq = a_samples[i:i+1, :, :]  # [1, horizon, action_dim]
            
            # Rollout world model
            with torch.no_grad():
                z_sequence = world_model.rollout(z0, a_seq, horizon=horizon)
                z_final = z_sequence[:, -1, :]  # [1, state_dim]
                
                # Cost: distance to goal
                cost = torch.mean((z_final - z_goal) ** 2).item()
                costs.append(cost)
        
        costs = np.array(costs)
        
        # Select elites (lowest cost)
        elite_indices = np.argsort(costs)[:n_elites]
        elite_actions = a_samples[elite_indices]  # [n_elites, horizon, action_dim]
        
        # Update distribution from elites
        mean = torch.mean(elite_actions, dim=0, keepdim=True)  # [1, horizon, action_dim]
        std = torch.std(elite_actions, dim=0, keepdim=True)  # [1, horizon, action_dim]
        
        # Add small minimum std to prevent collapse
        std = torch.clamp(std, min=0.01)
    
    # Return best action sequence (mean of final distribution)
    best_actions = mean.squeeze(0)  # [horizon, action_dim]
    
    if squeeze_output:
        return best_actions
    else:
        return best_actions.unsqueeze(0)

