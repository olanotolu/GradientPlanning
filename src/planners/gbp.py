"""
Gradient-Based Planner (GBP)

Optimizes action sequence by backpropagating through world model rollout.
"""

import torch
import torch.optim as optim
from typing import Optional
import numpy as np


def plan(
    world_model: torch.nn.Module,
    z0: torch.Tensor,
    z_goal: torch.Tensor,
    horizon: int = 25,
    n_steps: int = 300,
    lr: float = 1e-3,
    action_max: float = 0.1,
    intermediate_weight: float = 0.0,
    init_actions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Plan action sequence using gradient-based optimization.
    
    Args:
        world_model: World model f(z, a) -> z_next
        z0: Initial state [state_dim] or [batch, state_dim]
        z_goal: Goal state [state_dim] or [batch, state_dim]
        horizon: Planning horizon
        n_steps: Number of optimization steps
        lr: Learning rate for Adam optimizer
        action_max: Maximum action magnitude (actions will be clipped to [-action_max, action_max])
        intermediate_weight: Weight for intermediate goal loss (0 = only final state loss)
        init_actions: Optional initial action sequence [horizon, action_dim]
        
    Returns:
        Optimized action sequence [horizon, action_dim]
    """
    device = next(world_model.parameters()).device
    
    # Ensure tensors are on correct device and have batch dimension
    # Also ensure they require gradients (needed for backprop through model)
    if z0.dim() == 1:
        z0 = z0.unsqueeze(0).to(device).requires_grad_(True)
        z_goal = z_goal.unsqueeze(0).to(device)
        squeeze_output = True
    else:
        z0 = z0.to(device).requires_grad_(True)
        z_goal = z_goal.to(device)
        squeeze_output = False
    
    state_dim = z0.shape[-1]
    action_dim = 2  # Assume 2D actions
    
    # Initialize action parameters u (unbounded)
    if init_actions is not None:
        # Convert actions to unbounded parameters: u = atanh(a / action_max)
        u = torch.atanh(torch.clamp(init_actions / action_max, -0.99, 0.99))
        if u.dim() == 2:
            u = u.unsqueeze(0)
        u = u.to(device)
    else:
        # Random initialization
        u = torch.randn(1, horizon, action_dim, device=device) * 0.1
    
    u.requires_grad_(True)
    
    # Optimizer
    optimizer = optim.Adam([u], lr=lr)
    
    # Optimization loop
    for step in range(n_steps):
        optimizer.zero_grad()
        
        # Convert unbounded parameters to bounded actions: a = action_max * tanh(u)
        a_sequence = action_max * torch.tanh(u)  # [1, horizon, action_dim]
        
        # Rollout world model
        z_sequence = world_model.rollout(z0, a_sequence, horizon=horizon)
        # z_sequence: [1, horizon+1, state_dim]
        
        # Compute loss: distance to goal at final state
        z_final = z_sequence[:, -1, :]  # [1, state_dim]
        final_loss = torch.mean((z_final - z_goal) ** 2)
        
        # Optional: intermediate goal loss
        if intermediate_weight > 0:
            intermediate_loss = 0.0
            for t in range(1, horizon + 1):
                z_t = z_sequence[:, t, :]
                intermediate_loss += torch.mean((z_t - z_goal) ** 2)
            intermediate_loss = intermediate_weight * intermediate_loss / horizon
            loss = final_loss + intermediate_loss
        else:
            loss = final_loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_([u], max_norm=1.0)
        
        optimizer.step()
    
    # Convert back to actions
    a_sequence = action_max * torch.tanh(u)
    
    if squeeze_output:
        a_sequence = a_sequence.squeeze(0)  # [horizon, action_dim]
    
    return a_sequence.detach()

