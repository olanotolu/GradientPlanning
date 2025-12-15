"""
Gradient-based planner - optimize actions by backprop through world model.

Based on: Parthasarathy et al. 2024, arXiv:2512.09929
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
    Plan action sequence to reach goal using gradient descent.
    
    Uses tanh parameterization to keep actions bounded: a = action_max * tanh(u)
    """
    device = next(world_model.parameters()).device
    
    # Handle batch dimension and device
    if z0.dim() == 1:
        z0 = z0.unsqueeze(0).to(device).requires_grad_(True)
        z_goal = z_goal.unsqueeze(0).to(device)
        squeeze_output = True
    else:
        z0 = z0.to(device).requires_grad_(True)
        z_goal = z_goal.to(device)
        squeeze_output = False
    
    action_dim = 2
    
    # Initialize action parameters (unbounded, use tanh to bound)
    if init_actions is not None:
        u = torch.atanh(torch.clamp(init_actions / action_max, -0.99, 0.99))
        if u.dim() == 2:
            u = u.unsqueeze(0)
        u = u.to(device)
    else:
        u = torch.randn(1, horizon, action_dim, device=device) * 0.1
    
    u.requires_grad_(True)
    optimizer = optim.Adam([u], lr=lr)
    
    for step in range(n_steps):
        optimizer.zero_grad()
        a_sequence = action_max * torch.tanh(u)
        z_sequence = world_model.rollout(z0, a_sequence, horizon=horizon)
        
        z_final = z_sequence[:, -1, :]
        final_loss = torch.mean((z_final - z_goal) ** 2)
        
        if intermediate_weight > 0:
            intermediate_loss = sum(torch.mean((z_sequence[:, t, :] - z_goal) ** 2) 
                                   for t in range(1, horizon + 1))
            loss = final_loss + intermediate_weight * intermediate_loss / horizon
        else:
            loss = final_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_([u], max_norm=1.0)
        optimizer.step()
    
    a_sequence = action_max * torch.tanh(u)
    
    if squeeze_output:
        a_sequence = a_sequence.squeeze(0)  # [horizon, action_dim]
    
    return a_sequence.detach()

