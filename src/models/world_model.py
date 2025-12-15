"""
Simple MLP world model - predicts next state from current state + action.

Paper uses DINOv2 + Transformer, we use a tiny MLP instead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WorldModel(nn.Module):
    """Simple MLP: f(z, a) -> z_next"""
    
    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 3,
        use_residual: bool = True,
    ):
        """use_residual=True means predict delta: z_next = z + delta"""
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_residual = use_residual
        
        input_dim = state_dim + action_dim
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Predict next state from current state and action."""
        x = torch.cat([z, a], dim=-1)
        delta = self.net(x)
        return z + delta if self.use_residual else delta
    
    def rollout(
        self,
        z0: torch.Tensor,
        a_sequence: torch.Tensor,
        horizon: Optional[int] = None,
    ) -> torch.Tensor:
        """Rollout model for action sequence (teacher-forced)."""
        if z0.dim() == 1:
            z0 = z0.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        if a_sequence.dim() == 2:
            a_sequence = a_sequence.unsqueeze(0)
            squeeze_a = True
        else:
            squeeze_a = False
        
        if horizon is None:
            horizon = a_sequence.shape[1]
        
        z_sequence = [z0]
        z = z0
        for t in range(horizon):
            z = self.forward(z, a_sequence[:, t, :])
            z_sequence.append(z)
        
        z_sequence = torch.stack(z_sequence, dim=1)
        if squeeze_a:
            z_sequence = z_sequence.squeeze(0)
        
        return z_sequence

