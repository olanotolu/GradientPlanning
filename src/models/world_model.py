"""
Simple MLP World Model for 2D navigation.

Predicts next state given current state and action.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WorldModel(nn.Module):
    """MLP world model: f(z, a) -> z_next"""
    
    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 3,
        use_residual: bool = True,
    ):
        """
        Args:
            state_dim: Dimension of state z
            action_dim: Dimension of action a
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            use_residual: If True, predict delta: z_next = z + Δ(z, a)
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_residual = use_residual
        
        # Input: concat([z, a])
        input_dim = state_dim + action_dim
        
        # Build MLP
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, state_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Predict next state.
        
        Args:
            z: Current state [batch, state_dim]
            a: Action [batch, action_dim]
            
        Returns:
            Predicted next state [batch, state_dim]
        """
        # Concatenate state and action
        x = torch.cat([z, a], dim=-1)
        
        # Forward through network
        delta = self.net(x)
        
        if self.use_residual:
            # Residual connection: z_next = z + Δ
            z_next = z + delta
        else:
            z_next = delta
        
        return z_next
    
    def rollout(
        self,
        z0: torch.Tensor,
        a_sequence: torch.Tensor,
        horizon: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Rollout world model for a sequence of actions (teacher-forced).
        
        Args:
            z0: Initial state [batch, state_dim] or [state_dim]
            a_sequence: Action sequence [horizon, action_dim] or [batch, horizon, action_dim]
            horizon: Optional horizon (inferred from a_sequence if not provided)
            
        Returns:
            State sequence [horizon+1, state_dim] or [batch, horizon+1, state_dim]
        """
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
        
        batch_size = z0.shape[0]
        
        if horizon is None:
            horizon = a_sequence.shape[1]
        
        # Initialize state sequence
        z_sequence = [z0]
        z = z0
        
        # Rollout
        for t in range(horizon):
            a = a_sequence[:, t, :]
            z = self.forward(z, a)
            z_sequence.append(z)
        
        # Stack into tensor
        z_sequence = torch.stack(z_sequence, dim=1)  # [batch, horizon+1, state_dim]
        
        if squeeze_a:
            z_sequence = z_sequence.squeeze(0)  # [horizon+1, state_dim]
        
        return z_sequence

