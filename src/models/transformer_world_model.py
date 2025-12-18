"""
Transformer Dynamics Model.
Predicts next state from state-action history using causal transformer.
"""

import torch
import torch.nn as nn
import math
from typing import Optional

class TransformerWorldModel(nn.Module):
    """
    Transformer-based world model: f(z_{1:t}, a_{1:t}) -> z_{t+1}
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 100,
        encoder: Optional[nn.Module] = None,
        latent_dim: Optional[int] = None,
    ):
        super().__init__()
        
        if latent_dim is not None:
            state_dim = latent_dim
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.encoder = encoder
        
        # Input embeddings
        self.state_embed = nn.Linear(state_dim, embed_dim)
        self.action_embed = nn.Linear(action_dim, embed_dim)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.head = nn.Linear(embed_dim, state_dim)
        
        self.embed_dim = embed_dim
        self.max_len = max_len
        
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent state."""
        if self.encoder is not None:
            return self.encoder(obs)
        return obs

    def forward(self, z_seq: torch.Tensor, a_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (training/evaluation).
        
        Args:
            z_seq: (B, T, state_dim)
            a_seq: (B, T, action_dim)
            
        Returns:
            z_next_seq: (B, T, state_dim) - prediction for t+1
        """
        B, T, _ = z_seq.shape
        
        # Embed inputs
        # Combine state and action embeddings?
        # Typically we concat or add.
        # Paper uses: input_token_t = MLP(state_t, action_t)
        # Or simple linear projection of cat(state, action).
        # Let's use linear projection of combined.
        # But I defined separate embeddings. Let's merge them.
        
        # Re-define embedding layer in forward? No, use what we have or fix init.
        # Let's use separate embeddings and add them.
        emb = self.state_embed(z_seq) + self.action_embed(a_seq)
        
        # Add positional encoding
        # pos_embed is (1, max_len, D). Slice to (1, T, D)
        emb = emb + self.pos_embed[:, :T, :]
        
        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(z_seq.device)
        
        # Transformer pass
        features = self.transformer(emb, mask=mask, is_causal=True)
        
        # Predict delta or next state?
        # Let's predict delta
        delta = self.head(features)
        return z_seq + delta
        
    def rollout(
        self,
        z0: torch.Tensor,
        a_sequence: torch.Tensor,
        horizon: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive rollout.
        
        Args:
            z0: (B, state_dim) or (state_dim)
            a_sequence: (B, H, action_dim) or (H, action_dim)
            
        Returns:
            z_sequence: (B, H+1, state_dim)
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
            
        if horizon is None:
            horizon = a_sequence.shape[1]
            
        B = z0.shape[0]
        
        # Initialize sequence with z0
        # For Transformer, we need history.
        # At t=0, we have z0. We need a0 to predict z1.
        # rollout(z0, a_seq)
        # z_seq = [z0]
        # for t in range(horizon):
        #    input: z_{0:t}, a_{0:t}
        #    pred: z_{t+1}
        # This is O(H^2) or O(H^3) if we re-process everything.
        # With KV-caching it's O(H^2).
        # For simplicity (and since H is small ~25-200), we can just pass full history.
        
        current_z_seq = z0.unsqueeze(1) # (B, 1, state_dim)
        current_a_seq = a_sequence[:, :0, :] # Empty initially?
        # Wait, to predict z1, we need z0 and a0.
        # forward(z_{0:t}, a_{0:t}) -> z_{1:t+1}
        
        # Autoregressive loop
        z_preds = [z0]
        
        for t in range(horizon):
            # Current history: z_{0:t}, a_{0:t}
            # We want to predict z_{t+1}
            
            # Prepare inputs
            # z_seq input: z_{0:t}
            # a_seq input: a_{0:t}
            
            # We take a_sequence[:, t]
            a_t = a_sequence[:, t:t+1, :]
            
            if t == 0:
                z_in = z0.unsqueeze(1)
                a_in = a_t
            else:
                # Append last prediction
                z_last = z_preds[-1].unsqueeze(1)
                z_in = torch.cat([current_z_seq, z_last], dim=1) if t > 0 else z0.unsqueeze(1)
                # But careful, I am appending to list z_preds.
                # Construct z_in from z_preds
                z_in = torch.stack(z_preds, dim=1)
                
                a_in = a_sequence[:, :t+1, :]
            
            # Forward pass
            # We only need the last prediction
            pred_seq = self.forward(z_in, a_in)
            z_next = pred_seq[:, -1, :]
            
            z_preds.append(z_next)
            
        z_out = torch.stack(z_preds, dim=1)
        
        if squeeze_output and squeeze_a:
            z_out = z_out.squeeze(0)
            
        return z_out

