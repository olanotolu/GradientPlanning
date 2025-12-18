"""
DINOv2 Visual Encoder.
Uses Facebook's DINOv2 model to encode images into latent vectors.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from typing import Optional, Union, List

class DINOv2Encoder(nn.Module):
    """
    Wraps DINOv2 model for visual encoding.
    Freezes all parameters.
    """
    
    def __init__(self, model_size: str = "small", device: str = "cpu"):
        """
        Args:
            model_size: 'small', 'base', 'large', 'giant'
            device: 'cpu', 'cuda', or 'mps' (Apple Silicon)
        """
        super().__init__()
        # Convert torch.device to string if needed
        if isinstance(device, torch.device):
            self.device = str(device)
        else:
            self.device = device
        
        # Map size to model name
        model_names = {
            "small": "dinov2_vits14",
            "base": "dinov2_vitb14",
            "large": "dinov2_vitl14",
            "giant": "dinov2_vitg14"
        }
        
        if model_size not in model_names:
            raise ValueError(f"Unknown model size: {model_size}. Choose from {list(model_names.keys())}")
            
        print(f"Loading DINOv2 {model_size} model...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_names[model_size])
        self.backbone.to(device)
        self.backbone.eval()
        
        # Freeze parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Embedding dimensions
        self.embed_dim = {
            "small": 384,
            "base": 768,
            "large": 1024,
            "giant": 1536
        }[model_size]
        
        # Preprocessing transform
        # DINOv2 expects images in [0, 1], normalized with ImageNet stats
        self.transform = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent vectors.
        
        Args:
            images: Batch of images (B, C, H, W) in range [0, 1] or [0, 255]
            
        Returns:
            latents: (B, embed_dim)
        """
        # Ensure tensor is on correct device
        if images.device != self.device:
            images = images.to(self.device)
            
        # Normalize if in [0, 255]
        if images.max() > 1.0:
            images = images / 255.0
            
        # Apply transform
        x = self.transform(images)
        
        # Forward pass through backbone
        with torch.no_grad():
            features = self.backbone(x)
            
        return features

    @property
    def output_dim(self) -> int:
        return self.embed_dim

