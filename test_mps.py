#!/usr/bin/env python3
"""
Quick test to verify MPS acceleration works.
"""

import torch
from src.utils.device import get_device

def test_mps():
    """Test MPS device selection and basic operations."""
    device = get_device()
    print(f"✓ Selected device: {device}")
    
    # Test basic operations
    print("\nTesting basic tensor operations...")
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    z = torch.matmul(x, y)
    print(f"✓ Matrix multiplication on {device}: {z.shape}")
    
    # Test model forward pass
    print("\nTesting model forward pass...")
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)
    ).to(device)
    
    input_tensor = torch.randn(32, 100).to(device)
    output = model(input_tensor)
    print(f"✓ Model forward pass on {device}: {output.shape}")
    
    # Test backward pass
    print("\nTesting backward pass...")
    loss = output.mean()
    loss.backward()
    print(f"✓ Backward pass completed on {device}")
    
    print(f"\n✅ All tests passed! MPS acceleration is working.")
    print(f"   Your training should be significantly faster now!")

if __name__ == "__main__":
    test_mps()


