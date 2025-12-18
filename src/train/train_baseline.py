"""
Baseline training - standard MSE on expert data.

This creates the train-test gap: model works on expert data but fails during planning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.world_model import WorldModel
from src.models.encoders import DINOv2Encoder
from src.data import ExpertDataset
from src.utils.device import get_device, set_seed


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for z, a, z_next in dataloader:
        z = z.to(device)
        a = a.to(device)
        z_next = z_next.to(device)
        
        optimizer.zero_grad()
        
        # Encode images if using encoder
        if model.encoder is not None:
            with torch.no_grad():
                z = model.encode(z)
                z_next = model.encode(z_next)
        
        # Forward pass
        z_next_pred = model(z, a)
        
        # Loss: MSE
        loss = criterion(z_next_pred, z_next)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for z, a, z_next in dataloader:
            z = z.to(device)
            a = a.to(device)
            z_next = z_next.to(device)
            
            # Encode images if using encoder
            if model.encoder is not None:
                z = model.encode(z)
                z_next = model.encode(z_next)
            
            z_next_pred = model(z, a)
            loss = criterion(z_next_pred, z_next)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train baseline world model")
    parser.add_argument("--data_path", type=str, default="data/expert_data.npz", help="Path to expert data")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--use_residual", action="store_true", help="Use residual connection")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_images", action="store_true", help="Use images instead of states")
    parser.add_argument("--image_model_size", type=str, default="small", help="DINOv2 model size")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Device (supports CUDA, MPS for Apple Silicon, or CPU)
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize encoder if using images
    encoder = None
    if args.use_images:
        print(f"Initializing DINOv2 {args.image_model_size} encoder...")
        encoder = DINOv2Encoder(model_size=args.image_model_size, device=str(device))
        print(f"Encoder output dim: {encoder.output_dim}")
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    try:
        full_dataset = ExpertDataset(args.data_path, use_images=args.use_images)
    except ValueError as e:
        print(f"Error loading dataset: {e}")
        print("Did you generate data with --save_images?")
        return
    
    # Split train/val
    n_total = len(full_dataset)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train samples: {n_train}, Val samples: {n_val}")
    
    # Get state/action dimensions
    if args.use_images:
        state_dim = encoder.output_dim
    else:
        z_sample, a_sample, _ = full_dataset[0]
        state_dim = z_sample.shape[0]
        
    # Get action dim from dataset (assumes consistent action dim)
    _, a_sample, _ = full_dataset[0]
    action_dim = a_sample.shape[0]
    
    print(f"State dim (latent): {state_dim}, Action dim: {action_dim}")
    
    # Create model
    model = WorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_residual=args.use_residual,
        encoder=encoder,
    ).to(device)

    
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    print("\nStarting training...")
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / "baseline_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'use_residual': args.use_residual,
            }, checkpoint_path)
            print(f"  Saved best model (val_loss={val_loss:.6f})")
    
    # Save final model
    final_path = output_dir / "baseline_final.pt"
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'use_residual': args.use_residual,
    }, final_path)
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved to {output_dir}")


if __name__ == "__main__":
    main()

