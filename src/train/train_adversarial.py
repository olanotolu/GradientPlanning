"""
Adversarial World Modeling: Finetune on worst-case perturbations.

Uses FGSM-style single-step adversarial training to smooth the action loss landscape.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.world_model import WorldModel
from src.data import ExpertDataset


def train_epoch_adversarial(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    eps_z: float = 0.1,
    eps_a: float = 0.1,
    alpha: float = None,
    lambda_z: float = 0.05,
    lambda_a: float = 0.05,
):
    """
    Train for one epoch with adversarial perturbations.
    
    Args:
        model: World model
        dataloader: Data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device
        eps_z: Perturbation radius for states
        eps_a: Perturbation radius for actions
        alpha: Step size for perturbation update (defaults to eps)
        lambda_z: Scaling factor for state perturbations
        lambda_a: Scaling factor for action perturbations
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    if alpha is None:
        alpha = max(eps_z, eps_a)
    
    for z, a, z_next in dataloader:
        z = z.to(device)
        a = a.to(device)
        z_next = z_next.to(device)
        
        # Initialize perturbations uniformly in [-eps, eps]
        delta_z = torch.empty_like(z).uniform_(-eps_z, eps_z)
        delta_a = torch.empty_like(a).uniform_(-eps_a, eps_a)
        
        delta_z.requires_grad_(True)
        delta_a.requires_grad_(True)
        
        # Compute loss on perturbed inputs
        z_pert = z + delta_z
        a_pert = a + delta_a
        z_next_pred = model(z_pert, a_pert)
        loss_pert = criterion(z_next_pred, z_next)
        
        # Compute gradients w.r.t. perturbations
        grad_z = torch.autograd.grad(loss_pert, delta_z, create_graph=True)[0]
        grad_a = torch.autograd.grad(loss_pert, delta_a, create_graph=True)[0]
        
        # Update perturbations: FGSM-style (single step with sign)
        delta_z = delta_z + alpha * torch.sign(grad_z)
        delta_a = delta_a + alpha * torch.sign(grad_a)
        
        # Clip perturbations to bounds
        delta_z = torch.clamp(delta_z, -eps_z, eps_z)
        delta_a = torch.clamp(delta_a, -eps_a, eps_a)
        
        # Detach for final forward pass (don't backprop through perturbation update)
        delta_z = delta_z.detach()
        delta_a = delta_a.detach()
        
        # Final forward pass on perturbed inputs
        optimizer.zero_grad()
        z_pert = z + delta_z
        a_pert = a + delta_a
        z_next_pred = model(z_pert, a_pert)
        
        # Loss with scaling factors
        loss = criterion(z_next_pred, z_next)
        if lambda_z > 0:
            loss = loss + lambda_z * torch.mean(delta_z ** 2)
        if lambda_a > 0:
            loss = loss + lambda_a * torch.mean(delta_a ** 2)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


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
            
            z_next_pred = model(z, a)
            loss = criterion(z_next_pred, z_next)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description="Adversarial finetuning of world model")
    parser.add_argument("--data_path", type=str, default="data/expert_data.npz", help="Path to expert data")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/baseline_best.pt", help="Path to baseline checkpoint")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (lower for finetuning)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--eps_z", type=float, default=0.1, help="State perturbation radius")
    parser.add_argument("--eps_a", type=float, default=0.1, help="Action perturbation radius")
    parser.add_argument("--lambda_z", type=float, default=0.05, help="State perturbation scaling")
    parser.add_argument("--lambda_a", type=float, default=0.05, help="Action perturbation scaling")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--adaptive_eps", action="store_true", help="Adaptive perturbation radii from first batch")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load baseline model
    print(f"Loading baseline model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = WorldModel(
        state_dim=checkpoint['state_dim'],
        action_dim=checkpoint['action_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        use_residual=checkpoint['use_residual'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded baseline model")
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    full_dataset = ExpertDataset(args.data_path)
    
    # Split train/val
    n_total = len(full_dataset)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Adaptive perturbation radii
    if args.adaptive_eps:
        # Compute std from first batch
        z_sample, a_sample, _ = next(iter(train_loader))
        eps_z = float(0.1 * z_sample.std().item())
        eps_a = float(0.1 * a_sample.std().item())
        print(f"Adaptive eps_z={eps_z:.4f}, eps_a={eps_a:.4f}")
    else:
        eps_z = args.eps_z
        eps_a = args.eps_a
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    print("\nStarting adversarial finetuning...")
    
    for epoch in range(args.epochs):
        train_loss = train_epoch_adversarial(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            eps_z=eps_z,
            eps_a=eps_a,
            lambda_z=args.lambda_z,
            lambda_a=args.lambda_a,
        )
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / "adversarial_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'state_dim': checkpoint['state_dim'],
                'action_dim': checkpoint['action_dim'],
                'hidden_dim': checkpoint['hidden_dim'],
                'num_layers': checkpoint['num_layers'],
                'use_residual': checkpoint['use_residual'],
            }, checkpoint_path)
            print(f"  Saved best model (val_loss={val_loss:.6f})")
    
    # Save final model
    final_path = output_dir / "adversarial_final.pt"
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'state_dim': checkpoint['state_dim'],
        'action_dim': checkpoint['action_dim'],
        'hidden_dim': checkpoint['hidden_dim'],
        'num_layers': checkpoint['num_layers'],
        'use_residual': checkpoint['use_residual'],
    }, final_path)
    
    print(f"\nAdversarial finetuning complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved to {output_dir}")


if __name__ == "__main__":
    main()

