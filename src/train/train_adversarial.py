"""
Adversarial world modeling - finetune on worst-case perturbations.

Uses FGSM (single-step) to smooth the action loss landscape.
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

from src.models.transformer_world_model import TransformerWorldModel
from src.data import SequenceDataset
from src.utils.device import get_device, set_seed

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
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    if alpha is None:
        alpha = max(eps_z, eps_a)
    
    for batch in dataloader:
        if len(batch) == 3:
            z, a, z_next = batch
            z = z.to(device)
            a = a.to(device)
            z_next = z_next.to(device)
            
            # Encode images if using encoder
            if hasattr(model, 'encoder') and model.encoder is not None:
                with torch.no_grad():
                    if z.dim() == 5: 
                        B, T, C, H, W = z.shape
                        z = model.encode(z.view(-1, C, H, W)).view(B, T, -1)
                        z_next = model.encode(z_next.view(-1, C, H, W)).view(B, T, -1)
                    else:
                        z = model.encode(z)
                        z_next = model.encode(z_next)
            
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
            
            if isinstance(model, TransformerWorldModel):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
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
        for batch in dataloader:
            if len(batch) == 3:
                z, a, z_next = batch
                z = z.to(device)
                a = a.to(device)
                z_next = z_next.to(device)
                
                # Encode images if using encoder
                if hasattr(model, 'encoder') and model.encoder is not None:
                    if z.dim() == 5: 
                        B, T, C, H, W = z.shape
                        z = model.encode(z.view(-1, C, H, W)).view(B, T, -1)
                        z_next = model.encode(z_next.view(-1, C, H, W)).view(B, T, -1)
                    else:
                        z = model.encode(z)
                        z_next = model.encode(z_next)
                
                z_next_pred = model(z, a)
                loss = criterion(z_next_pred, z_next)
                
                total_loss += loss.item()
                n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


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
    parser.add_argument("--use_images", action="store_true", help="Use images instead of states")
    parser.add_argument("--image_model_size", type=str, default="small", help="DINOv2 model size")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "transformer"], help="Model type")
    parser.add_argument("--context_length", type=int, default=16, help="Context length for transformer")
    
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
    
    # Load baseline model
    print(f"Loading baseline model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    state_dim = checkpoint['state_dim']
    if args.use_images and encoder is not None:
        state_dim = encoder.output_dim
        
    if args.model_type == "transformer":
        model = TransformerWorldModel(
            state_dim=state_dim,
            action_dim=checkpoint['action_dim'],
            embed_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers'],
            num_heads=4,
            encoder=encoder,
            max_len=args.context_length + 10 # Buffer
        ).to(device)
    else:
        model = WorldModel(
            state_dim=state_dim,
            action_dim=checkpoint['action_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers'],
            use_residual=checkpoint['use_residual'],
            encoder=encoder,
        ).to(device)
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded baseline model")
    except RuntimeError:
        print("Warning: Could not load state dict. Starting from scratch/random init.")
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    try:
        if args.model_type == "transformer":
            full_dataset = SequenceDataset(args.data_path, context_length=args.context_length, use_images=args.use_images)
        else:
            full_dataset = ExpertDataset(args.data_path, use_images=args.use_images)
    except ValueError as e:
        print(f"Error loading dataset: {e}")
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
    
    # Adaptive perturbation radii
    if args.adaptive_eps:
        # Compute std from first batch
        batch = next(iter(train_loader))
        z_sample, a_sample, _ = batch
        
        if args.use_images:
            z_sample = z_sample.to(device)
            with torch.no_grad():
                if z_sample.dim() == 5: 
                    B, T, C, H, W = z_sample.shape
                    z_sample = model.encode(z_sample.view(-1, C, H, W)).view(B, T, -1)
                else:
                    z_sample = model.encode(z_sample)
        
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

