"""
Combined training: Online finetuning → Adversarial finetuning.

Loads online finetuned model and applies adversarial finetuning on top.
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
from src.data import ExpertDataset
from src.train.train_adversarial import train_epoch_adversarial, validate


def main():
    parser = argparse.ArgumentParser(description="Combined training: online → adversarial")
    parser.add_argument("--data_path", type=str, default="data/expert_data.npz", help="Path to expert data")
    parser.add_argument("--online_checkpoint", type=str, default="checkpoints/online_final.pt", help="Path to online finetuned checkpoint")
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
    
    # Load online finetuned model
    print(f"Loading online finetuned model from {args.online_checkpoint}...")
    try:
        checkpoint = torch.load(args.online_checkpoint, map_location=device)
    except FileNotFoundError:
        # Try alternative path
        alt_path = "checkpoints_v2/online_final.pt"
        print(f"  Not found, trying {alt_path}...")
        checkpoint = torch.load(alt_path, map_location=device)
    
    model = WorldModel(
        state_dim=checkpoint['state_dim'],
        action_dim=checkpoint['action_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        use_residual=checkpoint['use_residual'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded online finetuned model")
    
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
    print("\nStarting adversarial finetuning on online model...")
    
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
            checkpoint_path = output_dir / "combined_best.pt"
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
    final_path = output_dir / "combined_final.pt"
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
    
    print(f"\nCombined training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved to {output_dir}")


if __name__ == "__main__":
    main()

