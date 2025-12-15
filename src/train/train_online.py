"""
Online world modeling - finetune on planner rollouts corrected by simulator.

DAgger-style: add planner trajectories to training data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.world_model import WorldModel
from src.data import ExpertDataset
from src.planners.gbp import plan as gbp_plan
from src.envs.wall_door import WallDoorEnv
from src.utils.rollout import rollout_sim
from src.utils.metrics import planning_success


class ReplayBuffer:
    """Simple FIFO replay buffer for storing transitions."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = []
    
    def add(self, z, a, z_next):
        """Add transition to buffer."""
        self.buffer.append((z, a, z_next))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)  # FIFO
    
    def sample(self, n: int):
        """Sample n transitions."""
        indices = np.random.choice(len(self.buffer), size=min(n, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)
    
    def get_dataset(self):
        """Convert buffer to PyTorch dataset."""
        if len(self.buffer) == 0:
            return None
        
        z_list, a_list, z_next_list = zip(*self.buffer)
        z_tensor = torch.stack([torch.from_numpy(np.array(z, dtype=np.float32)) for z in z_list])
        a_tensor = torch.stack([torch.from_numpy(np.array(a, dtype=np.float32)) for a in a_list])
        z_next_tensor = torch.stack([torch.from_numpy(np.array(z_next, dtype=np.float32)) for z_next in z_next_list])
        
        return TensorDataset(z_tensor, a_tensor, z_next_tensor)


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
        
        z_next_pred = model(z, a)
        loss = criterion(z_next_pred, z_next)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def evaluate_success_rate(model, env, device, n_episodes=20, horizon=200, gbp_steps=300, seed=42):
    """
    Evaluate success rate of current model on test episodes.
    
    Args:
        model: World model to evaluate
        env: Environment
        device: Device
        n_episodes: Number of test episodes
        horizon: Planning horizon
        gbp_steps: GBP optimization steps
        seed: Random seed
        
    Returns:
        Success rate (0-1)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    successes = 0
    
    for episode in range(n_episodes):
        # Sample random start and goal on opposite sides of wall
        if np.random.rand() > 0.5:
            start = np.array([
                np.random.uniform(env.bounds[0], -0.5),
                np.random.uniform(env.bounds[2], env.bounds[3])
            ], dtype=np.float32)
            goal = np.array([
                np.random.uniform(0.5, env.bounds[1]),
                np.random.uniform(env.bounds[2], env.bounds[3])
            ], dtype=np.float32)
        else:
            start = np.array([
                np.random.uniform(0.5, env.bounds[1]),
                np.random.uniform(env.bounds[2], env.bounds[3])
            ], dtype=np.float32)
            goal = np.array([
                np.random.uniform(env.bounds[0], -0.5),
                np.random.uniform(env.bounds[2], env.bounds[3])
            ], dtype=np.float32)
        
        z0 = torch.from_numpy(start).float().to(device)
        z_goal = torch.from_numpy(goal).float().to(device)
        
        # Plan
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        
        try:
            actions = gbp_plan(
                model, z0, z_goal, horizon=horizon,
                n_steps=gbp_steps, action_max=env.action_max,
                intermediate_weight=0.1
            )
        except:
            # If planning fails, count as failure
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            continue
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        # Rollout in simulator
        actions = actions.cpu().numpy()
        states = rollout_sim(env, start, actions, horizon=horizon)
        
        # Check success
        if planning_success(states[-1], goal, env.goal_threshold):
            successes += 1
    
    return successes / n_episodes


def main():
    parser = argparse.ArgumentParser(description="Online finetuning of world model")
    parser.add_argument("--data_path", type=str, default="data/expert_data.npz", help="Path to expert data")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/baseline_best.pt", help="Path to baseline checkpoint")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n_iterations", type=int, default=10, help="Number of DAgger iterations")
    parser.add_argument("--n_rollouts_per_iter", type=int, default=10, help="Number of planner rollouts per iteration")
    parser.add_argument("--epochs_per_iter", type=int, default=5, help="Epochs of finetuning per iteration")
    parser.add_argument("--expert_mix_ratio", type=float, default=0.5, help="Ratio of expert data in mixed dataset")
    parser.add_argument("--horizon", type=int, default=25, help="Planning horizon")
    parser.add_argument("--gbp_steps", type=int, default=300, help="GBP optimization steps")
    parser.add_argument("--replay_buffer_size", type=int, default=10000, help="Replay buffer size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment with larger action scale
    env = WallDoorEnv(use_velocity=False, action_max=0.25)
    
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
    
    # Load expert dataset
    print(f"Loading expert dataset from {args.data_path}...")
    expert_dataset = ExpertDataset(args.data_path)
    expert_loader = DataLoader(expert_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Replay buffer for planner-generated data
    replay_buffer = ReplayBuffer(max_size=args.replay_buffer_size)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Track success rates over iterations
    success_rates = []
    
    print("\nStarting online finetuning (DAgger-style)...")
    
    for iteration in range(args.n_iterations):
        print(f"\n=== Iteration {iteration + 1}/{args.n_iterations} ===")
        
        # Generate planner rollouts
        print(f"Generating {args.n_rollouts_per_iter} planner rollouts...")
        for rollout_idx in range(args.n_rollouts_per_iter):
            # Sample random start and goal
            start = env.sample_random_state()
            goal = env.sample_goal()
            
            # Ensure start and goal on opposite sides of wall
            if np.random.rand() > 0.5:
                start[0] = np.random.uniform(env.bounds[0], -0.5)
                goal[0] = np.random.uniform(0.5, env.bounds[1])
            else:
                start[0] = np.random.uniform(0.5, env.bounds[1])
                goal[0] = np.random.uniform(env.bounds[0], -0.5)
            
            # Plan using current model
            z0_tensor = torch.from_numpy(start).float().to(device)
            z_goal_tensor = torch.from_numpy(goal).float().to(device)
            
            # GBP needs gradients enabled - temporarily set model to train mode
            model.train()
            for param in model.parameters():
                param.requires_grad = True
            
            a_sequence = gbp_plan(
                model,
                z0_tensor,
                z_goal_tensor,
                horizon=args.horizon,
                n_steps=args.gbp_steps,
                action_max=env.action_max,
            )
            a_sequence = a_sequence.cpu().numpy()
            
            # Set model back to eval mode for rollout
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            
            # Rollout in real simulator to get corrected states
            z_sim_sequence = rollout_sim(env, start, a_sequence, horizon=args.horizon)
            
            # Add transitions to replay buffer
            for t in range(len(a_sequence)):
                z_t = z_sim_sequence[t]
                a_t = a_sequence[t]
                z_next_t = z_sim_sequence[t + 1]
                replay_buffer.add(z_t, a_t, z_next_t)
        
        print(f"Replay buffer size: {len(replay_buffer)}")
        
        # Create mixed dataset (expert + planner data)
        planner_dataset = replay_buffer.get_dataset()
        
        if planner_dataset is not None and len(planner_dataset) > 0:
            # Mix expert and planner data
            n_expert = int(args.batch_size * args.expert_mix_ratio)
            n_planner = args.batch_size - n_expert
            
            expert_loader_mixed = DataLoader(expert_dataset, batch_size=n_expert, shuffle=True)
            planner_loader = DataLoader(planner_dataset, batch_size=n_planner, shuffle=True)
            
            # Finetune for a few epochs
            print(f"Finetuning for {args.epochs_per_iter} epochs...")
            model.train()  # Set to train mode for finetuning
            for param in model.parameters():
                param.requires_grad = True
            
            for epoch in range(args.epochs_per_iter):
                # Mix batches from expert and planner
                expert_iter = iter(expert_loader_mixed)
                planner_iter = iter(planner_loader)
                
                total_loss = 0.0
                n_batches = 0
                
                try:
                    while True:
                        # Get batches
                        z_expert, a_expert, z_next_expert = next(expert_iter)
                        z_planner, a_planner, z_next_planner = next(planner_iter)
                        
                        # Concatenate
                        z_batch = torch.cat([z_expert, z_planner], dim=0).to(device)
                        a_batch = torch.cat([a_expert, a_planner], dim=0).to(device)
                        z_next_batch = torch.cat([z_next_expert, z_next_planner], dim=0).to(device)
                        
                        # Train step
                        optimizer.zero_grad()
                        z_next_pred = model(z_batch, a_batch)
                        loss = criterion(z_next_pred, z_next_batch)
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()
                        n_batches += 1
                except StopIteration:
                    pass
                
                if n_batches > 0:
                    avg_loss = total_loss / n_batches
                    print(f"  Epoch {epoch + 1}/{args.epochs_per_iter}: loss={avg_loss:.6f}")
        else:
            print("No planner data yet, skipping finetuning")
        
        # Evaluate success rate
        print("Evaluating success rate...")
        success_rate = evaluate_success_rate(
            model, env, device, n_episodes=20, horizon=args.horizon,
            gbp_steps=args.gbp_steps, seed=args.seed + iteration
        )
        success_rates.append(success_rate)
        print(f"Success rate: {success_rate*100:.1f}%")
        
        # Save checkpoint
        checkpoint_path = output_dir / f"online_iter_{iteration + 1}.pt"
        torch.save({
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'replay_buffer_size': len(replay_buffer),
            'success_rate': success_rate,
            'state_dim': checkpoint['state_dim'],
            'action_dim': checkpoint['action_dim'],
            'hidden_dim': checkpoint['hidden_dim'],
            'num_layers': checkpoint['num_layers'],
            'use_residual': checkpoint['use_residual'],
        }, checkpoint_path)
    
    # Save final model
    final_path = output_dir / "online_final.pt"
    torch.save({
        'iteration': args.n_iterations,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'replay_buffer_size': len(replay_buffer),
        'state_dim': checkpoint['state_dim'],
        'action_dim': checkpoint['action_dim'],
        'hidden_dim': checkpoint['hidden_dim'],
        'num_layers': checkpoint['num_layers'],
        'use_residual': checkpoint['use_residual'],
    }, final_path)
    
    print(f"\nOnline finetuning complete!")
    print(f"Final replay buffer size: {len(replay_buffer)}")
    print(f"Models saved to {output_dir}")
    
    # Create success rate plot
    if len(success_rates) > 0:
        plt.figure(figsize=(8, 6))
        iterations = range(1, len(success_rates) + 1)
        plt.plot(iterations, [sr * 100 for sr in success_rates], 'o-', linewidth=2, markersize=8)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Success Rate (%)', fontsize=12)
        plt.title('Success Rate Over Online Finetuning Iterations', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, max(35, max(sr * 100 for sr in success_rates) * 1.2)])
        
        # Save plot
        plot_path = Path("results") / "online_training_success_rate.png"
        plot_path.parent.mkdir(exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Success rate plot saved to {plot_path}")
        plt.close()


if __name__ == "__main__":
    main()

