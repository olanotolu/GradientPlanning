#!/usr/bin/env python3
"""
Visualize loss landscape for gradient-based planning.

Similar to paper's Figure C: grid search over action subspace to show
that adversarial finetuning smooths the optimization landscape.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.models.world_model import WorldModel
from src.planners.gbp import plan as gbp_plan
from src.envs.wall_door import WallDoorEnv
from src.data.make_expert_data import ExpertPolicy


def get_expert_actions(env, start, goal, horizon=200):
    """Get expert action sequence for start->goal."""
    expert = ExpertPolicy(env)
    actions = []
    current = start.copy()
    
    for _ in range(horizon):
        action = expert.get_action(current, goal)
        actions.append(action)
        current = env.step(current, action)
        
        # Early termination if close to goal
        if np.linalg.norm(current[:2] - goal[:2]) < env.goal_threshold:
            break
    
    # Pad to horizon if needed
    while len(actions) < horizon:
        actions.append(np.zeros(2, dtype=np.float32))
    
    return np.array(actions[:horizon], dtype=np.float32)


def compute_loss_on_grid(model, z0, z_goal, a_expert, a_baseline, a_adversarial, 
                        grid_size=50, range_val=1.25, horizon=200):
    """
    Compute loss landscape over 2D subspace.
    
    Args:
        model: World model
        z0: Initial state [state_dim]
        z_goal: Goal state [state_dim]
        a_expert: Expert actions [horizon, action_dim]
        a_baseline: Baseline GBP actions [horizon, action_dim]
        a_adversarial: Adversarial GBP actions [horizon, action_dim]
        grid_size: Grid resolution (grid_size x grid_size)
        range_val: Range for α and β ([-range_val, range_val])
        horizon: Planning horizon
        
    Returns:
        alpha_grid, beta_grid, loss_grid
    """
    device = next(model.parameters()).device
    
    # Convert to tensors
    z0_tensor = torch.from_numpy(z0).float().to(device).unsqueeze(0)
    z_goal_tensor = torch.from_numpy(z_goal).float().to(device).unsqueeze(0)
    a_expert_tensor = torch.from_numpy(a_expert).float().to(device)
    a_baseline_tensor = torch.from_numpy(a_baseline).float().to(device)
    a_adversarial_tensor = torch.from_numpy(a_adversarial).float().to(device)
    
    # Define subspace
    alpha_vec = (a_baseline_tensor - a_expert_tensor).cpu().numpy()
    beta_vec = (a_adversarial_tensor - a_expert_tensor).cpu().numpy()
    
    # Normalize to unit vectors for grid search
    alpha_norm = np.linalg.norm(alpha_vec)
    beta_norm = np.linalg.norm(beta_vec)
    
    if alpha_norm < 1e-6:
        alpha_vec = np.ones_like(alpha_vec) / np.sqrt(alpha_vec.size)
        alpha_norm = 1.0
    if beta_norm < 1e-6:
        beta_vec = np.ones_like(beta_vec) / np.sqrt(beta_vec.size)
        beta_norm = 1.0
    
    alpha_unit = alpha_vec / alpha_norm
    beta_unit = beta_vec / beta_norm
    
    # Create grid
    alpha_vals = np.linspace(-range_val, range_val, grid_size)
    beta_vals = np.linspace(-range_val, range_val, grid_size)
    alpha_grid, beta_grid = np.meshgrid(alpha_vals, beta_vals)
    
    # Compute loss for each grid point
    loss_grid = np.zeros_like(alpha_grid)
    
    model.eval()
    with torch.no_grad():
        for i in range(grid_size):
            for j in range(grid_size):
                alpha = alpha_vals[i]
                beta = beta_vals[j]
                
                # Construct action sequence in subspace
                a_grid = a_expert + alpha * alpha_unit + beta * beta_unit
                a_grid_tensor = torch.from_numpy(a_grid).float().to(device).unsqueeze(0)
                
                # Rollout model
                z_sequence = model.rollout(z0_tensor, a_grid_tensor, horizon=horizon)
                z_final = z_sequence[:, -1, :]
                
                # Compute loss
                loss = torch.mean((z_final - z_goal_tensor) ** 2).item()
                loss_grid[j, i] = loss  # Note: j is row (beta), i is col (alpha)
    
    return alpha_grid, beta_grid, loss_grid


def main():
    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = WallDoorEnv(use_velocity=False, action_max=0.25, goal_threshold=1.0)
    
    # Fixed start and goal
    start = np.array([-1.5, 0.5], dtype=np.float32)
    goal = np.array([1.5, -0.5], dtype=np.float32)
    
    z0 = torch.from_numpy(start).float().to(device)
    z_goal = torch.from_numpy(goal).float().to(device)
    
    horizon = 200
    
    print("Loading models...")
    
    # Load baseline model
    baseline_ckpt = torch.load("checkpoints/baseline_best.pt", map_location=device)
    baseline_model = WorldModel(**{k: baseline_ckpt[k] for k in 
        ['state_dim', 'action_dim', 'hidden_dim', 'num_layers', 'use_residual']}).to(device)
    baseline_model.load_state_dict(baseline_ckpt['model_state_dict'])
    
    # Load adversarial model
    try:
        adv_ckpt = torch.load("checkpoints/adversarial_best.pt", map_location=device)
        adv_model = WorldModel(**{k: adv_ckpt[k] for k in 
            ['state_dim', 'action_dim', 'hidden_dim', 'num_layers', 'use_residual']}).to(device)
        adv_model.load_state_dict(adv_ckpt['model_state_dict'])
        has_adversarial = True
    except:
        print("  Adversarial model not found, using baseline")
        adv_model = baseline_model
        has_adversarial = False
    
    print("Computing action sequences...")
    
    # Get expert actions
    a_expert = get_expert_actions(env, start, goal, horizon=horizon)
    
    # Get baseline GBP actions
    baseline_model.train()
    for param in baseline_model.parameters():
        param.requires_grad = True
    a_baseline = gbp_plan(baseline_model, z0, z_goal, horizon=horizon, 
                         n_steps=300, action_max=0.25, intermediate_weight=0.1).cpu().numpy()
    baseline_model.eval()
    for param in baseline_model.parameters():
        param.requires_grad = False
    
    # Get adversarial GBP actions
    if has_adversarial:
        adv_model.train()
        for param in adv_model.parameters():
            param.requires_grad = True
        a_adversarial = gbp_plan(adv_model, z0, z_goal, horizon=horizon,
                                n_steps=300, action_max=0.25, intermediate_weight=0.1).cpu().numpy()
        adv_model.eval()
        for param in adv_model.parameters():
            param.requires_grad = False
    else:
        a_adversarial = a_baseline.copy()
    
    print("Computing loss landscapes...")
    
    # Compute loss landscape for baseline model
    print("  Baseline model...")
    alpha_grid, beta_grid, loss_baseline = compute_loss_on_grid(
        baseline_model, start, goal, a_expert, a_baseline, a_adversarial,
        grid_size=50, range_val=1.25, horizon=horizon
    )
    
    # Compute loss landscape for adversarial model
    if has_adversarial:
        print("  Adversarial model...")
        _, _, loss_adversarial = compute_loss_on_grid(
            adv_model, start, goal, a_expert, a_baseline, a_adversarial,
            grid_size=50, range_val=1.25, horizon=horizon
        )
    else:
        loss_adversarial = loss_baseline.copy()
    
    print("Creating visualizations...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 6))
    
    # 3D surface plots
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(alpha_grid, beta_grid, loss_baseline, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('α (baseline - expert)')
    ax1.set_ylabel('β (adversarial - expert)')
    ax1.set_zlabel('Loss')
    ax1.set_title('Baseline Model Loss Landscape')
    plt.colorbar(surf1, ax=ax1, shrink=0.5)
    
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(alpha_grid, beta_grid, loss_adversarial, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('α (baseline - expert)')
    ax2.set_ylabel('β (adversarial - expert)')
    ax2.set_zlabel('Loss')
    ax2.set_title('Adversarial Model Loss Landscape')
    plt.colorbar(surf2, ax=ax2, shrink=0.5)
    
    # Contour plot comparison
    ax3 = fig.add_subplot(133)
    contour1 = ax3.contour(alpha_grid, beta_grid, loss_baseline, levels=20, colors='blue', alpha=0.5, linestyles='--')
    contour2 = ax3.contour(alpha_grid, beta_grid, loss_adversarial, levels=20, colors='red', alpha=0.5)
    ax3.clabel(contour1, inline=True, fontsize=8, fmt='%.2f')
    ax3.set_xlabel('α (baseline - expert)')
    ax3.set_ylabel('β (adversarial - expert)')
    ax3.set_title('Loss Landscape Comparison')
    ax3.legend(['Baseline', 'Adversarial'])
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path("results") / "loss_landscape.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved loss landscape to {output_path}")
    plt.close()
    
    # Print statistics
    print("\nLoss Landscape Statistics:")
    print(f"  Baseline - Min: {loss_baseline.min():.4f}, Max: {loss_baseline.max():.4f}, Std: {loss_baseline.std():.4f}")
    if has_adversarial:
        print(f"  Adversarial - Min: {loss_adversarial.min():.4f}, Max: {loss_adversarial.max():.4f}, Std: {loss_adversarial.std():.4f}")
        print(f"  Smoothness improvement: {loss_baseline.std() / loss_adversarial.std():.2f}x")


if __name__ == '__main__':
    main()

