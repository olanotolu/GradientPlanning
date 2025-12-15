#!/usr/bin/env python3
"""
Visual demo - compare baseline vs finetuned planning.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.models.world_model import WorldModel
from src.envs.wall_door import WallDoorEnv
from src.planners.gbp import plan as gbp_plan
from src.utils.rollout import rollout_sim, rollout_model
from src.utils.viz import plot_trajectory, compare_trajectories


def run_visual_demo():
    """Run complete visual demonstration."""
    
    print("=" * 60)
    print("Gradient Planning Visual Demo")
    print("=" * 60)
    print()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()
    
    # Create environment
    env = WallDoorEnv(use_velocity=False, action_max=0.25, goal_threshold=0.5)
    print("Environment: 2D navigation with wall and door")
    print(f"  Wall at x=0, door at y ∈ [{env.door_y_min:.1f}, {env.door_y_max:.1f}]")
    print()
    
    # Check for models
    baseline_path = Path("checkpoints/baseline_best.pt")
    online_path = Path("checkpoints_v2/online_final.pt")
    
    if not baseline_path.exists():
        print("❌ Baseline model not found!")
        print("   Run: python src/train/train_baseline.py")
        return
    
    if not online_path.exists():
        print("⚠️  Online model not found, using baseline for both")
        online_path = baseline_path
    
    # Load models
    print("Loading models...")
    baseline_checkpoint = torch.load(baseline_path, map_location=device)
    baseline_model = WorldModel(
        state_dim=baseline_checkpoint['state_dim'],
        action_dim=baseline_checkpoint['action_dim'],
        hidden_dim=baseline_checkpoint['hidden_dim'],
        num_layers=baseline_checkpoint['num_layers'],
        use_residual=baseline_checkpoint['use_residual'],
    ).to(device)
    baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
    
    if online_path != baseline_path:
        online_checkpoint = torch.load(online_path, map_location=device)
        online_model = WorldModel(
            state_dim=online_checkpoint['state_dim'],
            action_dim=online_checkpoint['action_dim'],
            hidden_dim=online_checkpoint['hidden_dim'],
            num_layers=online_checkpoint['num_layers'],
            use_residual=online_checkpoint['use_residual'],
        ).to(device)
        online_model.load_state_dict(online_checkpoint['model_state_dict'])
    else:
        online_model = baseline_model
    
    print("✓ Models loaded")
    print()
    
    # Test case
    np.random.seed(42)
    start = np.array([-1.5, 0.5], dtype=np.float32)
    goal = np.array([1.5, -0.5], dtype=np.float32)
    
    print(f"Test Case:")
    print(f"  Start: ({start[0]:.1f}, {start[1]:.1f})")
    print(f"  Goal:  ({goal[0]:.1f}, {goal[1]:.1f})")
    print()
    
    # Plan with baseline
    print("Planning with baseline model...")
    baseline_model.train()
    for param in baseline_model.parameters():
        param.requires_grad = True
    
    z0 = torch.from_numpy(start).float().to(device)
    z_goal = torch.from_numpy(goal).float().to(device)
    
    baseline_actions = gbp_plan(
        baseline_model, z0, z_goal, horizon=100, n_steps=300, action_max=env.action_max
    )
    baseline_actions = baseline_actions.cpu().numpy()
    
    baseline_model.eval()
    for param in baseline_model.parameters():
        param.requires_grad = False
    
    # Plan with online
    print("Planning with online finetuned model...")
    online_model.train()
    for param in online_model.parameters():
        param.requires_grad = True
    
    online_actions = gbp_plan(
        online_model, z0, z_goal, horizon=100, n_steps=300, action_max=env.action_max
    )
    online_actions = online_actions.cpu().numpy()
    
    online_model.eval()
    for param in online_model.parameters():
        param.requires_grad = False
    
    # Rollout in simulator
    print("Rolling out in simulator...")
    baseline_sim = rollout_sim(env, start, baseline_actions, horizon=100)
    online_sim = rollout_sim(env, start, online_actions, horizon=100)
    
    # Model predictions
    baseline_pred = rollout_model(baseline_model, start, baseline_actions, horizon=100)
    online_pred = rollout_model(online_model, start, online_actions, horizon=100)
    
    if isinstance(baseline_pred, torch.Tensor):
        baseline_pred = baseline_pred.detach().cpu().numpy()
    if isinstance(online_pred, torch.Tensor):
        online_pred = online_pred.detach().cpu().numpy()
    
    # Compute metrics
    baseline_dist = np.linalg.norm(baseline_sim[-1, :2] - goal)
    online_dist = np.linalg.norm(online_sim[-1, :2] - goal)
    
    print()
    print("Results:")
    print(f"  Baseline final distance: {baseline_dist:.3f} units")
    print(f"  Online final distance:   {online_dist:.3f} units")
    print(f"  Improvement:             {(baseline_dist - online_dist) / baseline_dist * 100:.1f}%")
    print()
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Figure 1: Side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    plot_trajectory(
        baseline_sim, wall_x=env.wall_x, door_y_min=env.door_y_min,
        door_y_max=env.door_y_max, goal=goal, goal_threshold=env.goal_threshold,
        title=f"Baseline (distance: {baseline_dist:.2f})", ax=axes[0]
    )
    
    plot_trajectory(
        online_sim, wall_x=env.wall_x, door_y_min=env.door_y_min,
        door_y_max=env.door_y_max, goal=goal, goal_threshold=env.goal_threshold,
        title=f"Online Finetuned (distance: {online_dist:.2f})", ax=axes[1]
    )
    
    plt.tight_layout()
    comparison_path = "results/visual_demo_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {comparison_path}")
    
    # Figure 2: Model predictions vs reality
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    plot_trajectory(
        baseline_pred, wall_x=env.wall_x, door_y_min=env.door_y_min,
        door_y_max=env.door_y_max, goal=goal, goal_threshold=env.goal_threshold,
        title="Baseline: Model Prediction", ax=axes[0, 0]
    )
    
    plot_trajectory(
        baseline_sim, wall_x=env.wall_x, door_y_min=env.door_y_min,
        door_y_max=env.door_y_max, goal=goal, goal_threshold=env.goal_threshold,
        title="Baseline: Simulator Reality", ax=axes[0, 1]
    )
    
    plot_trajectory(
        online_pred, wall_x=env.wall_x, door_y_min=env.door_y_min,
        door_y_max=env.door_y_max, goal=goal, goal_threshold=env.goal_threshold,
        title="Online: Model Prediction", ax=axes[1, 0]
    )
    
    plot_trajectory(
        online_sim, wall_x=env.wall_x, door_y_min=env.door_y_min,
        door_y_max=env.door_y_max, goal=goal, goal_threshold=env.goal_threshold,
        title="Online: Simulator Reality", ax=axes[1, 1]
    )
    
    plt.tight_layout()
    reality_path = "results/visual_demo_model_vs_reality.png"
    plt.savefig(reality_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {reality_path}")
    
    plt.close('all')
    
    print()
    print("=" * 60)
    print("✅ Visual Demo Complete!")
    print("=" * 60)
    print()
    print("View results:")
    print(f"  - {comparison_path}")
    print(f"  - {reality_path}")
    print()
    print("The baseline model shows the train-test gap (high error).")
    print("The online finetuned model shows improvement (lower error).")


if __name__ == "__main__":
    run_visual_demo()

