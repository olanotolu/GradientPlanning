#!/usr/bin/env python3
"""
Demo script: Visualize the train-test gap and how finetuning fixes it.

Shows baseline vs online finetuned planning trajectories.
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


def demo_planning_comparison():
    """Compare baseline vs online finetuned planning."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = WallDoorEnv(use_velocity=False, action_max=0.25, goal_threshold=0.5)
    
    # Load baseline model
    print("Loading baseline model...")
    baseline_checkpoint = torch.load("checkpoints_v2/baseline_best.pt", map_location=device)
    baseline_model = WorldModel(
        state_dim=baseline_checkpoint['state_dim'],
        action_dim=baseline_checkpoint['action_dim'],
        hidden_dim=baseline_checkpoint['hidden_dim'],
        num_layers=baseline_checkpoint['num_layers'],
        use_residual=baseline_checkpoint['use_residual'],
    ).to(device)
    baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
    baseline_model.eval()
    
    # Load online finetuned model
    print("Loading online finetuned model...")
    online_checkpoint = torch.load("checkpoints_v2/online_final.pt", map_location=device)
    online_model = WorldModel(
        state_dim=online_checkpoint['state_dim'],
        action_dim=online_checkpoint['action_dim'],
        hidden_dim=online_checkpoint['hidden_dim'],
        num_layers=online_checkpoint['num_layers'],
        use_residual=online_checkpoint['use_residual'],
    ).to(device)
    online_model.load_state_dict(online_checkpoint['model_state_dict'])
    online_model.eval()
    
    # Sample a test case
    np.random.seed(42)
    start = np.array([-1.5, 0.5], dtype=np.float32)
    goal = np.array([1.5, -0.5], dtype=np.float32)
    
    print(f"\nTest case:")
    print(f"  Start: {start}")
    print(f"  Goal: {goal}")
    
    # Plan with baseline
    print("\nPlanning with baseline model...")
    z0_tensor = torch.from_numpy(start).float().to(device)
    z_goal_tensor = torch.from_numpy(goal).float().to(device)
    
    baseline_model.train()
    for param in baseline_model.parameters():
        param.requires_grad = True
    
    baseline_actions = gbp_plan(
        baseline_model,
        z0_tensor,
        z_goal_tensor,
        horizon=100,
        n_steps=300,
        action_max=env.action_max,
    )
    baseline_actions = baseline_actions.cpu().numpy()
    
    baseline_model.eval()
    for param in baseline_model.parameters():
        param.requires_grad = False
    
    # Plan with online model
    print("Planning with online finetuned model...")
    online_model.train()
    for param in online_model.parameters():
        param.requires_grad = True
    
    online_actions = gbp_plan(
        online_model,
        z0_tensor,
        z_goal_tensor,
        horizon=100,
        n_steps=300,
        action_max=env.action_max,
    )
    online_actions = online_actions.cpu().numpy()
    
    online_model.eval()
    for param in online_model.parameters():
        param.requires_grad = False
    
    # Rollout in simulator
    print("Rolling out in simulator...")
    baseline_sim_traj = rollout_sim(env, start, baseline_actions, horizon=100)
    online_sim_traj = rollout_sim(env, start, online_actions, horizon=100)
    
    # Rollout in models (for comparison)
    baseline_model_traj = rollout_model(baseline_model, start, baseline_actions, horizon=100)
    online_model_traj = rollout_model(online_model, start, online_actions, horizon=100)
    
    if isinstance(baseline_model_traj, torch.Tensor):
        baseline_model_traj = baseline_model_traj.detach().cpu().numpy()
    if isinstance(online_model_traj, torch.Tensor):
        online_model_traj = online_model_traj.detach().cpu().numpy()
    
    # Compute distances
    baseline_dist = np.linalg.norm(baseline_sim_traj[-1, :2] - goal)
    online_dist = np.linalg.norm(online_sim_traj[-1, :2] - goal)
    
    print(f"\nResults:")
    print(f"  Baseline final distance: {baseline_dist:.3f}")
    print(f"  Online final distance: {online_dist:.3f}")
    print(f"  Improvement: {(baseline_dist - online_dist) / baseline_dist * 100:.1f}%")
    
    # Create comparison plots
    print("\nCreating visualization...")
    
    # Plot 1: Simulator trajectories side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    plot_trajectory(
        baseline_sim_traj,
        wall_x=env.wall_x,
        door_y_min=env.door_y_min,
        door_y_max=env.door_y_max,
        goal=goal,
        goal_threshold=env.goal_threshold,
        title=f"Baseline (distance: {baseline_dist:.2f})",
        ax=axes[0],
    )
    
    plot_trajectory(
        online_sim_traj,
        wall_x=env.wall_x,
        door_y_min=env.door_y_min,
        door_y_max=env.door_y_max,
        goal=goal,
        goal_threshold=env.goal_threshold,
        title=f"Online Finetuned (distance: {online_dist:.2f})",
        ax=axes[1],
    )
    
    plt.tight_layout()
    plt.savefig("results/demo_comparison.png", dpi=150, bbox_inches='tight')
    print("Saved: results/demo_comparison.png")
    
    # Plot 2: Model predictions vs reality
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Baseline: model prediction
    plot_trajectory(
        baseline_model_traj,
        wall_x=env.wall_x,
        door_y_min=env.door_y_min,
        door_y_max=env.door_y_max,
        goal=goal,
        goal_threshold=env.goal_threshold,
        title="Baseline: Model Prediction",
        ax=axes[0, 0],
    )
    
    # Baseline: simulator reality
    plot_trajectory(
        baseline_sim_traj,
        wall_x=env.wall_x,
        door_y_min=env.door_y_min,
        door_y_max=env.door_y_max,
        goal=goal,
        goal_threshold=env.goal_threshold,
        title="Baseline: Simulator Reality",
        ax=axes[0, 1],
    )
    
    # Online: model prediction
    plot_trajectory(
        online_model_traj,
        wall_x=env.wall_x,
        door_y_min=env.door_y_min,
        door_y_max=env.door_y_max,
        goal=goal,
        goal_threshold=env.goal_threshold,
        title="Online: Model Prediction",
        ax=axes[1, 0],
    )
    
    # Online: simulator reality
    plot_trajectory(
        online_sim_traj,
        wall_x=env.wall_x,
        door_y_min=env.door_y_min,
        door_y_max=env.door_y_max,
        goal=goal,
        goal_threshold=env.goal_threshold,
        title="Online: Simulator Reality",
        ax=axes[1, 1],
    )
    
    plt.tight_layout()
    plt.savefig("results/demo_model_vs_reality.png", dpi=150, bbox_inches='tight')
    print("Saved: results/demo_model_vs_reality.png")
    
    plt.close('all')
    
    print("\nDemo complete!")


if __name__ == "__main__":
    demo_planning_comparison()

