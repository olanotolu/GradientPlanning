#!/usr/bin/env python3
"""
Create comparison plot showing baseline GBP fails, CEM succeeds, finetuned improves.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.models.world_model import WorldModel
from src.planners.gbp import plan as gbp_plan
from src.planners.cem import plan as cem_plan
from src.envs.wall_door import WallDoorEnv
from src.utils.rollout import rollout_sim
from src.utils.viz import plot_trajectory
from src.utils.metrics import planning_success

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizon = 200
    goal_threshold = 1.0
    env = WallDoorEnv(use_velocity=False, action_max=0.25, goal_threshold=goal_threshold)
    
    # Test case
    np.random.seed(42)
    start = np.array([-1.5, 0.5], dtype=np.float32)
    goal = np.array([1.5, -0.5], dtype=np.float32)
    
    # Load models
    baseline_ckpt = torch.load("checkpoints/baseline_best.pt", map_location=device)
    baseline_model = WorldModel(**{k: baseline_ckpt[k] for k in 
        ['state_dim', 'action_dim', 'hidden_dim', 'num_layers', 'use_residual']}).to(device)
    baseline_model.load_state_dict(baseline_ckpt['model_state_dict'])
    
    online_paths = ["checkpoints/online_final.pt", "checkpoints_v2/online_final.pt"]
    has_online = False
    for online_path in online_paths:
        try:
            online_ckpt = torch.load(online_path, map_location=device)
            online_model = WorldModel(**{k: online_ckpt[k] for k in 
                ['state_dim', 'action_dim', 'hidden_dim', 'num_layers', 'use_residual']}).to(device)
            online_model.load_state_dict(online_ckpt['model_state_dict'])
            has_online = True
            break
        except:
            continue
    
    if not has_online:
        online_model = baseline_model
    
    z0 = torch.from_numpy(start).float().to(device)
    z_goal = torch.from_numpy(goal).float().to(device)
    
    # Plan with each method
    baseline_model.train()
    for p in baseline_model.parameters():
        p.requires_grad = True
    baseline_actions = gbp_plan(baseline_model, z0, z_goal, horizon=horizon, 
                               n_steps=300, action_max=env.action_max,
                               intermediate_weight=0.1).cpu().numpy()
    baseline_model.eval()
    for p in baseline_model.parameters():
        p.requires_grad = False
    
    if has_online:
        online_model.train()
        for p in online_model.parameters():
            p.requires_grad = True
        online_actions = gbp_plan(online_model, z0, z_goal, horizon=horizon,
                                 n_steps=300, action_max=env.action_max,
                                 intermediate_weight=0.1).cpu().numpy()
        online_model.eval()
        for p in online_model.parameters():
            p.requires_grad = False
    else:
        online_actions = baseline_actions
    
    cem_actions = cem_plan(baseline_model, z0, z_goal, horizon=horizon,
                          action_max=env.action_max, n_iterations=20, n_samples=200).cpu().numpy()
    
    # Rollout
    baseline_sim = rollout_sim(env, start, baseline_actions, horizon=horizon)
    online_sim = rollout_sim(env, start, online_actions, horizon=horizon) if has_online else baseline_sim
    cem_sim = rollout_sim(env, start, cem_actions, horizon=horizon)
    
    # Check success
    baseline_success = planning_success(baseline_sim[-1], goal, goal_threshold)
    online_success = planning_success(online_sim[-1], goal, goal_threshold) if has_online else False
    cem_success = planning_success(cem_sim[-1], goal, goal_threshold)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    plot_trajectory(baseline_sim, wall_x=0, door_y_min=-0.5, door_y_max=0.5,
                   goal=goal, goal_threshold=goal_threshold,
                   title=f"Baseline GBP ({'✓' if baseline_success else '✗'})", ax=axes[0])
    
    if has_online:
        plot_trajectory(online_sim, wall_x=0, door_y_min=-0.5, door_y_max=0.5,
                       goal=goal, goal_threshold=goal_threshold,
                       title=f"Online Finetuned GBP ({'✓' if online_success else '✗'})", ax=axes[1])
    else:
        axes[1].text(0.5, 0.5, "Online model not found", ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title("Online Finetuned GBP")
    
    plot_trajectory(cem_sim, wall_x=0, door_y_min=-0.5, door_y_max=0.5,
                   goal=goal, goal_threshold=goal_threshold,
                   title=f"CEM ({'✓' if cem_success else '✗'})", ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('results/method_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved results/method_comparison.png")

if __name__ == '__main__':
    main()
