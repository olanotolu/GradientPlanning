#!/usr/bin/env python3
"""Test improved planning strategies."""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.models.world_model import WorldModel
from src.envs.wall_door import WallDoorEnv
from src.planners.gbp import plan as gbp_plan
from src.planners.gbp_improved import plan_with_expert_init, plan_mpc_style
from src.utils.rollout import rollout_sim
from src.utils.metrics import distance_to_goal, planning_success


def test_improved_planners():
    """Test different planning strategies."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = WallDoorEnv(use_velocity=False, action_max=0.25, goal_threshold=0.5)
    
    # Load online model
    checkpoint = torch.load("checkpoints_v2/online_final.pt", map_location=device)
    model = WorldModel(
        state_dim=checkpoint['state_dim'],
        action_dim=checkpoint['action_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        use_residual=checkpoint['use_residual'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    np.random.seed(42)
    n_episodes = 10
    
    results = {
        'baseline_gbp': [],
        'expert_init': [],
        'mpc_style': [],
    }
    
    print("Testing improved planning strategies...\n")
    
    for episode in range(n_episodes):
        # Sample start and goal
        start = env.sample_random_state()
        goal = env.sample_goal()
        
        if np.random.rand() > 0.5:
            start[0] = np.random.uniform(env.bounds[0], -0.5)
            goal[0] = np.random.uniform(0.5, env.bounds[1])
        else:
            start[0] = np.random.uniform(0.5, env.bounds[1])
            goal[0] = np.random.uniform(env.bounds[0], -0.5)
        
        z0 = torch.from_numpy(start).float().to(device)
        z_goal = torch.from_numpy(goal).float().to(device)
        
        print(f"Episode {episode + 1}/{n_episodes}: start={start[:2]}, goal={goal}")
        
        # 1. Baseline GBP
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        
        baseline_actions = gbp_plan(
            model, z0, z_goal, horizon=150, n_steps=500, action_max=env.action_max
        )
        baseline_actions = baseline_actions.cpu().numpy()
        
        baseline_traj = rollout_sim(env, start, baseline_actions, horizon=len(baseline_actions))
        baseline_dist = distance_to_goal(baseline_traj[-1], goal)
        results['baseline_gbp'].append(baseline_dist)
        
        # 2. Expert initialization
        expert_actions = plan_with_expert_init(
            model, env, z0, z_goal, horizon=150, n_steps=500, action_max=env.action_max
        )
        expert_actions = expert_actions.cpu().numpy()
        
        expert_traj = rollout_sim(env, start, expert_actions, horizon=len(expert_actions))
        expert_dist = distance_to_goal(expert_traj[-1], goal)
        results['expert_init'].append(expert_dist)
        
        # 3. MPC-style
        mpc_actions = plan_mpc_style(
            model, env, z0, z_goal, horizon=50, n_steps=200, 
            replan_every=10, action_max=env.action_max
        )
        mpc_actions = mpc_actions.cpu().numpy()
        
        mpc_traj = rollout_sim(env, start, mpc_actions, horizon=len(mpc_actions))
        mpc_dist = distance_to_goal(mpc_traj[-1], goal)
        results['mpc_style'].append(mpc_dist)
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        print(f"  Baseline: {baseline_dist:.3f}, Expert Init: {expert_dist:.3f}, MPC: {mpc_dist:.3f}\n")
    
    # Summary
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for method, dists in results.items():
        avg_dist = np.mean(dists)
        std_dist = np.std(dists)
        success_rate = np.mean([d < env.goal_threshold for d in dists]) * 100
        print(f"{method:20s}: avg={avg_dist:.3f} ± {std_dist:.3f}, success={success_rate:.1f}%")


if __name__ == "__main__":
    test_improved_planners()

