"""
Improved Gradient-Based Planner with expert initialization and waypoint planning.
"""

import torch
import torch.optim as optim
from typing import Optional
import numpy as np
from src.envs.wall_door import WallDoorEnv
from src.data.make_expert_data import ExpertPolicy


def plan_with_expert_init(
    world_model: torch.nn.Module,
    env: WallDoorEnv,
    z0: torch.Tensor,
    z_goal: torch.Tensor,
    horizon: int = 25,
    n_steps: int = 300,
    lr: float = 1e-3,
    action_max: float = 0.1,
    use_waypoints: bool = True,
) -> torch.Tensor:
    """
    Plan with expert policy initialization and optional waypoint planning.
    
    Args:
        world_model: World model
        env: Environment
        z0: Initial state
        z_goal: Goal state
        horizon: Planning horizon
        n_steps: Optimization steps
        lr: Learning rate
        action_max: Max action magnitude
        use_waypoints: If True, plan to door first, then goal
        
    Returns:
        Optimized action sequence
    """
    device = next(world_model.parameters()).device
    
    if z0.dim() == 1:
        z0 = z0.unsqueeze(0).to(device)
        z_goal = z_goal.unsqueeze(0).to(device)
        squeeze_output = True
    else:
        z0 = z0.to(device)
        z_goal = z_goal.to(device)
        squeeze_output = False
    
    state_dim = z0.shape[-1]
    action_dim = 2
    
    # Initialize with expert policy
    expert = ExpertPolicy(env, kp=2.0)
    z0_np = z0[0].detach().cpu().numpy()
    z_goal_np = z_goal[0].detach().cpu().numpy()
    
    if use_waypoints:
        # Plan to door first, then goal
        door_center = np.array([0.0, 0.0])
        if z0_np[0] < 0:
            # Start left of wall - plan to door first
            waypoint = door_center
            horizon1 = horizon // 2
            horizon2 = horizon - horizon1
        else:
            # Start right of wall - go directly to goal
            waypoint = z_goal_np
            horizon1 = horizon
            horizon2 = 0
    else:
        waypoint = z_goal_np
        horizon1 = horizon
        horizon2 = 0
    
    # Generate expert actions for initialization
    init_actions = []
    z = z0_np.copy()
    for _ in range(horizon1):
        a = expert.get_action(z, waypoint)
        init_actions.append(a)
        z = env.step(z, a)
    
    if horizon2 > 0:
        for _ in range(horizon2):
            a = expert.get_action(z, z_goal_np)
            init_actions.append(a)
            z = env.step(z, a)
    
    init_actions = np.array(init_actions, dtype=np.float32)
    
    # Convert to unbounded parameters
    u = torch.atanh(torch.clamp(
        torch.from_numpy(init_actions).float().to(device) / action_max,
        -0.99, 0.99
    )).unsqueeze(0)
    
    u.requires_grad_(True)
    optimizer = optim.Adam([u], lr=lr)
    
    # Optimization loop
    for step in range(n_steps):
        optimizer.zero_grad()
        
        a_sequence = action_max * torch.tanh(u)
        z_sequence = world_model.rollout(z0, a_sequence, horizon=horizon)
        
        z_final = z_sequence[:, -1, :]
        loss = torch.mean((z_final - z_goal) ** 2)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_([u], max_norm=1.0)
        optimizer.step()
    
    a_sequence = action_max * torch.tanh(u)
    
    if squeeze_output:
        a_sequence = a_sequence.squeeze(0)
    
    return a_sequence.detach()


def plan_mpc_style(
    world_model: torch.nn.Module,
    env: WallDoorEnv,
    z0: torch.Tensor,
    z_goal: torch.Tensor,
    horizon: int = 25,
    n_steps: int = 300,
    replan_every: int = 10,
    lr: float = 1e-3,
    action_max: float = 0.1,
) -> torch.Tensor:
    """
    MPC-style planning: replan every N steps.
    
    Args:
        world_model: World model
        env: Environment
        z0: Initial state
        z_goal: Goal state
        horizon: Planning horizon per replan
        n_steps: Optimization steps per replan
        replan_every: Replan every N steps
        lr: Learning rate
        action_max: Max action magnitude
        
    Returns:
        Full action sequence
    """
    device = next(world_model.parameters()).device
    
    if z0.dim() == 1:
        z0 = z0.unsqueeze(0).to(device)
        z_goal = z_goal.unsqueeze(0).to(device)
        squeeze_output = True
    else:
        z0 = z0.to(device)
        z_goal = z_goal.to(device)
        squeeze_output = False
    
    all_actions = []
    current_state = z0[0].clone()
    
    # Total steps needed
    total_steps = int(np.ceil(4.0 / (action_max * 0.1)))  # Rough estimate for 4 unit distance
    total_steps = min(total_steps, 200)  # Cap at 200
    
    step = 0
    while step < total_steps:
        # Plan for next horizon
        world_model.train()
        for param in world_model.parameters():
            param.requires_grad = True
        
        u = torch.randn(1, horizon, 2, device=device) * 0.1
        u.requires_grad_(True)
        optimizer = optim.Adam([u], lr=lr)
        
        for opt_step in range(n_steps):
            optimizer.zero_grad()
            a_seq = action_max * torch.tanh(u)
            z_seq = world_model.rollout(current_state.unsqueeze(0), a_seq, horizon=horizon)
            z_final = z_seq[:, -1, :]
            loss = torch.mean((z_final - z_goal) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([u], max_norm=1.0)
            optimizer.step()
        
        world_model.eval()
        for param in world_model.parameters():
            param.requires_grad = False
        
        # Get actions for next replan_every steps
        a_seq = action_max * torch.tanh(u)
        actions_to_use = a_seq[0, :replan_every, :].detach().cpu().numpy()
        all_actions.extend(actions_to_use)
        
        # Update current state using simulator
        current_state_np = current_state.detach().cpu().numpy()
        for a in actions_to_use:
            current_state_np = env.step(current_state_np, a)
        current_state = torch.from_numpy(current_state_np).float().to(device)
        
        step += replan_every
        
        # Check if close to goal
        dist = torch.norm(current_state[:2] - z_goal[0, :2])
        if dist < 0.3:
            break
    
    all_actions = np.array(all_actions, dtype=np.float32)
    return torch.from_numpy(all_actions).float().to(device)

