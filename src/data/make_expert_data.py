"""
Generate expert trajectories for training.

Simple two-waypoint controller: go to door center, then to goal.
"""

import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.envs.wall_door import WallDoorEnv


class ExpertPolicy:
    """Two-waypoint expert policy: go to door, then to goal."""
    
    def __init__(self, env: WallDoorEnv, kp: float = 2.0):
        """
        Args:
            env: Environment instance
            kp: Proportional gain for waypoint controller
        """
        self.env = env
        self.kp = kp
        self.door_center = np.array([0.0, 0.0])
    
    def get_action(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """
        Compute expert action using two-waypoint controller.
        
        Args:
            state: Current state [x, y] or [x, y, vx, vy]
            goal: Goal position [x_goal, y_goal]
            
        Returns:
            Action [dx, dy]
        """
        pos = state[:2]  # Extract position
        
        # Check if we're on the left side of the wall
        if pos[0] < 0:
            # First waypoint: door center
            waypoint = self.door_center
        else:
            # Second waypoint: goal
            waypoint = goal
        
        # Proportional controller: action = kp * (waypoint - pos)
        direction = waypoint - pos
        distance = np.linalg.norm(direction)
        
        if distance < 0.05:
            # Close enough, move directly to goal
            direction = goal - pos
            distance = np.linalg.norm(direction)
        
        if distance > 0.01:
            direction = direction / distance  # Normalize
            action = self.kp * direction * self.env.dt
            # Clip to action bounds
            action = np.clip(action, -self.env.action_max, self.env.action_max)
        else:
            action = np.zeros(2, dtype=np.float32)
        
        return action


def generate_expert_trajectory(
    env: WallDoorEnv,
    expert: ExpertPolicy,
    start: np.ndarray,
    goal: np.ndarray,
    max_steps: int = 100,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Generate a single expert trajectory.
    
    Args:
        env: Environment
        expert: Expert policy
        start: Starting state
        goal: Goal position
        max_steps: Maximum trajectory length
        
    Returns:
        (states, actions, next_states) lists
    """
    states = []
    actions = []
    next_states = []
    
    state = start.copy()
    states.append(state.copy())
    
    for _ in range(max_steps):
        # Check if goal reached
        if env.is_goal_reached(state, goal):
            break
        
        # Get expert action
        action = expert.get_action(state, goal)
        
        # Step environment
        next_state = env.step(state, action)
        
        states.append(state.copy())
        actions.append(action.copy())
        next_states.append(next_state.copy())
        
        state = next_state
    
    return states[:-1], actions, next_states  # Last state has no next_state


def generate_dataset(
    env: WallDoorEnv,
    n_trajectories: int = 1000,
    max_steps: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate dataset of expert trajectories.
    
    Args:
        env: Environment
        n_trajectories: Number of trajectories to generate
        max_steps: Maximum steps per trajectory
        seed: Random seed
        
    Returns:
        (states, actions, next_states) arrays
    """
    np.random.seed(seed)
    expert = ExpertPolicy(env)
    
    all_states = []
    all_actions = []
    all_next_states = []
    
    print(f"Generating {n_trajectories} expert trajectories...")
    
    for i in range(n_trajectories):
        # Sample random start and goal
        start = env.sample_random_state()
        goal = env.sample_goal()
        
        # Ensure start and goal are on opposite sides of wall for interesting trajectories
        if np.random.rand() > 0.5:
            # Start left, goal right
            start[0] = np.random.uniform(env.bounds[0], -0.5)
            goal[0] = np.random.uniform(0.5, env.bounds[1])
        else:
            # Start right, goal left
            start[0] = np.random.uniform(0.5, env.bounds[1])
            goal[0] = np.random.uniform(env.bounds[0], -0.5)
        
        # Generate trajectory
        states, actions, next_states = generate_expert_trajectory(
            env, expert, start, goal, max_steps
        )
        
        all_states.extend(states)
        all_actions.extend(actions)
        all_next_states.extend(next_states)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_trajectories} trajectories ({len(all_states)} transitions)")
    
    # Convert to arrays
    states_array = np.array(all_states, dtype=np.float32)
    actions_array = np.array(all_actions, dtype=np.float32)
    next_states_array = np.array(all_next_states, dtype=np.float32)
    
    print(f"\nGenerated dataset:")
    print(f"  Total transitions: {len(all_states)}")
    print(f"  States shape: {states_array.shape}")
    print(f"  Actions shape: {actions_array.shape}")
    print(f"  Next states shape: {next_states_array.shape}")
    
    return states_array, actions_array, next_states_array


def main():
    parser = argparse.ArgumentParser(description="Generate expert trajectory dataset")
    parser.add_argument("--output", type=str, default="data/expert_data.npz", help="Output file path")
    parser.add_argument("--n_trajectories", type=int, default=1000, help="Number of trajectories")
    parser.add_argument("--max_steps", type=int, default=100, help="Max steps per trajectory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--door_width", type=float, default=0.5, help="Door width")
    parser.add_argument("--action_max", type=float, default=0.1, help="Max action magnitude")
    
    args = parser.parse_args()
    
    # Create environment
    env = WallDoorEnv(
        door_width=args.door_width,
        action_max=args.action_max,
        use_velocity=False,
    )
    
    # Generate dataset
    states, actions, next_states = generate_dataset(
        env,
        n_trajectories=args.n_trajectories,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    
    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_path,
        states=states,
        actions=actions,
        next_states=next_states,
    )
    
    print(f"\nSaved dataset to {output_path}")


if __name__ == "__main__":
    main()

