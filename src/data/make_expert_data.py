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


from src.envs.pusht import PushTEnv
from src.envs.pointmaze import PointMazeEnv
from typing import Optional


class WallExpert:
    """Expert for Wall-Door task."""
    def __init__(self, env, kp=2.0):
        self.env = env
        self.kp = kp
        self.door_center = np.array([0.0, 0.0])
        
    def get_action(self, state, goal):
        pos = state[:2]
        if pos[0] < 0:
            waypoint = self.door_center
        else:
            waypoint = goal
            
        direction = waypoint - pos
        distance = np.linalg.norm(direction)
        
        if distance < 0.05:
            direction = goal - pos
            distance = np.linalg.norm(direction)
            
        if distance > 0.01:
            direction = direction / distance
            action = self.kp * direction * self.env.dt
            action = np.clip(action, -self.env.action_max, self.env.action_max)
        else:
            action = np.zeros(2, dtype=np.float32)
        return action

class PushTExpert:
    """Expert for PushT task."""
    def __init__(self, env, kp=1.0):
        self.env = env
        self.kp = kp
        
    def get_action(self, state, goal):
        # Heuristic: Go behind block relative to goal, then push
        agent_pos = state[:2]
        block_pos = state[2:4]
        
        # Vector from block to goal
        push_vec = goal[:2] - block_pos
        push_dist = np.linalg.norm(push_vec)
        
        if push_dist < 0.05:
            return np.zeros(2)
            
        push_dir = push_vec / push_dist
        
        # Target position for agent: behind block
        offset = self.env.agent_radius + self.env.block_size/2 + 0.05
        target_agent_pos = block_pos - push_dir * offset
        
        agent_to_target = target_agent_pos - agent_pos
        dist_to_target = np.linalg.norm(agent_to_target)
        
        if dist_to_target > 0.1:
            action = self.kp * agent_to_target
        else:
            action = self.kp * push_vec
            
        action = np.clip(action, -self.env.action_max, self.env.action_max)
        return action

class PointMazeExpert:
    """Expert for PointMaze."""
    def __init__(self, env, kp=2.0):
        self.env = env
        self.kp = kp
        
    def get_action(self, state, goal):
        pos = state[:2]
        
        if self.env.maze_type == "u_maze":
            if pos[1] < 1.2 and pos[0] < 0:
                waypoint = np.array([-1.5, 1.5])
            elif pos[1] >= 1.0 and pos[0] < 1.2:
                waypoint = np.array([1.5, 1.5])
            else:
                waypoint = goal
        else:
            waypoint = goal
            
        direction = waypoint - pos
        distance = np.linalg.norm(direction)
        if distance > 0.01:
            direction = direction / distance
            action = self.kp * direction * self.env.dt
            action = np.clip(action, -self.env.action_max, self.env.action_max)
        else:
            action = np.zeros(2)
        return action


def generate_expert_trajectory(
    env,
    expert,
    start: np.ndarray,
    goal: np.ndarray,
    max_steps: int = 100,
    save_images: bool = False,
    image_resolution: int = 224,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], Optional[List[np.ndarray]], Optional[List[np.ndarray]], List[bool]]:
    """
    Generate a single expert trajectory.
    """
    states = []
    actions = []
    next_states = []
    images = [] if save_images else None
    next_images = [] if save_images else None
    dones = []
    
    state = start.copy()
    
    if save_images:
        img = env.render(state, goal=goal, resolution=image_resolution)
    
    for _ in range(max_steps):
        # Check if goal reached
        done = False
        if env.is_goal_reached(state, goal):
            done = True
            break
        
        # Get expert action
        action = expert.get_action(state, goal)
        
        # Step environment
        next_state = env.step(state, action)
        
        # Check if done after step
        done = env.is_goal_reached(next_state, goal)
        
        if save_images:
            next_img = env.render(next_state, goal=goal, resolution=image_resolution)
            images.append(img)
            next_images.append(next_img)
            img = next_img
        
        states.append(state.copy())
        actions.append(action.copy())
        next_states.append(next_state.copy())
        dones.append(done)
        
        state = next_state
        if done:
            break
    
    return states, actions, next_states, images, next_images, dones


def generate_dataset(
    env,
    task: str,
    n_trajectories: int = 1000,
    max_steps: int = 100,
    seed: int = 42,
    save_images: bool = False,
    image_resolution: int = 224,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """
    Generate dataset of expert trajectories.
    """
    np.random.seed(seed)
    
    if task == "wall":
        expert = WallExpert(env)
    elif task == "pusht":
        expert = PushTExpert(env)
    elif task == "pointmaze":
        expert = PointMazeExpert(env)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    all_states = []
    all_actions = []
    all_next_states = []
    all_images = [] if save_images else None
    all_next_images = [] if save_images else None
    all_dones = []
    
    print(f"Generating {n_trajectories} expert trajectories for {task}...")
    if save_images:
        print(f"Rendering images at {image_resolution}x{image_resolution}")
    
    for i in range(n_trajectories):
        # Sample random start and goal based on task
        if task == "wall":
            start = env.sample_random_state()
            goal = env.sample_goal()
            if np.random.rand() > 0.5:
                start[0] = np.random.uniform(env.bounds[0], -0.5)
                goal[0] = np.random.uniform(0.5, env.bounds[1])
            else:
                start[0] = np.random.uniform(0.5, env.bounds[1])
                goal[0] = np.random.uniform(env.bounds[0], -0.5)
        elif task == "pointmaze":
            start = np.array([-1.5, -1.5, 0.0, 0.0]) # Bottom left
            goal = np.array([1.5, -1.5]) # Bottom right
            start[:2] += np.random.uniform(-0.1, 0.1, 2)
            goal += np.random.uniform(-0.1, 0.1, 2)
        else: # pusht
            start = env.sample_random_state()
            goal = env.sample_goal()
        
        # Generate trajectory
        states, actions, next_states, imgs, next_imgs, dones = generate_expert_trajectory(
            env, expert, start, goal, max_steps, save_images, image_resolution
        )
        
        all_states.extend(states)
        all_actions.extend(actions)
        all_next_states.extend(next_states)
        all_dones.extend(dones)
        
        if save_images:
            all_images.extend(imgs)
            all_next_images.extend(next_imgs)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{n_trajectories} trajectories ({len(all_states)} transitions)")
    
    # Convert to arrays
    states_array = np.array(all_states, dtype=np.float32)
    actions_array = np.array(all_actions, dtype=np.float32)
    next_states_array = np.array(all_next_states, dtype=np.float32)
    dones_array = np.array(all_dones, dtype=bool)
    
    images_array = None
    next_images_array = None
    
    if save_images:
        images_array = np.array(all_images, dtype=np.uint8)
        next_images_array = np.array(all_next_images, dtype=np.uint8)
    
    print(f"\nGenerated dataset:")
    print(f"  Total transitions: {len(all_states)}")
    print(f"  States shape: {states_array.shape}")
    print(f"  Actions shape: {actions_array.shape}")
    
    return states_array, actions_array, next_states_array, images_array, next_images_array, dones_array


def main():
    parser = argparse.ArgumentParser(description="Generate expert trajectory dataset")
    parser.add_argument("--output", type=str, default="data/expert_data.npz", help="Output file path")
    parser.add_argument("--n_trajectories", type=int, default=1000, help="Number of trajectories")
    parser.add_argument("--max_steps", type=int, default=100, help="Max steps per trajectory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--task", type=str, default="wall", choices=["wall", "pusht", "pointmaze"], help="Task")
    parser.add_argument("--save_images", action="store_true", help="Save rendered images")
    parser.add_argument("--image_resolution", type=int, default=224, help="Image resolution")
    
    args = parser.parse_args()
    
    # Create environment
    if args.task == "wall":
        env = WallDoorEnv(use_velocity=False, action_max=0.25)
    elif args.task == "pusht":
        env = PushTEnv(action_max=0.1)
    elif args.task == "pointmaze":
        env = PointMazeEnv(action_max=1.0)
    
    # Generate dataset
    states, actions, next_states, images, next_images, dones = generate_dataset(
        env,
        task=args.task,
        n_trajectories=args.n_trajectories,
        max_steps=args.max_steps,
        seed=args.seed,
        save_images=args.save_images,
        image_resolution=args.image_resolution,
    )
    
    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'states': states,
        'actions': actions,
        'next_states': next_states,
        'dones': dones,
        'task': args.task,
    }
    
    if args.save_images:
        save_dict['images'] = images
        save_dict['next_images'] = next_images
    
    np.savez(
        output_path,
        **save_dict
    )
    
    print(f"\nSaved dataset to {output_path}")


if __name__ == "__main__":
    main()

