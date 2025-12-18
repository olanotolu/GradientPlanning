"""
PushT Environment.
Task: Push a T-shaped block to a target T-shaped goal.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.patches as patches
from typing import Optional, Tuple

class PushTEnv:
    """
    2D block pushing task.
    State: [agent_x, agent_y, block_x, block_y, block_theta]
    Action: [dx, dy]
    """
    
    def __init__(
        self,
        action_max: float = 0.1,
        dt: float = 0.1,
        goal_threshold: float = 0.1,
        bounds: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
    ):
        self.action_max = action_max
        self.dt = dt
        self.goal_threshold = goal_threshold
        self.bounds = bounds
        
        # State: agent (2) + block (3) = 5
        self.state_dim = 5
        self.action_dim = 2
        
        # Block properties
        self.block_size = 0.2
        self.agent_radius = 0.05
        
        # Target pose (fixed or random)
        self.target_pose = np.array([0.5, 0.5, 0.0]) # x, y, theta
        
    def reset(self):
        """Reset environment to random state."""
        return self.sample_random_state()
        
    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Apply action."""
        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        
        # Clip action
        action = np.clip(action, -self.action_max, self.action_max)
        
        # Agent dynamics (kinematic)
        agent_pos = state[:2]
        block_pose = state[2:]
        
        # Move agent
        next_agent_pos = agent_pos + action
        
        # Clip agent to bounds
        next_agent_pos[0] = np.clip(next_agent_pos[0], self.bounds[0], self.bounds[1])
        next_agent_pos[1] = np.clip(next_agent_pos[1], self.bounds[2], self.bounds[3])
        
        # Block interaction (quasi-static pushing)
        # If agent overlaps block, move block
        # Simple circular approximation for collision for now
        # Or just simple distance check
        
        dist = np.linalg.norm(next_agent_pos - block_pose[:2])
        interaction_dist = self.agent_radius + self.block_size/2
        
        next_block_pose = block_pose.copy()
        
        if dist < interaction_dist:
            # Collision! Push block
            push_dir = next_agent_pos - block_pose[:2]
            push_dir = push_dir / (np.linalg.norm(push_dir) + 1e-6)
            
            # Move block away from agent
            # Displacement
            overlap = interaction_dist - dist
            # Move block by overlap amount in push direction (simplified)
            # Actually, agent pushes block. Block moves in direction of push.
            # Let's say block moves with agent if pushing.
            # For simplicity:
            displacement = next_agent_pos - agent_pos
            next_block_pose[:2] += displacement
            
            # Rotation? Ignore for "shitty version" unless crucial
            # Let's add some rotation if pushing off-center?
            # Maybe later.
            
        # Clip block to bounds
        next_block_pose[0] = np.clip(next_block_pose[0], self.bounds[0], self.bounds[1])
        next_block_pose[1] = np.clip(next_block_pose[1], self.bounds[2], self.bounds[3])
        
        return np.concatenate([next_agent_pos, next_block_pose])
        
    def is_goal_reached(self, state: np.ndarray, goal: Optional[np.ndarray] = None) -> bool:
        """Check success."""
        # Goal is usually fixed target pose for block
        # goal argument might be target pose if variable
        target = goal if goal is not None else self.target_pose
        
        block_pose = state[2:]
        # Distance in position and angle
        pos_dist = np.linalg.norm(block_pose[:2] - target[:2])
        angle_dist = np.abs(block_pose[2] - target[2])
        # Normalize angle to [-pi, pi]
        angle_dist = (angle_dist + np.pi) % (2 * np.pi) - np.pi
        angle_dist = np.abs(angle_dist)
        
        return pos_dist < self.goal_threshold and angle_dist < 0.5 # Generous angle threshold
        
    def sample_random_state(self) -> np.ndarray:
        """Sample random state."""
        agent_pos = np.random.uniform(self.bounds[0], self.bounds[1], 2)
        block_pos = np.random.uniform(self.bounds[0], self.bounds[1], 2)
        block_theta = np.random.uniform(-np.pi, np.pi)
        return np.concatenate([agent_pos, block_pos, [block_theta]]).astype(np.float32)
        
    def sample_goal(self) -> np.ndarray:
        """Sample random target pose."""
        # Typically fixed in PushT, but can be random
        return self.sample_random_state()[2:] # Just block pose
        
    def render(self, state: np.ndarray, goal: Optional[np.ndarray] = None, resolution: int = 224) -> np.ndarray:
        """Render RGB image."""
        # Setup plot
        fig = plt.figure(figsize=(4, 4), dpi=resolution/4)
        ax = fig.add_subplot(111)
        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[2], self.bounds[3])
        ax.set_aspect('equal')
        ax.axis('off')
        fig.subplots_adjust(0, 0, 1, 1)
        
        agent_pos = state[:2]
        block_pose = state[2:]
        
        # Draw target (ghost block)
        target = goal if goal is not None else self.target_pose
        target_patch = patches.Rectangle(
            (target[0] - self.block_size/2, target[1] - self.block_size/2),
            self.block_size, self.block_size,
            angle=np.degrees(target[2]),
            color='green', alpha=0.3
        )
        ax.add_patch(target_patch)
        
        # Draw block
        block_patch = patches.Rectangle(
            (block_pose[0] - self.block_size/2, block_pose[1] - self.block_size/2),
            self.block_size, self.block_size,
            angle=np.degrees(block_pose[2]),
            color='gray'
        )
        ax.add_patch(block_patch)
        
        # Draw agent
        agent_patch = patches.Circle(agent_pos, radius=self.agent_radius, color='blue')
        ax.add_patch(agent_patch)
        
        # Render
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        # Convert to numpy array (modern matplotlib API)
        buf = np.asarray(canvas.buffer_rgba())
        image = buf[:, :, :3]  # Take RGB, drop alpha
        plt.close(fig)
        
        return image

