"""
PointMaze Environment.
Task: Navigate a point mass through a maze to a goal.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.patches as patches
from typing import Optional, Tuple, List

class PointMazeEnv:
    """
    2D point maze navigation.
    State: [x, y, vx, vy]
    Action: [ax, ay]
    """
    
    def __init__(
        self,
        maze_type: str = "u_maze",
        action_max: float = 1.0,
        dt: float = 0.1,
        goal_threshold: float = 0.2,
        bounds: Tuple[float, float, float, float] = (-2.0, 2.0, -2.0, 2.0),
    ):
        self.maze_type = maze_type
        self.action_max = action_max
        self.dt = dt
        self.goal_threshold = goal_threshold
        self.bounds = bounds
        
        self.state_dim = 4
        self.action_dim = 2
        self.agent_radius = 0.1
        
        # Define walls based on maze type
        self.walls = []
        if maze_type == "u_maze":
            # U-shaped maze
            # Center block
            self.walls.append((-1.0, 1.0, -1.0, 0.5)) # x_min, x_max, y_min, y_max
        else:
            # Empty
            pass
            
    def reset(self):
        return self.sample_random_state()
        
    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        
        action = np.clip(action, -self.action_max, self.action_max)
        
        x, y, vx, vy = state
        
        # Update velocity
        vx_new = vx + action[0] * self.dt
        vy_new = vy + action[1] * self.dt
        
        # Friction
        vx_new *= 0.95
        vy_new *= 0.95
        
        # Update position
        x_new = x + vx_new * self.dt
        y_new = y + vy_new * self.dt
        
        # Collision detection (simple AABB)
        for w in self.walls:
            wx_min, wx_max, wy_min, wy_max = w
            # Check overlap (agent as point for simplicity, or small box)
            if (x_new > wx_min - self.agent_radius and x_new < wx_max + self.agent_radius and
                y_new > wy_min - self.agent_radius and y_new < wy_max + self.agent_radius):
                # Collision! Stop.
                x_new = x
                y_new = y
                vx_new = 0.0
                vy_new = 0.0
                break
                
        # Bounds check
        x_new = np.clip(x_new, self.bounds[0], self.bounds[1])
        y_new = np.clip(y_new, self.bounds[2], self.bounds[3])
        
        return np.array([x_new, y_new, vx_new, vy_new], dtype=np.float32)
        
    def is_goal_reached(self, state: np.ndarray, goal: np.ndarray) -> bool:
        dist = np.linalg.norm(state[:2] - goal[:2])
        return dist < self.goal_threshold
        
    def sample_random_state(self) -> np.ndarray:
        # Sample valid state
        while True:
            x = np.random.uniform(self.bounds[0], self.bounds[1])
            y = np.random.uniform(self.bounds[2], self.bounds[3])
            
            # Check collision
            valid = True
            for w in self.walls:
                wx_min, wx_max, wy_min, wy_max = w
                if (x > wx_min - self.agent_radius and x < wx_max + self.agent_radius and
                    y > wy_min - self.agent_radius and y < wy_max + self.agent_radius):
                    valid = False
                    break
            if valid:
                return np.array([x, y, 0.0, 0.0], dtype=np.float32)
                
    def sample_goal(self) -> np.ndarray:
        # Sample valid goal
        s = self.sample_random_state()
        return s[:2]
        
    def render(self, state: np.ndarray, goal: Optional[np.ndarray] = None, resolution: int = 224) -> np.ndarray:
        fig = plt.figure(figsize=(4, 4), dpi=resolution/4)
        ax = fig.add_subplot(111)
        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[2], self.bounds[3])
        ax.set_aspect('equal')
        ax.axis('off')
        fig.subplots_adjust(0, 0, 1, 1)
        
        # Walls
        for w in self.walls:
            wx_min, wx_max, wy_min, wy_max = w
            rect = patches.Rectangle(
                (wx_min, wy_min),
                wx_max - wx_min,
                wy_max - wy_min,
                color='black'
            )
            ax.add_patch(rect)
            
        # Goal
        if goal is not None:
            g = patches.Circle(goal, radius=self.goal_threshold, color='green', alpha=0.5)
            ax.add_patch(g)
            
        # Agent
        a = patches.Circle(state[:2], radius=self.agent_radius, color='blue')
        ax.add_patch(a)
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        # Convert to numpy array (modern matplotlib API)
        buf = np.asarray(canvas.buffer_rgba())
        image = buf[:, :, :3]  # Take RGB, drop alpha
        plt.close(fig)
        
        return image

