"""
2D Wall-Door Navigation Environment

Simple 2D navigation task with a wall at x=0 and a door segment.
Agent must navigate from start to goal, going through the door.
"""

import numpy as np
from typing import Tuple, Optional


class WallDoorEnv:
    """2D navigation environment with wall and door."""
    
    def __init__(
        self,
        door_width: float = 0.5,
        action_max: float = 0.1,
        dt: float = 0.1,
        goal_threshold: float = 0.1,
        bounds: Tuple[float, float, float, float] = (-2.0, 2.0, -2.0, 2.0),
        use_velocity: bool = False,
    ):
        """
        Args:
            door_width: Half-width of door segment (door spans y ∈ [-door_width, door_width])
            action_max: Maximum action magnitude (clips to [-action_max, action_max])
            dt: Time step for Euler integration
            goal_threshold: Distance threshold for goal success
            bounds: (x_min, x_max, y_min, y_max) bounds for valid states
            use_velocity: If True, state includes velocity [x, y, vx, vy], else just [x, y]
        """
        self.door_width = door_width
        self.action_max = action_max
        self.dt = dt
        self.goal_threshold = goal_threshold
        self.bounds = bounds
        self.use_velocity = use_velocity
        
        self.wall_x = 0.0
        self.door_y_min = -door_width
        self.door_y_max = door_width
        
        # State dimension: [x, y] or [x, y, vx, vy]
        self.state_dim = 4 if use_velocity else 2
        self.action_dim = 2
    
    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Apply action and return next state.
        
        Args:
            state: Current state [x, y] or [x, y, vx, vy]
            action: Action [dx, dy] (will be clipped)
            
        Returns:
            Next state after applying action and handling collisions
        """
        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        
        # Clip action
        action = np.clip(action, -self.action_max, self.action_max)
        
        if self.use_velocity:
            # State is [x, y, vx, vy]
            x, y, vx, vy = state
            
            # Update velocity (simple dynamics: action directly affects velocity)
            vx_new = vx + action[0] * self.dt
            vy_new = vy + action[1] * self.dt
            
            # Update position
            x_new = x + vx_new * self.dt
            y_new = y + vy_new * self.dt
            
            next_state = np.array([x_new, y_new, vx_new, vy_new], dtype=np.float32)
        else:
            # State is [x, y], action directly moves position
            x, y = state
            x_new = x + action[0] * self.dt
            y_new = y + action[1] * self.dt
            next_state = np.array([x_new, y_new], dtype=np.float32)
        
        # Handle wall collision
        next_state = self._handle_collision(state, next_state)
        
        return next_state
    
    def _handle_collision(self, state: np.ndarray, next_state: np.ndarray) -> np.ndarray:
        """
        Handle collision with wall. If crossing wall outside door, clamp to wall.
        
        Args:
            state: Previous state
            next_state: Proposed next state
            
        Returns:
            Corrected next state
        """
        x_prev = state[0]
        y_prev = state[1]
        x_next = next_state[0]
        y_next = next_state[1]
        
        # Check if crossing the wall (x=0)
        if (x_prev < self.wall_x and x_next > self.wall_x) or \
           (x_prev > self.wall_x and x_next < self.wall_x):
            # Crossing the wall - check if through door
            if not (self.door_y_min <= y_next <= self.door_y_max):
                # Collision! Clamp to wall boundary
                x_next = self.wall_x
                if self.use_velocity:
                    # Also zero out velocity in x direction
                    next_state = np.array([x_next, y_next, 0.0, next_state[3]], dtype=np.float32)
                else:
                    next_state = np.array([x_next, y_next], dtype=np.float32)
        
        # Clamp to bounds
        x_min, x_max, y_min, y_max = self.bounds
        x_next = np.clip(next_state[0], x_min, x_max)
        y_next = np.clip(next_state[1], y_min, y_max)
        
        if self.use_velocity:
            next_state = np.array([x_next, y_next, next_state[2], next_state[3]], dtype=np.float32)
        else:
            next_state = np.array([x_next, y_next], dtype=np.float32)
        
        return next_state
    
    def is_valid(self, state: np.ndarray) -> bool:
        """
        Check if state is in valid region (within bounds).
        
        Args:
            state: State to check
            
        Returns:
            True if state is valid
        """
        x, y = state[0], state[1]
        x_min, x_max, y_min, y_max = self.bounds
        return (x_min <= x <= x_max) and (y_min <= y <= y_max)
    
    def is_goal_reached(self, state: np.ndarray, goal: np.ndarray) -> bool:
        """
        Check if goal is reached.
        
        Args:
            state: Current state
            goal: Goal state [x_goal, y_goal]
            
        Returns:
            True if distance to goal < threshold
        """
        pos = state[:2]  # Extract position
        distance = np.linalg.norm(pos - goal)
        return distance < self.goal_threshold
    
    def distance_to_goal(self, state: np.ndarray, goal: np.ndarray) -> float:
        """Compute distance from state to goal."""
        pos = state[:2]
        return float(np.linalg.norm(pos - goal))
    
    def sample_random_state(self) -> np.ndarray:
        """Sample a random valid state."""
        x_min, x_max, y_min, y_max = self.bounds
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        
        if self.use_velocity:
            vx = np.random.uniform(-0.1, 0.1)
            vy = np.random.uniform(-0.1, 0.1)
            return np.array([x, y, vx, vy], dtype=np.float32)
        else:
            return np.array([x, y], dtype=np.float32)
    
    def sample_goal(self) -> np.ndarray:
        """Sample a random goal position."""
        x_min, x_max, y_min, y_max = self.bounds
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        return np.array([x, y], dtype=np.float32)

