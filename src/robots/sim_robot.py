"""
Simulation Robot Wrapper.
Wraps environment classes to Robot interface.
"""

import numpy as np
from typing import Dict, Any, Optional
from .base import Robot

class SimRobot(Robot):
    """Wrapper for simulated environments."""
    
    def __init__(self, env, resolution: int = 224):
        self.env = env
        self.resolution = resolution
        self.current_state = None
        self.goal = None # Some envs need goal for rendering?
        
    def reset(self) -> Dict[str, Any]:
        self.current_state = self.env.sample_random_state()
        # Goal is needed for rendering in some envs (e.g. WallDoor)
        # But Robot interface implies general control.
        # Ideally env has internal goal or we set it.
        # For WallDoor, render takes goal.
        # Let's sample a dummy goal for visualization if needed
        if hasattr(self.env, 'sample_goal'):
            self.goal = self.env.sample_goal()
            
        return self.get_observation()
        
    def get_observation(self) -> Dict[str, Any]:
        if self.current_state is None:
            self.reset()
            
        obs = {}
        obs['state'] = self.current_state.copy()
        
        # Render image
        if hasattr(self.env, 'render'):
            # Some envs need goal for render
            # Try passing goal if available
            try:
                img = self.env.render(self.current_state, goal=self.goal, resolution=self.resolution)
            except TypeError:
                img = self.env.render(self.current_state, resolution=self.resolution)
            except:
                img = self.env.render(self.current_state) # Fallback
                
            obs['image'] = img
            
        return obs
        
    def execute_action(self, action: np.ndarray) -> Dict[str, Any]:
        if self.current_state is None:
            self.reset()
            
        next_state = self.env.step(self.current_state, action)
        self.current_state = next_state
        
        return self.get_observation()
        
    def get_state(self) -> np.ndarray:
        return self.current_state

