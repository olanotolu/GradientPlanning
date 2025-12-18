"""
Robot Interface Abstraction.
Defines standard interface for interacting with robots (simulated or real).
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any

class Robot(ABC):
    """Abstract base class for robots."""
    
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """
        Reset robot to initial state.
        Returns:
            observation dictionary (e.g. {'image': ..., 'state': ...})
        """
        pass
        
    @abstractmethod
    def get_observation(self) -> Dict[str, Any]:
        """
        Get current observation.
        Returns:
            observation dictionary
        """
        pass
        
    @abstractmethod
    def execute_action(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Execute action on robot.
        Args:
            action: action vector
        Returns:
            observation dictionary after action
        """
        pass
        
    @abstractmethod
    def get_state(self) -> np.ndarray:
        """
        Get ground truth state (if available).
        Returns:
            state vector
        """
        pass

