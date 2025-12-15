"""
Visualization utilities for trajectories.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple


def plot_trajectory(
    z_sequence: np.ndarray,
    wall_x: float = 0.0,
    door_y_min: float = -0.5,
    door_y_max: float = 0.5,
    goal: Optional[np.ndarray] = None,
    goal_threshold: float = 0.1,
    title: str = "Trajectory",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot 2D trajectory with wall, door, and goal.
    
    Args:
        z_sequence: State sequence [horizon+1, state_dim]
        wall_x: x-coordinate of wall
        door_y_min: Bottom of door segment
        door_y_max: Top of door segment
        goal: Goal position [x, y]
        goal_threshold: Radius for goal circle
        title: Plot title
        ax: Optional matplotlib axes (creates new if None)
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Extract positions
    positions = z_sequence[:, :2]  # [horizon+1, 2]
    
    # Plot wall
    y_wall = np.linspace(door_y_min - 1, door_y_max + 1, 100)
    x_wall = np.ones_like(y_wall) * wall_x
    ax.plot(x_wall, y_wall, 'k-', linewidth=3, label='Wall')
    
    # Highlight door opening
    y_door = np.linspace(door_y_min, door_y_max, 50)
    x_door = np.ones_like(y_door) * wall_x
    ax.plot(x_door, y_door, 'g-', linewidth=5, alpha=0.5, label='Door')
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
    ax.scatter(positions[0, 0], positions[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, marker='x', label='End', zorder=5)
    
    # Plot goal
    if goal is not None:
        goal_circle = plt.Circle(
            (goal[0], goal[1]),
            goal_threshold,
            color='orange',
            fill=False,
            linewidth=2,
            linestyle='--',
            label='Goal',
        )
        ax.add_patch(goal_circle)
        ax.scatter(goal[0], goal[1], c='orange', s=100, marker='*', zorder=5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')
    
    return ax


def compare_trajectories(
    trajectories: List[Tuple[np.ndarray, str]],
    wall_x: float = 0.0,
    door_y_min: float = -0.5,
    door_y_max: float = 0.5,
    goal: Optional[np.ndarray] = None,
    goal_threshold: float = 0.1,
    title: str = "Trajectory Comparison",
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Plot multiple trajectories side-by-side.
    
    Args:
        trajectories: List of (z_sequence, label) tuples
        wall_x: x-coordinate of wall
        door_y_min: Bottom of door segment
        door_y_max: Top of door segment
        goal: Goal position [x, y]
        goal_threshold: Radius for goal circle
        title: Figure title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_trajectories = len(trajectories)
    n_cols = min(2, n_trajectories)
    n_rows = (n_trajectories + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_trajectories == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle(title, fontsize=14)
    
    for idx, (z_sequence, label) in enumerate(trajectories):
        ax = axes[idx]
        plot_trajectory(
            z_sequence,
            wall_x=wall_x,
            door_y_min=door_y_min,
            door_y_max=door_y_max,
            goal=goal,
            goal_threshold=goal_threshold,
            title=label,
            ax=ax,
        )
    
    # Hide unused subplots
    for idx in range(n_trajectories, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig

