"""
Rendering utilities for 2D environments.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.patches as patches

class Renderer:
    """
    Renders 2D environments to RGB images.
    """
    
    def __init__(self, resolution: int = 224, bounds: tuple = (-2.0, 2.0, -2.0, 2.0)):
        """
        Args:
            resolution: Output image resolution (width=height)
            bounds: (x_min, x_max, y_min, y_max)
        """
        self.resolution = resolution
        self.bounds = bounds
        self.x_min, self.x_max, self.y_min, self.y_max = bounds
        
        # Setup figure
        self.fig = plt.figure(figsize=(4, 4), dpi=resolution/4)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasAgg(self.fig)
        
    def setup_ax(self):
        """Reset and configure axis."""
        self.ax.clear()
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.set_aspect('equal')
        self.ax.axis('off')  # Hide axis
        self.fig.subplots_adjust(0, 0, 1, 1)  # Remove margins
        
    def render_wall_door(self, 
                        agent_pos: np.ndarray, 
                        wall_x: float, 
                        door_y_min: float, 
                        door_y_max: float,
                        goal_pos: np.ndarray = None):
        """
        Render the Wall-Door environment.
        
        Args:
            agent_pos: [x, y]
            wall_x: x-coordinate of wall
            door_y_min: min y of door
            door_y_max: max y of door
            goal_pos: [x, y] (optional)
            
        Returns:
            rgb_array: (resolution, resolution, 3) uint8 array
        """
        self.setup_ax()
        
        # Background is white by default
        
        # Draw Wall
        # Wall is split into two parts: below door and above door
        
        # Bottom wall segment
        rect_bottom = patches.Rectangle(
            (wall_x - 0.1, self.y_min),  # (x, y) bottom-left
            0.2,                         # width
            door_y_min - self.y_min,     # height
            linewidth=0,
            edgecolor=None,
            facecolor='black'
        )
        self.ax.add_patch(rect_bottom)
        
        # Top wall segment
        rect_top = patches.Rectangle(
            (wall_x - 0.1, door_y_max),  # (x, y) bottom-left
            0.2,                         # width
            self.y_max - door_y_max,     # height
            linewidth=0,
            edgecolor=None,
            facecolor='black'
        )
        self.ax.add_patch(rect_top)
        
        # Draw Goal
        if goal_pos is not None:
            goal = patches.Circle(
                goal_pos,
                radius=0.1,
                color='green',
                alpha=0.7
            )
            self.ax.add_patch(goal)
            
        # Draw Agent
        agent = patches.Circle(
            agent_pos,
            radius=0.1,
            color='blue'
        )
        self.ax.add_patch(agent)
        
        # Render to numpy array
        self.canvas.draw()
        
        # Convert to numpy array (modern matplotlib API)
        width, height = self.fig.get_size_inches() * self.fig.get_dpi()
        buf = np.asarray(self.canvas.buffer_rgba())
        image = buf[:, :, :3]  # Take RGB, drop alpha
        
        return image
        
    def close(self):
        plt.close(self.fig)
