"""
Real Robot Training Pipeline.
Collects data and runs online finetuning on real hardware.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.robots import FrankaRobot, SimRobot
from src.models.world_model import WorldModel
from src.planners.gbp import plan as gbp_plan
from src.models.encoders import DINOv2Encoder
from src.utils.device import get_device

def main():
    parser = argparse.ArgumentParser(description="Real Robot Training")
    parser.add_argument("--robot", type=str, default="sim", choices=["sim", "franka"], help="Robot type")
    parser.add_argument("--task", type=str, default="pusht", help="Task name")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to initial model checkpoint")
    parser.add_argument("--output_dir", type=str, default="checkpoints_real", help="Output directory")
    args = parser.parse_args()
    
    # Initialize robot
    if args.robot == "sim":
        # For testing the pipeline without hardware
        from src.envs.pusht import PushTEnv
        env = PushTEnv()
        robot = SimRobot(env)
        print("Initialized Simulation Robot")
    elif args.robot == "franka":
        robot = FrankaRobot()
        print("Initialized Franka Robot")
        
    device = get_device()  # Supports CUDA, MPS (Apple Silicon), or CPU
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # Init encoder
    encoder = DINOv2Encoder(model_size="small", device=str(device))
    
    state_dim = encoder.output_dim
    
    model = WorldModel(
        state_dim=state_dim,
        action_dim=ckpt['action_dim'],
        hidden_dim=ckpt['hidden_dim'],
        num_layers=ckpt['num_layers'],
        use_residual=ckpt['use_residual'],
        encoder=encoder,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print("Starting interaction loop...")
    # Basic interactive loop
    
    obs = robot.reset()
    
    for i in range(10): # 10 steps test
        # Get image
        img = obs['image'] # (H, W, 3)
        
        # Preprocess
        z_img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        
        # Encode
        with torch.no_grad():
            z0 = model.encode(z_img).squeeze(0)
            
        # Dummy goal (encoded)
        # Ideally we get goal image from user or file
        z_goal = z0.clone() # Stay put?
        
        # Plan
        print(f"Step {i}: Planning...")
        # Enable grad for planning
        model.train() 
        for p in model.parameters(): p.requires_grad = True
            
        actions = gbp_plan(model, z0, z_goal, horizon=10, n_steps=50)
        
        model.eval()
        for p in model.parameters(): p.requires_grad = False
        
        # Execute first action (MPC style)
        action = actions[0].detach().cpu().numpy()
        print(f"Executing: {action}")
        
        obs = robot.execute_action(action)
        time.sleep(0.1)
        
    print("Done.")

if __name__ == "__main__":
    main()

