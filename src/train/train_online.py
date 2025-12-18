"""
Online world modeling - finetune on planner rollouts corrected by simulator.

DAgger-style: add planner trajectories to training data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.world_model import WorldModel
from src.models.encoders import DINOv2Encoder
from src.data import ExpertDataset
from src.planners.gbp import plan as gbp_plan
from src.envs.wall_door import WallDoorEnv
from src.envs.pusht import PushTEnv
from src.envs.pointmaze import PointMazeEnv
from src.utils.rollout import rollout_sim
from src.utils.metrics import planning_success


from src.models.transformer_world_model import TransformerWorldModel
from src.utils.device import get_device, set_seed


def resize_image(img: np.ndarray, target_resolution: int) -> np.ndarray:
    """Resize image to target resolution."""
    if img.shape[0] == target_resolution and img.shape[1] == target_resolution:
        return img
    # img is (H, W, 3) uint8
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((target_resolution, target_resolution), Image.Resampling.LANCZOS)
    return np.array(pil_img)


class ReplayBuffer:
    """Simple FIFO replay buffer for storing transitions."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = []
    
    def add(self, z, a, z_next):
        """Add transition to buffer."""
        self.buffer.append((z, a, z_next))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)  # FIFO
    
    def sample(self, n: int):
        """Sample n transitions."""
        indices = np.random.choice(len(self.buffer), size=min(n, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)
    
    def get_dataset(self, use_images: bool = False):
        """Convert buffer to PyTorch dataset."""
        if len(self.buffer) == 0:
            return None
        
        z_list, a_list, z_next_list = zip(*self.buffer)
        
        if use_images:
            # z_list contains images (H, W, 3)
            # Convert to (N, 3, H, W) float [0, 1]
            z_np = np.array(z_list)
            z_next_np = np.array(z_next_list)
            
            z_np = np.transpose(z_np, (0, 3, 1, 2))
            z_next_np = np.transpose(z_next_np, (0, 3, 1, 2))
            
            z_tensor = torch.from_numpy(z_np).float() / 255.0
            z_next_tensor = torch.from_numpy(z_next_np).float() / 255.0
        else:
            z_tensor = torch.stack([torch.from_numpy(np.array(z, dtype=np.float32)) for z in z_list])
            z_next_tensor = torch.stack([torch.from_numpy(np.array(z_next, dtype=np.float32)) for z_next in z_next_list])
            
        a_tensor = torch.stack([torch.from_numpy(np.array(a, dtype=np.float32)) for a in a_list])
        
        return TensorDataset(z_tensor, a_tensor, z_next_tensor)


class SequenceReplayBuffer:
    """Replay buffer for storing trajectories (for Transformer)."""
    
    def __init__(self, max_trajectories: int = 1000, context_length: int = 16):
        self.max_trajectories = max_trajectories
        self.context_length = context_length
        self.trajectories = [] # List of dicts {'states': ..., 'actions': ...}
        
    def add_trajectory(self, states, actions):
        """Add full trajectory."""
        self.trajectories.append({
            'states': states, # List or array
            'actions': actions
        })
        if len(self.trajectories) > self.max_trajectories:
            self.trajectories.pop(0)
            
    def __len__(self):
        return len(self.trajectories)
        
    def get_dataset(self, use_images: bool = False):
        """Return a Dataset object compatible with DataLoader."""
        if len(self.trajectories) == 0:
            return None
            
        # We can implement a custom Dataset that samples from trajectories
        # similar to SequenceDataset in data/__init__.py
        # But SequenceDataset expects a file path.
        # So we define a temporary class here.
        
        class InMemorySequenceDataset(torch.utils.data.Dataset):
            def __init__(self, trajectories, context_length, use_images):
                self.trajectories = trajectories
                self.context_length = context_length
                self.use_images = use_images
                
                self.indices = []
                for traj_idx, traj in enumerate(self.trajectories):
                    T = len(traj['states']) # Assuming states includes last state
                    # states has T+1 items (s_0...s_T), actions has T items (a_0...a_{T-1})
                    # But wait, add_trajectory received what?
                    # In main loop, we'll pass full lists.
                    
                    # Effective length for sequence modeling (input-target pairs) is len(actions)
                    # We need s_{t:t+L}, a_{t:t+L} -> predict s_{t+1:t+L+1}
                    
                    # Let's verify lengths
                    n_actions = len(traj['actions'])
                    
                    if n_actions >= context_length:
                        for t in range(n_actions - context_length + 1):
                            self.indices.append((traj_idx, t))
                            
            def __len__(self):
                return len(self.indices)
                
            def __getitem__(self, idx):
                traj_idx, start_idx = self.indices[idx]
                traj = self.trajectories[traj_idx]
                end_idx = start_idx + self.context_length
                
                # States: s_t ... s_{t+L-1}
                # Next States: s_{t+1} ... s_{t+L}
                
                # Convert to tensor on fly if not already
                # Assuming inputs are lists of numpy arrays (or images)
                
                s_seq_raw = traj['states'][start_idx:end_idx]
                a_seq_raw = traj['actions'][start_idx:end_idx]
                ns_seq_raw = traj['states'][start_idx+1:end_idx+1]
                
                if self.use_images:
                    s_np = np.array(s_seq_raw)
                    ns_np = np.array(ns_seq_raw)
                    # (T, H, W, 3) -> (T, 3, H, W)
                    s_np = np.transpose(s_np, (0, 3, 1, 2))
                    ns_np = np.transpose(ns_np, (0, 3, 1, 2))
                    
                    s_seq = torch.from_numpy(s_np).float() / 255.0
                    ns_seq = torch.from_numpy(ns_np).float() / 255.0
                else:
                    s_seq = torch.from_numpy(np.array(s_seq_raw, dtype=np.float32))
                    ns_seq = torch.from_numpy(np.array(ns_seq_raw, dtype=np.float32))
                    
                a_seq = torch.from_numpy(np.array(a_seq_raw, dtype=np.float32))
                
                return s_seq, a_seq, ns_seq
                
        return InMemorySequenceDataset(self.trajectories, self.context_length, use_images)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        if len(batch) == 3:
            z, a, z_next = batch
            z = z.to(device)
            a = a.to(device)
            z_next = z_next.to(device)
            
            # Encode images if using encoder
            if hasattr(model, 'encoder') and model.encoder is not None:
                with torch.no_grad():
                    if z.dim() == 5: 
                        B, T, C, H, W = z.shape
                        z = model.encode(z.view(-1, C, H, W)).view(B, T, -1)
                        z_next = model.encode(z_next.view(-1, C, H, W)).view(B, T, -1)
                    else:
                        z = model.encode(z)
                        z_next = model.encode(z_next)
            
            optimizer.zero_grad()
            
            z_next_pred = model(z, a)
            loss = criterion(z_next_pred, z_next)
            
            loss.backward()
            
            if isinstance(model, TransformerWorldModel):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0



def evaluate_success_rate(model, env, device, n_episodes=20, horizon=200, gbp_steps=300, seed=42, use_images=False, image_resolution=224, task_name="wall"):
    """
    Evaluate success rate of current model on test episodes.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    successes = 0
    
    for episode in range(n_episodes):
        # Sample random start and goal (task-specific)
        if task_name == "wall":
            # Wall task: opposite sides
            if np.random.rand() > 0.5:
                start = np.array([
                    np.random.uniform(env.bounds[0], -0.5),
                    np.random.uniform(env.bounds[2], env.bounds[3])
                ], dtype=np.float32)
                goal = np.array([
                    np.random.uniform(0.5, env.bounds[1]),
                    np.random.uniform(env.bounds[2], env.bounds[3])
                ], dtype=np.float32)
            else:
                start = np.array([
                    np.random.uniform(0.5, env.bounds[1]),
                    np.random.uniform(env.bounds[2], env.bounds[3])
                ], dtype=np.float32)
                goal = np.array([
                    np.random.uniform(env.bounds[0], -0.5),
                    np.random.uniform(env.bounds[2], env.bounds[3])
                ], dtype=np.float32)
        elif task_name == "pointmaze":
            # U-maze: bottom left to bottom right
            start = np.array([-1.5, -1.5, 0.0, 0.0], dtype=np.float32)
            goal = np.array([1.5, -1.5], dtype=np.float32)
        else:
            # pusht or other: use environment's sampling
            start = env.sample_random_state()
            goal = env.sample_goal()
            
        if use_images:
            # Render start and goal images
            img_start = env.render(start, goal=goal, resolution=image_resolution)
            img_start = resize_image(img_start, image_resolution)
            # For goal rendering, construct a full state if needed
            if task_name == "pusht":
                # goal is block pose (3D), need full state (5D: agent + block)
                goal_state = np.concatenate([start[:2], goal])  # Use start's agent pos, goal's block pose
                img_goal = env.render(goal_state, goal=goal, resolution=image_resolution)
            else:
                img_goal = env.render(goal, goal=goal, resolution=image_resolution)
            img_goal = resize_image(img_goal, image_resolution)
            
            # Prepare tensors (1, 3, H, W)
            z0_img = torch.from_numpy(img_start).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            z_goal_img = torch.from_numpy(img_goal).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            
            # Encode
            with torch.no_grad():
                z0 = model.encode(z0_img).squeeze(0)
                z_goal = model.encode(z_goal_img).squeeze(0)
        else:
            z0 = torch.from_numpy(start).float().to(device)
            z_goal = torch.from_numpy(goal).float().to(device)
        
        # Plan
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        
        try:
            actions = gbp_plan(
                model, z0, z_goal, horizon=horizon,
                n_steps=gbp_steps, action_max=env.action_max,
                intermediate_weight=0.1
            )
        except:
            # If planning fails, count as failure
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            continue
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        # Rollout in simulator
        actions = actions.cpu().numpy()
        states = rollout_sim(env, start, actions, horizon=horizon)
        
        # Check success
        if planning_success(states[-1], goal, env.goal_threshold):
            successes += 1
    
    return successes / n_episodes


def main():
    parser = argparse.ArgumentParser(description="Online finetuning of world model")
    parser.add_argument("--data_path", type=str, default="data/expert_data.npz", help="Path to expert data")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/baseline_best.pt", help="Path to baseline checkpoint")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n_iterations", type=int, default=10, help="Number of DAgger iterations")
    parser.add_argument("--n_rollouts_per_iter", type=int, default=10, help="Number of planner rollouts per iteration")
    parser.add_argument("--epochs_per_iter", type=int, default=5, help="Epochs of finetuning per iteration")
    parser.add_argument("--expert_mix_ratio", type=float, default=0.5, help="Ratio of expert data in mixed dataset")
    parser.add_argument("--horizon", type=int, default=200, help="Planning horizon")
    parser.add_argument("--gbp_steps", type=int, default=300, help="GBP optimization steps")
    parser.add_argument("--intermediate_weight", type=float, default=0.1, help="Weight for intermediate goal loss")
    parser.add_argument("--replay_buffer_size", type=int, default=10000, help="Replay buffer size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_images", action="store_true", help="Use images instead of states")
    parser.add_argument("--image_model_size", type=str, default="small", help="DINOv2 model size")
    parser.add_argument("--image_resolution", type=int, default=224, help="Image resolution")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "transformer"], help="Model type")
    parser.add_argument("--context_length", type=int, default=16, help="Context length for transformer")
    parser.add_argument("--task", type=str, default="wall", choices=["wall", "pusht", "pointmaze"], help="Task")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device (supports CUDA, MPS for Apple Silicon, or CPU)
    device = get_device()
    print(f"Using device: {device}")
    
    # Create environment based on task
    if args.task == "wall":
        env = WallDoorEnv(use_velocity=False, action_max=0.25, goal_threshold=1.0)
    elif args.task == "pusht":
        env = PushTEnv(action_max=0.1, goal_threshold=1.0)
    elif args.task == "pointmaze":
        env = PointMazeEnv(action_max=1.0, goal_threshold=1.0)
    
    # Initialize encoder if using images
    encoder = None
    if args.use_images:
        print(f"Initializing DINOv2 {args.image_model_size} encoder...")
        encoder = DINOv2Encoder(model_size=args.image_model_size, device=str(device))
        print(f"Encoder output dim: {encoder.output_dim}")
    
    # Load baseline model
    print(f"Loading baseline model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    state_dim = checkpoint['state_dim']
    if args.use_images and encoder is not None:
        state_dim = encoder.output_dim
        
    if args.model_type == "transformer":
        model = TransformerWorldModel(
            state_dim=state_dim,
            action_dim=checkpoint['action_dim'],
            embed_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers'],
            num_heads=4,
            encoder=encoder,
            max_len=args.context_length + args.horizon + 10 # Buffer
        ).to(device)
    else:
        model = WorldModel(
            state_dim=state_dim,
            action_dim=checkpoint['action_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers'],
            use_residual=checkpoint['use_residual'],
            encoder=encoder,
        ).to(device)
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError:
        print("Warning: Could not load state dict. Starting from scratch/random init.")
    
    print("Loaded baseline model")
    
    # Load expert dataset
    print(f"Loading expert dataset from {args.data_path}...")
    try:
        if args.model_type == "transformer":
            from src.data import SequenceDataset
            expert_dataset = SequenceDataset(args.data_path, context_length=args.context_length, use_images=args.use_images)
        else:
            expert_dataset = ExpertDataset(args.data_path, use_images=args.use_images)
    except ValueError as e:
        print(f"Error loading dataset: {e}")
        return

    expert_loader = DataLoader(expert_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Replay buffer for planner-generated data
    if args.model_type == "transformer":
        replay_buffer = SequenceReplayBuffer(max_trajectories=args.replay_buffer_size // 10, context_length=args.context_length)
    else:
        replay_buffer = ReplayBuffer(max_size=args.replay_buffer_size)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Track success rates over iterations
    success_rates = []
    
    print("\nStarting online finetuning (DAgger-style)...")
    
    for iteration in range(args.n_iterations):
        print(f"\n=== Iteration {iteration + 1}/{args.n_iterations} ===")
        
        # Generate planner rollouts
        print(f"Generating {args.n_rollouts_per_iter} planner rollouts...")
        for rollout_idx in range(args.n_rollouts_per_iter):
            # Sample random start and goal (task-specific)
            if args.task == "wall":
                # Wall task: opposite sides
                if np.random.rand() > 0.5:
                    start = np.array([
                        np.random.uniform(env.bounds[0], -0.5),
                        np.random.uniform(env.bounds[2], env.bounds[3])
                    ], dtype=np.float32)
                    goal = np.array([
                        np.random.uniform(0.5, env.bounds[1]),
                        np.random.uniform(env.bounds[2], env.bounds[3])
                    ], dtype=np.float32)
                else:
                    start = np.array([
                        np.random.uniform(0.5, env.bounds[1]),
                        np.random.uniform(env.bounds[2], env.bounds[3])
                    ], dtype=np.float32)
                    goal = np.array([
                        np.random.uniform(env.bounds[0], -0.5),
                        np.random.uniform(env.bounds[2], env.bounds[3])
                    ], dtype=np.float32)
            elif args.task == "pointmaze":
                # U-maze: bottom left to bottom right
                start = np.array([-1.5, -1.5, 0.0, 0.0], dtype=np.float32)
                goal = np.array([1.5, -1.5], dtype=np.float32)
            else:
                # pusht or other: use environment's sampling
                start = env.sample_random_state()
                goal = env.sample_goal()
            
            if args.use_images:
                img_start = env.render(start, goal=goal, resolution=args.image_resolution)
                img_start = resize_image(img_start, args.image_resolution)
                # For goal rendering, construct a full state if needed
                if args.task == "pusht":
                    # goal is block pose (3D), need full state (5D: agent + block)
                    goal_state = np.concatenate([start[:2], goal])  # Use start's agent pos, goal's block pose
                    img_goal = env.render(goal_state, goal=goal, resolution=args.image_resolution)
                else:
                    img_goal = env.render(goal, goal=goal, resolution=args.image_resolution)
                img_goal = resize_image(img_goal, args.image_resolution)
                z0_tensor = torch.from_numpy(img_start).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                z_goal_tensor = torch.from_numpy(img_goal).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                
                with torch.no_grad():
                    z0_tensor = model.encode(z0_tensor).squeeze(0)
                    z_goal_tensor = model.encode(z_goal_tensor).squeeze(0)
            else:
                z0_tensor = torch.from_numpy(start).float().to(device)
                z_goal_tensor = torch.from_numpy(goal).float().to(device)
            
            # GBP needs gradients enabled - temporarily set model to train mode
            model.train()
            for param in model.parameters():
                param.requires_grad = True
            
            a_sequence = gbp_plan(
                model,
                z0_tensor,
                z_goal_tensor,
                horizon=args.horizon,
                n_steps=args.gbp_steps,
                action_max=env.action_max,
                intermediate_weight=args.intermediate_weight,
            )
            a_sequence = a_sequence.cpu().numpy()
            
            # Set model back to eval mode for rollout
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            
            # Rollout in real simulator to get corrected states
            z_sim_sequence = rollout_sim(env, start, a_sequence, horizon=args.horizon)
            
            # Add transitions to replay buffer
            if args.model_type == "transformer":
                # Add full trajectory
                traj_states = []
                # z_sim_sequence has T+1 states
                for s in z_sim_sequence:
                    if args.use_images:
                        img = env.render(s, goal=goal, resolution=args.image_resolution)
                        img = resize_image(img, args.image_resolution)
                        traj_states.append(img)
                    else:
                        traj_states.append(s)
                replay_buffer.add_trajectory(traj_states, a_sequence)
            else:
                for t in range(len(a_sequence)):
                    state_t = z_sim_sequence[t]
                    state_next_t = z_sim_sequence[t + 1]
                    
                    if args.use_images:
                        z_t = env.render(state_t, goal=goal, resolution=args.image_resolution)
                        z_t = resize_image(z_t, args.image_resolution)
                        z_next_t = env.render(state_next_t, goal=goal, resolution=args.image_resolution)
                        z_next_t = resize_image(z_next_t, args.image_resolution)
                    else:
                        z_t = state_t
                        z_next_t = state_next_t
                        
                    a_t = a_sequence[t]
                    replay_buffer.add(z_t, a_t, z_next_t)
        
        print(f"Replay buffer size: {len(replay_buffer)}")
        
        # Create mixed dataset (expert + planner data)
        planner_dataset = replay_buffer.get_dataset(use_images=args.use_images)
        
        if planner_dataset is not None and len(planner_dataset) > 0:
            # Mix expert and planner data
            n_expert = int(args.batch_size * args.expert_mix_ratio)
            n_planner = args.batch_size - n_expert
            
            expert_loader_mixed = DataLoader(expert_dataset, batch_size=n_expert, shuffle=True)
            planner_loader = DataLoader(planner_dataset, batch_size=n_planner, shuffle=True)
            
            # Finetune for a few epochs
            print(f"Finetuning for {args.epochs_per_iter} epochs...")
            model.train()  # Set to train mode for finetuning
            for param in model.parameters():
                param.requires_grad = True
            
            for epoch in range(args.epochs_per_iter):
                # Mix batches from expert and planner
                expert_iter = iter(expert_loader_mixed)
                planner_iter = iter(planner_loader)
                
                total_loss = 0.0
                n_batches = 0
                
                try:
                    while True:
                        # Get batches
                        batch_expert = next(expert_iter)
                        batch_planner = next(planner_iter)
                        
                        # Handle variable number of return items (transition vs sequence)
                        if len(batch_expert) == 3: # (z, a, z_next) or (z_seq, a_seq, z_next_seq)
                            z_expert, a_expert, z_next_expert = batch_expert
                            z_planner, a_planner, z_next_planner = batch_planner
                            
                            z_batch = torch.cat([z_expert, z_planner], dim=0).to(device)
                            a_batch = torch.cat([a_expert, a_planner], dim=0).to(device)
                            z_next_batch = torch.cat([z_next_expert, z_next_planner], dim=0).to(device)
                            
                            # Encode if using images
                            if model.encoder is not None:
                                with torch.no_grad():
                                    if z_batch.dim() == 5: 
                                        B, T, C, H, W = z_batch.shape
                                        z_batch = model.encode(z_batch.view(-1, C, H, W)).view(B, T, -1)
                                        z_next_batch = model.encode(z_next_batch.view(-1, C, H, W)).view(B, T, -1)
                                    else:
                                        z_batch = model.encode(z_batch)
                                        z_next_batch = model.encode(z_next_batch)
                            
                            optimizer.zero_grad()
                            z_next_pred = model(z_batch, a_batch)
                            loss = criterion(z_next_pred, z_next_batch)
                            loss.backward()
                            
                            if isinstance(model, TransformerWorldModel):
                                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                                
                            optimizer.step()
                            
                            total_loss += loss.item()
                            n_batches += 1
                except StopIteration:
                    pass
                
                if n_batches > 0:
                    avg_loss = total_loss / n_batches
                    print(f"  Epoch {epoch + 1}/{args.epochs_per_iter}: loss={avg_loss:.6f}")
        else:
            print("No planner data yet, skipping finetuning")
        
        # Evaluate success rate
        print("Evaluating success rate...")
        success_rate = evaluate_success_rate(
            model, env, device, n_episodes=20, horizon=args.horizon,
            gbp_steps=args.gbp_steps, seed=args.seed + iteration,
            use_images=args.use_images, image_resolution=args.image_resolution,
            task_name=args.task
        )
        success_rates.append(success_rate)
        print(f"Success rate: {success_rate*100:.1f}%")
        
        # Save checkpoint
        checkpoint_path = output_dir / f"online_iter_{iteration + 1}.pt"
        torch.save({
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'replay_buffer_size': len(replay_buffer),
            'success_rate': success_rate,
            'state_dim': state_dim,
            'action_dim': checkpoint['action_dim'],
            'hidden_dim': checkpoint['hidden_dim'],
            'num_layers': checkpoint['num_layers'],
            'use_residual': checkpoint['use_residual'],
            'model_type': args.model_type,
        }, checkpoint_path)
    
    # Save final model
    final_path = output_dir / "online_final.pt"
    torch.save({
        'iteration': args.n_iterations,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'replay_buffer_size': len(replay_buffer),
        'state_dim': state_dim,
        'action_dim': checkpoint['action_dim'],
        'hidden_dim': checkpoint['hidden_dim'],
        'num_layers': checkpoint['num_layers'],
        'use_residual': checkpoint['use_residual'],
        'model_type': args.model_type,
    }, final_path)
    
    print(f"\nOnline finetuning complete!")
    print(f"Final replay buffer size: {len(replay_buffer)}")
    print(f"Models saved to {output_dir}")
    
    # Create success rate plot
    if len(success_rates) > 0:
        plt.figure(figsize=(8, 6))
        iterations = range(1, len(success_rates) + 1)
        plt.plot(iterations, [sr * 100 for sr in success_rates], 'o-', linewidth=2, markersize=8)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Success Rate (%)', fontsize=12)
        plt.title('Success Rate Over Online Finetuning Iterations', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, max(35, max(sr * 100 for sr in success_rates) * 1.2)])
        
        # Save plot
        plot_path = Path("results") / "online_training_success_rate.png"
        plot_path.parent.mkdir(exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Success rate plot saved to {plot_path}")
        plt.close()


if __name__ == "__main__":
    main()
