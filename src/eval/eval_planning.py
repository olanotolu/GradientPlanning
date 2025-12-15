"""
Evaluation script - compare planning methods.

Measures success rate, world model error, and planning time.
"""

import torch
import argparse
from pathlib import Path
import sys
import numpy as np
import time
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.world_model import WorldModel
from src.planners.gbp import plan as gbp_plan
from src.planners.cem import plan as cem_plan
from src.envs.wall_door import WallDoorEnv
from src.utils.rollout import rollout_model, rollout_sim
from src.utils.metrics import world_model_error, planning_success, distance_to_goal
from src.utils.viz import plot_trajectory, compare_trajectories
import matplotlib.pyplot as plt


def evaluate_planner(
    model,
    env,
    planner_name,
    n_episodes=100,
    horizon=25,
    gbp_steps=300,
    cem_iterations=10,
    cem_samples=100,
    seed=42,
    save_trajectories=False,
):
    """
    Evaluate a planner on multiple episodes.
    
    Args:
        model: World model
        env: Environment
        planner_name: 'gbp' or 'cem'
        n_episodes: Number of test episodes
        horizon: Planning horizon
        gbp_steps: Number of GBP optimization steps
        cem_iterations: Number of CEM iterations
        cem_samples: Number of CEM samples
        seed: Random seed
        save_trajectories: If True, save trajectory visualizations
        
    Returns:
        Dictionary with metrics
    """
    np.random.seed(seed)
    device = next(model.parameters()).device
    
    successes = []
    distances = []
    planning_times = []
    world_model_errors = []
    trajectories = []
    
    print(f"\nEvaluating {planner_name.upper()} planner on {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        # Sample random start and goal
        start = env.sample_random_state()
        goal = env.sample_goal()
        
        # Ensure start and goal on opposite sides of wall
        if np.random.rand() > 0.5:
            start[0] = np.random.uniform(env.bounds[0], -0.5)
            goal[0] = np.random.uniform(0.5, env.bounds[1])
        else:
            start[0] = np.random.uniform(0.5, env.bounds[1])
            goal[0] = np.random.uniform(env.bounds[0], -0.5)
        
        # Plan
        z0_tensor = torch.from_numpy(start).float().to(device)
        z_goal_tensor = torch.from_numpy(goal).float().to(device)
        
        start_time = time.time()
        
        if planner_name == 'gbp':
            # GBP needs gradients enabled - temporarily set model to train mode
            model.train()
            # Enable gradients for model parameters (needed for backprop through model)
            for param in model.parameters():
                param.requires_grad = True
            
            a_sequence = gbp_plan(
                model,
                z0_tensor,
                z_goal_tensor,
                horizon=horizon,
                n_steps=gbp_steps,
                action_max=env.action_max,
            )
            a_sequence = a_sequence.cpu().numpy()
            
            # Set model back to eval mode
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        elif planner_name == 'cem':
            with torch.no_grad():
                a_sequence = cem_plan(
                    model,
                    z0_tensor,
                    z_goal_tensor,
                    horizon=horizon,
                    n_iterations=cem_iterations,
                    n_samples=cem_samples,
                    action_max=env.action_max,
                )
                a_sequence = a_sequence.cpu().numpy()
        else:
            raise ValueError(f"Unknown planner: {planner_name}")
        
        planning_time = time.time() - start_time
        planning_times.append(planning_time)
        
        # Rollout in simulator
        z_sim_sequence = rollout_sim(env, start, a_sequence, horizon=horizon)
        z_final_sim = z_sim_sequence[-1]
        
        # Check success
        success = planning_success(z_final_sim, goal, threshold=env.goal_threshold)
        successes.append(success)
        
        # Distance to goal
        dist = distance_to_goal(z_final_sim, goal)
        distances.append(dist)
        
        # World model error (on planning trajectory)
        z_model_sequence = rollout_model(model, start, a_sequence, horizon=horizon)
        if isinstance(z_model_sequence, torch.Tensor):
            z_model_sequence = z_model_sequence.detach().cpu().numpy()
        
        wm_error = world_model_error(model, z_model_sequence, a_sequence, z_sim_sequence)
        world_model_errors.append(wm_error)
        
        # Save trajectory for visualization
        if save_trajectories and episode < 5:
            trajectories.append((z_sim_sequence, z_model_sequence, goal, success))
        
        if (episode + 1) % 10 == 0:
            current_success_rate = np.mean(successes)
            print(f"  Episode {episode + 1}/{n_episodes}: success_rate={current_success_rate:.2%}")
    
    results = {
        'success_rate': np.mean(successes),
        'avg_distance': np.mean(distances),
        'avg_planning_time': np.mean(planning_times),
        'avg_world_model_error': np.mean(world_model_errors),
        'successes': successes,
        'distances': distances,
        'planning_times': planning_times,
        'world_model_errors': world_model_errors,
        'trajectories': trajectories,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate planning methods")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, default="baseline", choices=["baseline", "adversarial", "online"], help="Model type")
    parser.add_argument("--planner", type=str, default="gbp", choices=["gbp", "cem", "both"], help="Planner to evaluate")
    parser.add_argument("--n_episodes", type=int, default=100, help="Number of test episodes")
    parser.add_argument("--horizon", type=int, default=25, help="Planning horizon")
    parser.add_argument("--gbp_steps", type=int, default=300, help="GBP optimization steps")
    parser.add_argument("--cem_iterations", type=int, default=10, help="CEM iterations")
    parser.add_argument("--cem_samples", type=int, default=100, help="CEM samples")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--save_plots", action="store_true", help="Save trajectory plots")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = WorldModel(
        state_dim=checkpoint['state_dim'],
        action_dim=checkpoint['action_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        use_residual=checkpoint['use_residual'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded {args.model_type} model")
    
    # Create environment with larger action scale
    env = WallDoorEnv(use_velocity=False, action_max=0.25, goal_threshold=0.5)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate planners
    all_results = {}
    
    if args.planner in ['gbp', 'both']:
        results_gbp = evaluate_planner(
            model,
            env,
            'gbp',
            n_episodes=args.n_episodes,
            horizon=args.horizon,
            gbp_steps=args.gbp_steps,
            seed=args.seed,
            save_trajectories=args.save_plots,
        )
        all_results['gbp'] = results_gbp
    
    if args.planner in ['cem', 'both']:
        results_cem = evaluate_planner(
            model,
            env,
            'cem',
            n_episodes=args.n_episodes,
            horizon=args.horizon,
            cem_iterations=args.cem_iterations,
            cem_samples=args.cem_samples,
            seed=args.seed,
            save_trajectories=args.save_plots,
        )
        all_results['cem'] = results_cem
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    for planner_name, results in all_results.items():
        print(f"\n{planner_name.upper()} Planner:")
        print(f"  Success Rate: {results['success_rate']:.2%}")
        print(f"  Avg Distance to Goal: {results['avg_distance']:.4f}")
        print(f"  Avg Planning Time: {results['avg_planning_time']:.4f} seconds")
        print(f"  Avg World Model Error: {results['avg_world_model_error']:.6f}")
    
    # Save results
    results_path = output_dir / f"results_{args.model_type}_{args.planner}.txt"
    with open(results_path, 'w') as f:
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Episodes: {args.n_episodes}\n")
        f.write("\n")
        
        for planner_name, results in all_results.items():
            f.write(f"{planner_name.upper()} Planner:\n")
            f.write(f"  Success Rate: {results['success_rate']:.2%}\n")
            f.write(f"  Avg Distance: {results['avg_distance']:.4f}\n")
            f.write(f"  Avg Planning Time: {results['avg_planning_time']:.4f}s\n")
            f.write(f"  Avg World Model Error: {results['avg_world_model_error']:.6f}\n")
            f.write("\n")
    
    print(f"\nResults saved to {results_path}")
    
    # Save trajectory plots
    if args.save_plots:
        for planner_name, results in all_results.items():
            if len(results['trajectories']) > 0:
                fig, axes = plt.subplots(1, len(results['trajectories']), figsize=(5 * len(results['trajectories']), 5))
                if len(results['trajectories']) == 1:
                    axes = [axes]
                
                for idx, (z_sim, z_model, goal, success) in enumerate(results['trajectories']):
                    ax = axes[idx]
                    plot_trajectory(
                        z_sim,
                        wall_x=env.wall_x,
                        door_y_min=env.door_y_min,
                        door_y_max=env.door_y_max,
                        goal=goal,
                        goal_threshold=env.goal_threshold,
                        title=f"{planner_name.upper()} - Episode {idx+1} ({'Success' if success else 'Fail'})",
                        ax=ax,
                    )
                
                plt.tight_layout()
                plot_path = output_dir / f"trajectories_{args.model_type}_{planner_name}.png"
                plt.savefig(plot_path)
                plt.close()
                print(f"Saved trajectory plots to {plot_path}")


if __name__ == "__main__":
    main()

