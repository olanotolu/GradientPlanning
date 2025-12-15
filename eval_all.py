#!/usr/bin/env python3
"""
Comprehensive evaluation: compare all methods on many episodes.

Runs baseline GBP, online finetuned GBP, adversarial finetuned GBP, and CEM.
"""

import torch
import argparse
from pathlib import Path
import sys
import numpy as np
import json
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from src.models.world_model import WorldModel
from src.planners.gbp import plan as gbp_plan
from src.planners.cem import plan as cem_plan
from src.envs.wall_door import WallDoorEnv
from src.utils.rollout import rollout_sim
from src.utils.metrics import world_model_error, planning_success, distance_to_goal


def evaluate_method(model, env, planner_name, n_episodes=100, horizon=200, 
                   gbp_steps=300, seed=42, action_max=0.25, goal_threshold=1.0):
    """Evaluate a method on multiple episodes."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = next(model.parameters()).device
    model.eval()
    
    results = {
        'successes': 0,
        'distances': [],
        'model_errors': [],
        'planning_times': []
    }
    
    for episode in range(n_episodes):
        # Random start/goal
        start = np.array([
            np.random.uniform(-1.5, -0.5),
            np.random.uniform(-1.5, 1.5)
        ], dtype=np.float32)
        
        goal = np.array([
            np.random.uniform(0.5, 1.5),
            np.random.uniform(-1.5, 1.5)
        ], dtype=np.float32)
        
        z0 = torch.from_numpy(start).float().to(device)
        z_goal = torch.from_numpy(goal).float().to(device)
        
        # Plan
        import time
        t0 = time.time()
        
        if planner_name == 'gbp':
            model.train()
            for param in model.parameters():
                param.requires_grad = True
            
            # Use intermediate goal loss for better gradients
            actions = gbp_plan(model, z0, z_goal, horizon=horizon, 
                             n_steps=gbp_steps, action_max=action_max,
                             intermediate_weight=0.1)
            
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        else:  # cem
            # More iterations and samples for CEM
            actions = cem_plan(model, z0, z_goal, horizon=horizon, 
                             action_max=action_max, n_iterations=20, n_samples=200)
        
        planning_time = time.time() - t0
        actions = actions.cpu().numpy()
        
        # Rollout in simulator
        states = rollout_sim(env, start, actions, horizon=horizon)
        
        # Metrics
        success = planning_success(states[-1], goal, goal_threshold)
        distance = distance_to_goal(states[-1], goal)
        
        # World model error - compute using one-step predictions
        model_states = [start]
        z = z0
        for a in actions:
            a_tensor = torch.from_numpy(a).float().unsqueeze(0).to(device)
            z = model(z.unsqueeze(0), a_tensor).squeeze(0)
            model_states.append(z.detach().cpu().numpy())
        model_states = np.array(model_states)  # [horizon+1, state_dim]
        
        wm_error = world_model_error(model, model_states, actions, states)
        
        results['successes'] += success
        results['distances'].append(distance)
        results['model_errors'].append(wm_error)
        results['planning_times'].append(planning_time)
        
        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{n_episodes}: "
                  f"Success rate: {results['successes']/(episode+1)*100:.1f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action_max", type=float, default=0.25)
    parser.add_argument("--goal_threshold", type=float, default=1.0)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = WallDoorEnv(use_velocity=False, action_max=args.action_max, goal_threshold=args.goal_threshold)
    
    methods = {}
    
    # Baseline
    print("Evaluating baseline GBP...")
    baseline_ckpt = torch.load("checkpoints/baseline_best.pt", map_location=device)
    baseline_model = WorldModel(**{k: baseline_ckpt[k] for k in 
        ['state_dim', 'action_dim', 'hidden_dim', 'num_layers', 'use_residual']}).to(device)
    baseline_model.load_state_dict(baseline_ckpt['model_state_dict'])
    methods['Baseline GBP'] = evaluate_method(baseline_model, env, 'gbp', 
                                             args.n_episodes, args.horizon, seed=args.seed,
                                             goal_threshold=args.goal_threshold)
    
    # Online finetuned
    print("\nEvaluating online finetuned GBP...")
    online_paths = ["checkpoints/online_final.pt", "checkpoints_v2/online_final.pt"]
    online_loaded = False
    for online_path in online_paths:
        try:
            online_ckpt = torch.load(online_path, map_location=device)
            online_model = WorldModel(**{k: online_ckpt[k] for k in 
                ['state_dim', 'action_dim', 'hidden_dim', 'num_layers', 'use_residual']}).to(device)
            online_model.load_state_dict(online_ckpt['model_state_dict'])
            methods['Online GBP'] = evaluate_method(online_model, env, 'gbp',
                                                  args.n_episodes, args.horizon, seed=args.seed,
                                                  goal_threshold=args.goal_threshold)
            online_loaded = True
            break
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"  Error loading {online_path}: {e}")
            continue
    if not online_loaded:
        print("  Online model not found, skipping")
    
    # Adversarial finetuned
    print("\nEvaluating adversarial finetuned GBP...")
    try:
        adv_ckpt = torch.load("checkpoints/adversarial_best.pt", map_location=device)
        adv_model = WorldModel(**{k: adv_ckpt[k] for k in 
            ['state_dim', 'action_dim', 'hidden_dim', 'num_layers', 'use_residual']}).to(device)
        adv_model.load_state_dict(adv_ckpt['model_state_dict'])
        methods['Adversarial GBP'] = evaluate_method(adv_model, env, 'gbp',
                                                   args.n_episodes, args.horizon, seed=args.seed,
                                                   goal_threshold=args.goal_threshold)
    except:
        print("  Adversarial model not found, skipping")
    
    # Combined (online → adversarial)
    print("\nEvaluating combined (online → adversarial) GBP...")
    try:
        combined_ckpt = torch.load("checkpoints/combined_final.pt", map_location=device)
        combined_model = WorldModel(**{k: combined_ckpt[k] for k in 
            ['state_dim', 'action_dim', 'hidden_dim', 'num_layers', 'use_residual']}).to(device)
        combined_model.load_state_dict(combined_ckpt['model_state_dict'])
        methods['Combined GBP'] = evaluate_method(combined_model, env, 'gbp',
                                                  args.n_episodes, args.horizon, seed=args.seed,
                                                  goal_threshold=args.goal_threshold)
    except FileNotFoundError:
        print("  Combined model not found, skipping (run train_combined.py first)")
    except Exception as e:
        print(f"  Error loading combined model: {e}")
    
    # CEM baseline
    print("\nEvaluating CEM...")
    methods['CEM'] = evaluate_method(baseline_model, env, 'cem',
                                    args.n_episodes, args.horizon, seed=args.seed,
                                    goal_threshold=args.goal_threshold)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\n{'Method':<20} {'Success %':<12} {'Avg Distance':<15} {'WM Error':<12} {'Time (s)':<12}")
    print("-"*60)
    
    for name, res in methods.items():
        success_pct = res['successes'] / args.n_episodes * 100
        avg_dist = np.mean(res['distances'])
        avg_error = np.mean(res['model_errors'])
        avg_time = np.mean(res['planning_times'])
        print(f"{name:<20} {success_pct:>6.1f}%     {avg_dist:>6.3f}        {avg_error:>6.3f}     {avg_time:>6.3f}")
    
    # Save results
    output = {
        'n_episodes': args.n_episodes,
        'horizon': args.horizon,
        'seed': args.seed,
        'methods': {name: {
            'success_rate': res['successes'] / args.n_episodes,
            'avg_distance': float(np.mean(res['distances'])),
            'avg_wm_error': float(np.mean(res['model_errors'])),
            'avg_time': float(np.mean(res['planning_times']))
        } for name, res in methods.items()}
    }
    
    with open('results/eval_all.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to results/eval_all.json")


if __name__ == '__main__':
    main()

