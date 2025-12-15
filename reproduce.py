#!/usr/bin/env python3
"""
One-command reproduction script.

Runs the full pipeline: data -> baseline -> finetune -> evaluate.
"""

import subprocess
import sys
from pathlib import Path

def run(cmd, check=True):
    """Run a command."""
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print('='*60)
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print(f"Error: {cmd} failed")
        sys.exit(1)
    return result

def main():
    print("Reproducing full pipeline...")
    
    # 1. Generate data
    run("python src/data/make_expert_data.py --n_trajectories 1000 --seed 42")
    
    # 2. Train baseline
    run("python src/train/train_baseline.py --epochs 50 --seed 42")
    
    # 3. Finetune online
    run("python src/train/train_online.py --n_iterations 5 --seed 42")
    
    # 4. Finetune adversarial
    run("python src/train/train_adversarial.py --epochs 20 --seed 42")
    
    # 5. Evaluate all
    run("python eval_all.py --n_episodes 100 --seed 42")
    
    print("\n" + "="*60)
    print("Pipeline complete! Check results/eval_all.json")
    print("="*60)

if __name__ == '__main__':
    main()

