"""
Configuration for reproducible experiments.
"""

# Data generation
DATA_CONFIG = {
    'n_trajectories': 1000,
    'max_steps': 100,
    'seed': 42,
    'door_width': 0.5,
    'action_max': 0.25,
}

# Training
TRAIN_CONFIG = {
    'batch_size': 64,
    'lr': 1e-3,
    'epochs': 50,
    'val_split': 0.1,
    'seed': 42,
}

# Adversarial training
ADVERSARIAL_CONFIG = {
    'eps_z': 0.1,
    'eps_a': 0.1,
    'lambda_z': 0.05,
    'lambda_a': 0.05,
    'lr': 1e-4,  # Lower for finetuning
    'epochs': 20,
}

# Online training
ONLINE_CONFIG = {
    'n_iterations': 5,
    'rollouts_per_iter': 50,
    'gbp_steps': 300,
    'lr': 1e-4,
    'epochs_per_iter': 5,
}

# Planning
PLANNING_CONFIG = {
    'horizon': 200,
    'gbp_steps': 300,
    'gbp_lr': 1e-3,
    'action_max': 0.25,
    'intermediate_weight': 0.1,  # Weight for intermediate goal loss
    'cem_iterations': 20,
    'cem_samples': 200,
}

# Evaluation
EVAL_CONFIG = {
    'n_episodes': 100,
    'horizon': 200,
    'seed': 42,
    'goal_threshold': 1.0,  # Increased from 0.5
}

# Environment
ENV_CONFIG = {
    'door_width': 0.5,
    'action_max': 0.25,
    'dt': 0.1,
    'goal_threshold': 1.0,  # Increased from 0.5
    'bounds': (-2.0, 2.0, -2.0, 2.0),
    'use_velocity': False,
}

