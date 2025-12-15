# Project Structure

```
GradientPlanning/
├── .github/
│   ├── workflows/
│   │   └── python.yml          # CI/CD workflow
│   └── ISSUE_TEMPLATE/         # Issue templates
├── docs/
│   └── blog.md                 # Technical blog post
├── results/                     # Evaluation results and plots
│   ├── demo_*.png              # Demo visualizations
│   ├── trajectories_*.png      # Trajectory plots
│   └── results_*.txt            # Evaluation metrics
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── make_expert_data.py # Expert trajectory generation
│   ├── envs/
│   │   ├── __init__.py
│   │   └── wall_door.py        # 2D navigation environment
│   ├── models/
│   │   ├── __init__.py
│   │   └── world_model.py      # MLP world model
│   ├── planners/
│   │   ├── __init__.py
│   │   ├── gbp.py              # Gradient-based planner
│   │   ├── gbp_improved.py     # Improved planners
│   │   └── cem.py              # Cross-entropy method
│   ├── train/
│   │   ├── __init__.py
│   │   ├── train_baseline.py   # Baseline training
│   │   ├── train_adversarial.py # Adversarial finetuning
│   │   └── train_online.py     # Online finetuning
│   ├── eval/
│   │   ├── __init__.py
│   │   └── eval_planning.py    # Evaluation script
│   └── utils/
│       ├── __init__.py
│       ├── rollout.py          # Rollout utilities
│       ├── metrics.py           # Evaluation metrics
│       └── viz.py               # Visualization tools
├── .gitignore
├── LICENSE
├── README.md                    # Main documentation
├── CHANGELOG.md                 # Version history
├── CONTRIBUTING.md              # Contribution guidelines
├── CODE_OF_CONDUCT.md           # Code of conduct
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project metadata
├── setup.py                    # Package setup
├── demo.py                     # Demo visualization script
├── test_improvements.py        # Test improved planners
├── STATUS.md                   # Project status
├── RESULTS.md                  # Results analysis
├── IMPROVEMENTS.md             # Planning improvements
├── SUMMARY.md                  # Complete summary
├── NEXT_STEPS.md               # Future work
└── PROJECT_STRUCTURE.md        # This file
```

## Key Files

- **`README.md`**: Main documentation and quickstart guide
- **`docs/blog.md`**: Technical blog post explaining the implementation
- **`src/`**: All source code organized by component
- **`demo.py`**: Quick demo script to visualize results
- **`requirements.txt`**: Python dependencies

## Data Directories (Gitignored)

- **`data/`**: Expert trajectory datasets (`.npz` files)
- **`checkpoints/`**: Trained model checkpoints (`.pt` files)
- **`results/`**: Evaluation results (plots and metrics)

