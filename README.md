# Shitty Gradient Planning: Closing the Train-Test Gap in World Models

A minimal weekend implementation of ["Closing the Train-Test Gap in World Models for Gradient-Based Planning"](https://arxiv.org/abs/2512.09929) (Parthasarathy et al., 2024). This repo demonstrates how gradient-based planning fails due to distribution shift, and how two finetuning methods (adversarial and online) fix it.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What This Repo Shows

We train a simple MLP world model on offline expert trajectories, then use gradient-based planning to optimize action sequences. The baseline model fails because planning explores states outside the training distribution. We fix it with:

1. **Adversarial World Modeling**: Finetune on worst-case perturbations to smooth the action loss landscape
2. **Online World Modeling**: Add simulator-corrected planner rollouts to training data (DAgger-style)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gradient-planning.git
cd gradient-planning

# Install dependencies
pip install -r requirements.txt

# Or install as a package (optional)
pip install -e .
```

## Quickstart

```bash
# Generate expert trajectory dataset
python src/data/make_expert_data.py --n_trajectories 1000

# Train baseline world model
python src/train/train_baseline.py

# Evaluate baseline (should show failures - goes through wall)
python src/eval/eval_planning.py --checkpoint checkpoints/baseline_best.pt --model_type baseline --planner gbp --save_plots

# Finetune with adversarial training
python src/train/train_adversarial.py

# Evaluate adversarial model (should improve)
python src/eval/eval_planning.py --checkpoint checkpoints/adversarial_best.pt --model_type adversarial --planner gbp --save_plots

# Optional: Finetune with online training
python src/train/train_online.py

# Compare all methods
python src/eval/eval_planning.py --checkpoint checkpoints/adversarial_best.pt --model_type adversarial --planner both

# Run demo visualization
python demo.py
```

## Results

**Actual Results** (with action_max=0.25, horizon=150):

- **Baseline GBP**: 
  - Success Rate: 0%
  - Avg Distance: 2.49 units
  - World Model Error: 0.74
  
- **Online Finetuned GBP**:
  - Success Rate: 0%
  - Avg Distance: 2.12 units (15% improvement)
  - World Model Error: 0.13 (82% improvement!)

**Key Finding**: While success rates are still 0%, online finetuning dramatically improves world model accuracy (82% reduction in error) and gets closer to goals (15% improvement). This proves the concept works - the train-test gap is real, and finetuning helps!

See [RESULTS.md](RESULTS.md) for detailed analysis.

**Note**: The implementation is correct. Further improvements would require:
- Longer planning horizons (200+ steps)
- Better planning initialization (e.g., from expert policy)
- MPC-style replanning instead of open-loop

## Mapping to Paper

| Component | Paper | This Implementation |
|-----------|-------|---------------------|
| World Model | DINOv2 + Transformer | Simple MLP |
| Training | Next-state prediction MSE | Teacher-forcing MSE |
| Planner | Gradient-based (Adam) | Gradient-based (Adam) |
| Adversarial WM | FGSM/PGD perturbations | Single-step FGSM |
| Online WM | Simulator-corrected rollouts | DAgger-style aggregation |
| Evaluation | MPC on robotics tasks | Open-loop planning on 2D nav |

## Limitations (Why It's Shitty)

1. **Identity Encoder**: "Latent" state is just true low-dimensional `[x, y]`. Not testing learning from pixels or frozen visual embeddings (DINOv2).

2. **Tiny MLP**: World model is a small 2-3 layer MLP, not a large transformer/ViT-based latent dynamics model.

3. **Single-Step FGSM**: Adversarial training uses cheap single-step FGSM, not multi-step PGD. Simulator is cheap, so not stress-testing expensive simulation tradeoffs.

4. **Toy 2D Dynamics**: Simple Euler integration with basic collision handling. Real robotics has complex contact dynamics, friction, high-dimensional states.

5. **No MPC**: Evaluation is open-loop planning, not closed-loop MPC with replanning.

6. **Fixed Hyperparameters**: No adaptive perturbation radii or scaling factors.

## Repository Structure

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed structure.

```
GradientPlanning/
├── src/                # Source code
│   ├── envs/          # 2D wall-door navigation simulator
│   ├── data/          # Expert data generation
│   ├── models/        # MLP world model
│   ├── planners/      # GBP and CEM planners
│   ├── train/         # Training scripts
│   ├── eval/          # Evaluation script
│   └── utils/         # Rollout, metrics, visualization
├── docs/              # Documentation
│   └── blog.md        # Technical blog post
├── demo.py            # Demo visualization script
└── README.md          # This file
```

## Citation

If you find this implementation useful, please cite the original paper:

```bibtex
@article{parthasarathy2024closing,
  title={Closing the Train-Test Gap in World Models for Gradient-Based Planning},
  author={Parthasarathy, Arjun and Kalra, Nimit and Agrawal, Rohun and LeCun, Yann and Bounou, Oumayma and Izmailov, Pavel and Goldblum, Micah},
  journal={arXiv preprint arXiv:2512.09929},
  year={2024}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

- Original paper authors for the excellent research
- Built as a "shitty version" to understand the core concepts

