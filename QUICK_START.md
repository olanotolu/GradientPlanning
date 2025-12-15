# Quick Start Guide

Get up and running in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gradient-planning.git
cd gradient-planning

# Install dependencies
pip install -r requirements.txt
```

## Minimal Example

```bash
# 1. Generate expert data (takes ~1 minute)
python src/data/make_expert_data.py --n_trajectories 500

# 2. Train baseline model (takes ~2 minutes)
python src/train/train_baseline.py --epochs 20

# 3. Evaluate baseline (shows train-test gap)
python src/eval/eval_planning.py \
  --checkpoint checkpoints/baseline_best.pt \
  --model_type baseline \
  --planner gbp \
  --n_episodes 10 \
  --save_plots

# 4. Run demo visualization
python demo.py
```

## What You'll See

- **Baseline**: High world model error (0.74), trajectories go through walls
- **After finetuning**: Low error (0.13), better trajectories
- **Visualizations**: Trajectory plots in `results/` folder

## Next Steps

- Read `README.md` for full documentation
- Check `docs/blog.md` for technical details
- See `RESULTS.md` for analysis
- Try `test_improvements.py` for advanced planners

## Troubleshooting

**Import errors?**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Out of memory?**
- Reduce `--n_trajectories` to 200
- Reduce `--batch_size` to 32
- Use CPU instead of GPU

**Slow training?**
- Reduce `--epochs` to 10
- Use smaller `--hidden_dim` (64 instead of 128)

