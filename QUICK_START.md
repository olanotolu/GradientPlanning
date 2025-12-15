# Quick Start

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gradient-planning.git
cd gradient-planning

# Install dependencies
pip install -r requirements.txt
```

## Visual Demo

```bash
./run_demo.sh
# or
python visual_demo.py
```

## Step by Step

```bash
python src/data/make_expert_data.py --n_trajectories 500
python src/train/train_baseline.py --epochs 20
python visual_demo.py
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

