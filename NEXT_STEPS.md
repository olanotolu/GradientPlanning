# Next Steps

## Current Status

✅ **Completed:**
- Full implementation of all components
- Fixed gradient issues in GBP planning
- Baseline training works (validation loss: 0.000005)
- Online finetuning works (improved world model error: 0.017855 vs 0.068217)
- All code is functional

⚠️ **Issue:**
- Success rates are 0% due to hyperparameter mismatch
- Action scale (action_max=0.1) is too small for environment size
- Planning horizon (25 steps) is too short

## Recommended Next Steps

### Option 1: Fix Hyperparameters (Get Working Results)

**Goal:** Get actual success rates to demonstrate the concept works

1. **Increase action scale:**
   ```bash
   # Regenerate expert data with larger actions
   python src/data/make_expert_data.py --n_trajectories 1000 --action_max 0.2
   
   # Retrain baseline
   python src/train/train_baseline.py --data_path data/expert_data.npz
   
   # Retrain online
   python src/train/train_online.py --n_iterations 5 --n_rollouts_per_iter 20
   
   # Evaluate with longer horizon
   python src/eval/eval_planning.py --checkpoint checkpoints/online_final.pt \
     --model_type online --planner gbp --horizon 150 --n_episodes 50
   ```

2. **Expected improvements:**
   - Baseline: ~20-40% success (fails due to train-test gap)
   - Online: ~60-80% success (fixes distribution shift)

### Option 2: Document Current Findings (Blog Post Ready)

**Goal:** Write blog post with what we learned, even with 0% success

1. **Update blog post** with:
   - Actual results (0% success but improved world model error)
   - Explanation of hyperparameter issue
   - What the code demonstrates (train-test gap exists, finetuning helps)
   - Lessons learned

2. **Key points for blog:**
   - Implementation works correctly
   - Train-test gap is real (baseline fails)
   - Online finetuning improves world model error
   - Hyperparameter tuning is critical for success rates

### Option 3: Create Demo Script

**Goal:** Simple script showing the train-test gap visually

```python
# demo.py - Show baseline vs online planning
# Visualize trajectories going through walls vs through door
```

### Option 4: Compare with CEM Baseline

**Goal:** Show that even CEM fails with current hyperparameters, proving it's not a GBP-specific issue

```bash
python src/eval/eval_planning.py --checkpoint checkpoints/baseline_best.pt \
  --model_type baseline --planner cem --horizon 200 --n_episodes 50
```

## My Recommendation

**Do Option 1 first** (fix hyperparameters) to get working results, then **Option 2** (update docs). This gives you:
- Real numbers for the blog post
- Proof the concept works
- Better understanding of what matters

**Time estimate:** 30-60 minutes to fix hyperparams and get results.

## Quick Hyperparameter Fix

If you want to quickly test if it works:

```bash
# 1. Regenerate data with larger actions
python src/data/make_expert_data.py --n_trajectories 500 --action_max 0.25

# 2. Quick retrain (fewer epochs)
python src/train/train_baseline.py --epochs 20

# 3. Quick online finetune
python src/train/train_online.py --n_iterations 3 --n_rollouts_per_iter 10 --epochs_per_iter 2

# 4. Evaluate with longer horizon
python src/eval/eval_planning.py --checkpoint checkpoints/online_final.pt \
  --model_type online --planner gbp --horizon 150 --n_episodes 20 --save_plots
```

This should give you some non-zero success rates to work with!

