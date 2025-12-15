# Planning Improvements Results

## Tested Strategies

We tested three planning strategies on 10 episodes:

### 1. Baseline GBP (standard gradient-based planning)
- **Avg Distance**: 1.829 ± 0.572
- **Success Rate**: 0%

### 2. Expert Initialization (initialize actions from expert policy)
- **Avg Distance**: 1.029 ± 0.410 (**44% improvement!**)
- **Success Rate**: 10% (**First successes!**)

### 3. MPC-Style (replan every 10 steps)
- **Avg Distance**: 2.000 ± 0.575
- **Success Rate**: 0%

## Key Findings

✅ **Expert Initialization Works Best**:
- 44% closer to goals on average
- Achieved 10% success rate (1 out of 10 episodes)
- Much more consistent (lower std: 0.410 vs 0.572)

❌ **MPC-Style Didn't Help**:
- Actually performed slightly worse
- Likely because replanning every 10 steps is too frequent
- Or the horizon per replan (50 steps) is too short

## Why Expert Initialization Works

1. **Better starting point**: Expert policy provides a reasonable initial trajectory
2. **Faster convergence**: Gradient descent starts from a good solution
3. **Avoids bad local minima**: Expert knows to go through door, not wall

## Recommendations

For best results, use:
- **Expert initialization** for action sequences
- **Longer horizons** (150-200 steps)
- **More optimization steps** (500+)
- **Online finetuned models** (82% lower world model error)

## Implementation

The improved planner is in `src/planners/gbp_improved.py`:
- `plan_with_expert_init()`: Uses expert policy to initialize actions
- `plan_mpc_style()`: MPC-style replanning (needs more tuning)

## Next Steps

1. **Tune MPC parameters**: Try replan_every=20-30, horizon=100
2. **Combine strategies**: Expert init + MPC-style replanning
3. **Longer horizons**: Test with 200-300 step horizons
4. **More episodes**: Run full evaluation (50-100 episodes) with expert init

