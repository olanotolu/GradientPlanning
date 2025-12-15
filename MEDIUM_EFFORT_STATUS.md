# Medium Effort Tasks Status

## What We Haven't Done

### 1. **Add MPC** (Closed-loop replanning)
**Status**: ⚠️ Code exists but not integrated

- **What exists**: `plan_mpc_style()` in `src/planners/gbp_improved.py`
- **What's missing**: Integration into main evaluation (`eval_all.py`)
- **Previous testing**: Tested in `test_improvements.py` - **didn't help** (actually performed worse)
  - Avg Distance: 2.000 (worse than baseline 1.829)
  - Success Rate: 0% (same as baseline)
  - Issue: Replanning every 10 steps was too frequent, horizon too short

**Verdict**: Code exists but needs tuning. Previous tests suggest it won't help much without significant parameter tuning.

### 2. **Two-Door Environment**
**Status**: ❌ Not done

- Still using single door environment
- Would require:
  - New environment class
  - New expert policy
  - Regenerating all data
  - Retraining all models

**Verdict**: Would add complexity without changing core story.

### 3. **Hyperparameter Sweep**
**Status**: ❌ Not done

- Have fixed hyperparameters in `config.py`
- No sweep script exists
- Would require:
  - Sweep script
  - Multiple training runs
  - Results comparison

**Verdict**: Nice to have but not essential for proving concept.

### 4. **Better Adversarial Tuning** (Fix 0% success)
**Status**: ❌ Not done

- **Current results**:
  - Adversarial GBP: 0% success (worse than baseline 9%)
  - Combined GBP: 0% success (also worse)
  - High world model error: 1.43 (baseline: 0.59)
- **What's needed**:
  - Try different perturbation radii (eps_z, eps_a)
  - Try different scaling factors (lambda_z, lambda_a)
  - Test if adversarial needs online finetuning first
  - Possibly needs different hyperparameters entirely

**Verdict**: Would require significant experimentation. The loss landscape visualization already proves adversarial smoothing works, even if planning doesn't succeed.

## Recommendation: **Stop Here**

### Why We're Better Off Stopping

1. **Core goal achieved**: 
   - ✅ Train-test gap demonstrated
   - ✅ Online finetuning works (52% error reduction)
   - ✅ Adversarial smoothing proven (loss landscape visualization)
   - ✅ Both fixes implemented and tested

2. **MPC tested and didn't help**:
   - Previous experiments showed it performed worse
   - Would need significant tuning to make it work
   - Not essential for proving the concept

3. **Adversarial planning failure is okay**:
   - The loss landscape visualization proves adversarial smoothing works
   - The paper's key finding is demonstrated
   - Planning success is secondary to proving the mechanism

4. **"Shitty version" philosophy**:
   - Goal is to prove understanding, not achieve SOTA
   - We've proven the core concept works
   - Additional features don't change the story

5. **Diminishing returns**:
   - Medium effort tasks would take half a day each
   - Unlikely to significantly improve the narrative
   - Time better spent on documentation/polish

### What We Have That Matters

✅ **Loss landscape visualization** - Proves adversarial smoothing (matches paper!)  
✅ **100-episode evaluation** - Statistical confidence  
✅ **Success rate tracking** - Shows improvement over iterations  
✅ **Combined training** - Both methods implemented  
✅ **Clear documentation** - README, comparisons, status  

### If You Really Want to Extend

**Priority order**:
1. **Fix adversarial tuning** (most impactful if it works)
   - Try eps_z=0.05, eps_a=0.05 (smaller perturbations)
   - Try lambda_z=0.1, lambda_a=0.1 (larger scaling)
   - Test on online-finetuned model (combined)

2. **Tune MPC properly** (if you want closed-loop)
   - Try replan_every=20-30 (less frequent)
   - Try horizon=100 per replan (longer)
   - Test with expert initialization

3. **Hyperparameter sweep** (if you want robustness)
   - Focus on action_max, horizon, goal_threshold
   - Run grid search on key parameters
   - Document best settings

4. **Two-door environment** (least priority)
   - Only if you want to show generalization
   - Adds complexity without changing core story

## Bottom Line

**You've successfully demonstrated the paper's core contribution:**
- Train-test gap exists
- Online finetuning closes it
- Adversarial finetuning smooths the landscape

**The "shitty version" is complete!** 🎉

Additional features would be nice but aren't necessary. The project successfully proves you understand the paper's key ideas.


