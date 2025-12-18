# MPS Acceleration for Apple Silicon

## ✅ What Was Done

All training scripts have been updated to automatically use **MPS (Metal Performance Shaders)** on Apple Silicon Macs, which provides GPU acceleration similar to CUDA on NVIDIA GPUs.

### Changes Made:

1. **Created `src/utils/device.py`** - Utility functions for device selection:
   - `get_device()` - Automatically selects best device: CUDA > MPS > CPU
   - `set_seed()` - Handles seeding for all device types

2. **Updated All Training Scripts**:
   - `src/train/train_baseline.py`
   - `src/train/train_online.py`
   - `src/train/train_adversarial.py`
   - `src/train/train_combined.py`
   - `src/train/train_real_robot.py`

3. **Updated DINOv2 Encoder** - Now properly handles MPS device strings

## 🚀 Performance Improvement

**Before**: Training on CPU (very slow, ~12 hours for your run)  
**After**: Training on MPS (should be **5-10x faster** on M4)

### Expected Speedup:
- **DINOv2 encoding**: ~5-10x faster
- **Model training**: ~5-10x faster
- **Overall training time**: Should reduce from 12 hours to **1-2 hours**

## 📝 Usage

No changes needed! The scripts automatically detect and use MPS:

```bash
# This will now use MPS automatically
python src/train/train_online.py --use_images --task pusht \
  --checkpoint checkpoints/baseline_best.pt \
  --data_path data/expert_data.npz \
  --n_iterations 5 --n_rollouts_per_iter 10 \
  --epochs_per_iter 3 --image_resolution 448
```

The output will show:
```
Using device: mps
```

## ⚠️ Known Limitations

1. **Some operations may fall back to CPU**: MPS doesn't support all PyTorch operations yet. If you see warnings or errors, they're usually non-critical.

2. **DINOv2 on MPS**: Vision Transformers work on MPS, but some operations might be slightly slower than CUDA. Still much faster than CPU!

3. **Memory**: MPS uses unified memory, so you have access to all your RAM for GPU operations.

## 🧪 Testing

Run the test script to verify MPS is working:

```bash
python test_mps.py
```

You should see:
```
✓ Selected device: mps
✓ All tests passed! MPS acceleration is working.
```

## 📊 Next Steps

Your training should now be **much faster**. The online finetuning that was taking 12 hours should complete in 1-2 hours on your M4 Mac!


