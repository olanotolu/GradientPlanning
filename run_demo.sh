#!/bin/bash
# Quick visual demo script
# Runs the full pipeline and opens visualizations

set -e

echo "🚀 Gradient Planning Visual Demo"
echo "=================================="
echo ""

# Check if data exists
if [ ! -f "data/expert_data.npz" ]; then
    echo "📊 Generating expert data..."
    python src/data/make_expert_data.py --n_trajectories 500
    echo "✓ Data generated"
    echo ""
fi

# Check if baseline model exists
if [ ! -f "checkpoints/baseline_best.pt" ]; then
    echo "🎓 Training baseline model..."
    python src/train/train_baseline.py --epochs 20
    echo "✓ Baseline model trained"
    echo ""
fi

# Run demo visualization
echo "🎨 Running demo visualization..."
python demo.py
echo "✓ Demo complete"
echo ""

# Open results
if command -v open &> /dev/null; then
    echo "📂 Opening visualization results..."
    open results/demo_comparison.png
    open results/demo_model_vs_reality.png
elif command -v xdg-open &> /dev/null; then
    xdg-open results/demo_comparison.png
    xdg-open results/demo_model_vs_reality.png
else
    echo "📂 View results in: results/demo_*.png"
fi

echo ""
echo "✅ Demo complete! Check results/ folder for visualizations."

