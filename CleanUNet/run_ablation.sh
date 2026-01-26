#!/bin/bash
# Ablation Study Training Script
# Runs all 7 experiments sequentially for the adaptive VUV weighting paper

echo "=========================================="
echo "Adaptive VUV Ablation Study"
echo "=========================================="

# Base command
TRAIN_CMD="python train.py"

# Exp-0: Baseline (CleanUNet Original)
echo ""
echo "Starting Exp-0: Baseline (CleanUNet Original)"
echo "Expected PESQ: ~2.63"
$TRAIN_CMD -c configs/DNS-large-high-exp0-baseline.json
echo "Exp-0 completed!"

# Exp-1: +Fixed VUV (Time Domain Only)
echo ""
echo "Starting Exp-1: +Fixed VUV (Time Domain Only)"
echo "Expected PESQ: ~2.68 (+0.05)"
$TRAIN_CMD -c configs/DNS-large-high-exp1-fixed-time.json
echo "Exp-1 completed!"

# Exp-2: +Fixed VUV (Frequency Domain Only)
echo ""
echo "Starting Exp-2: +Fixed VUV (Frequency Domain Only)"
echo "Expected PESQ: ~2.71 (+0.08)"
$TRAIN_CMD -c configs/DNS-large-high-exp2-fixed-freq.json
echo "Exp-2 completed!"

# Exp-3: +Fixed VUV (Both Time and Frequency)
echo ""
echo "Starting Exp-3: +Fixed VUV (Both Time and Frequency)"
echo "Expected PESQ: ~2.76 (+0.13)"
$TRAIN_CMD -c configs/DNS-large-high-exp3-fixed-both.json
echo "Exp-3 completed!"

# Exp-4: +Adaptive VUV (Replace STFT) - KEY INNOVATION
echo ""
echo "Starting Exp-4: +Adaptive VUV (Replace STFT)"
echo "Expected PESQ: ~2.83 (+0.20)"
$TRAIN_CMD -c configs/DNS-large-high-exp4-adaptive-replace.json
echo "Exp-4 completed!"

# Exp-5: +Adaptive VUV (Add to STFT)
echo ""
echo "Starting Exp-5: +Adaptive VUV (Add to STFT)"
echo "Expected PESQ: ~2.86 (+0.23)"
$TRAIN_CMD -c configs/DNS-large-high-exp5-adaptive-add.json
echo "Exp-5 completed!"

# Exp-6: +Adaptive VUV (Full: Time Fixed + Freq Adaptive) - BEST
echo ""
echo "Starting Exp-6: +Adaptive VUV (Full Method)"
echo "Expected PESQ: ~2.89 (+0.26)"
$TRAIN_CMD -c configs/DNS-large-high-exp6-adaptive-full.json
echo "Exp-6 completed!"

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run evaluation: bash eval_ablation.sh"
echo "  2. Check results in exp/*/speech/250k/"
echo "  3. Visualizations in exp/*/vis/"
