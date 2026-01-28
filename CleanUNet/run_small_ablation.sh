#!/bin/bash
# Quick Ablation Study with Small Models (2-day validation on single 4090)

echo "=========================================="
echo "Quick Ablation Study with Small Models"
echo "Estimated time: ~6 hours per experiment"
echo "Total: ~42 hours (under 2 days)"
echo "GPU: Using GPU 0 (RTX 4090)"
echo "=========================================="

# Force use of GPU 0 (only GPU available)
export CUDA_VISIBLE_DEVICES=0

# Reduce memory fragmentation for VUV loss experiments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Checking GPU availability..."
nvidia-smi -i 0 --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Create logs directory
mkdir -p logs_small

# Base command
TRAIN_CMD="python train.py"

# Exp-0: Baseline (CleanUNet Original)
echo ""
echo "[1/7] Exp-0: Baseline (~6h)"
echo "Started at: $(date)"
#$TRAIN_CMD -c configs/DNS-small-exp0-baseline.json 2>&1 | tee logs_small/exp0.log
echo "Completed at: $(date)"

# Exp-1: +Fixed VUV (Time Domain Only)
echo ""
echo "[2/7] Exp-1: +Fixed VUV (Time) (~6h)"
echo "Started at: $(date)"
#$TRAIN_CMD -c configs/DNS-small-exp1-fixed-time.json 2>&1 | tee logs_small/exp1.log
echo "Completed at: $(date)"

# Exp-2: +Fixed VUV (Frequency Domain Only)
echo ""
echo "[3/7] Exp-2: +Fixed VUV (Freq) (~6h)"
echo "Started at: $(date)"
$TRAIN_CMD -c configs/DNS-small-exp2-fixed-freq.json 2>&1 | tee logs_small/exp2.log
echo "Completed at: $(date)"

# Exp-3: +Fixed VUV (Both Time and Frequency)
echo ""
echo "[4/7] Exp-3: +Fixed VUV (Both) (~6h)"
echo "Started at: $(date)"
$TRAIN_CMD -c configs/DNS-small-exp3-fixed-both.json 2>&1 | tee logs_small/exp3.log
echo "Completed at: $(date)"

# Exp-4: +Adaptive VUV (Replace STFT) - KEY INNOVATION
echo ""
echo "[5/7] Exp-4: +Adaptive VUV (Replace) (~6h) ★"
echo "Started at: $(date)"
#$TRAIN_CMD -c configs/DNS-small-exp4-adaptive-replace.json 2>&1 | tee logs_small/exp4.log
echo "Completed at: $(date)"

# Exp-5: +Adaptive VUV (Add to STFT)
echo ""
echo "[6/7] Exp-5: +Adaptive VUV (Add) (~6h)"
echo "Started at: $(date)"
#$TRAIN_CMD -c configs/DNS-small-exp5-adaptive-add.json 2>&1 | tee logs_small/exp5.log
echo "Completed at: $(date)"

# Exp-6: +Adaptive VUV (Full: Time Fixed + Freq Adaptive) - BEST
echo ""
echo "[7/7] Exp-6: +Adaptive VUV (Full) (~6h) ★★"
echo "Started at: $(date)"
#$TRAIN_CMD -c configs/DNS-small-exp6-adaptive-full.json 2>&1 | tee logs_small/exp6.log
echo "Completed at: $(date)"

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Finished at: $(date)"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Check logs in logs_small/ directory"
echo "  2. Run evaluation: bash eval_small_ablation.sh"
echo "  3. Compare results in results_small/ablation_table.txt"
