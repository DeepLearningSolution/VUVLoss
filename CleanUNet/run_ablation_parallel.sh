#!/bin/bash
# Parallel Training Script for Ablation Study
# Runs experiments on different GPUs simultaneously

echo "=========================================="
echo "Parallel Ablation Study (Multi-GPU)"
echo "=========================================="

# Check available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"

if [ $NUM_GPUS -lt 4 ]; then
    echo "Warning: Less than 4 GPUs available. Some experiments will run sequentially."
fi

# Run first batch (Exp 0-3) in parallel
echo ""
echo "Running Batch 1: Exp 0-3"
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/DNS-large-high-exp0-baseline.json > logs/exp0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python train.py -c configs/DNS-large-high-exp1-fixed-time.json > logs/exp1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python train.py -c configs/DNS-large-high-exp2-fixed-freq.json > logs/exp2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python train.py -c configs/DNS-large-high-exp3-fixed-both.json > logs/exp3.log 2>&1 &

# Wait for first batch to complete
wait
echo "Batch 1 completed!"

# Run second batch (Exp 4-6) in parallel
echo ""
echo "Running Batch 2: Exp 4-6 (Adaptive methods)"
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/DNS-large-high-exp4-adaptive-replace.json > logs/exp4.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python train.py -c configs/DNS-large-high-exp5-adaptive-add.json > logs/exp5.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python train.py -c configs/DNS-large-high-exp6-adaptive-full.json > logs/exp6.log 2>&1 &

# Wait for second batch to complete
wait
echo "Batch 2 completed!"

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Check logs in logs/ directory"
