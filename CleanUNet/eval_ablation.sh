#!/bin/bash
# Evaluation Script for Ablation Study
# Evaluates all experiments and generates comparison table

echo "=========================================="
echo "Ablation Study Evaluation"
echo "=========================================="

# Test set path
TESTSET="dns/datasets/test_set/synthetic/no_reverb/"

# Create results directory
mkdir -p results

# Experiment names
experiments=(
    "exp0-baseline"
    "exp1-fixed-time"
    "exp2-fixed-freq"
    "exp3-fixed-both"
    "exp4-adaptive-replace"
    "exp5-adaptive-add"
    "exp6-adaptive-full"
)

# Evaluate each experiment
for exp in "${experiments[@]}"; do
    echo ""
    echo "Evaluating DNS-large-high-${exp}..."
    
    # Step 1: Generate denoised audio
    echo "  [1/2] Denoising test set..."
    python denoise.py -c configs/DNS-large-high-${exp}.json --ckpt_iter 250000
    
    # Step 2: Evaluate metrics
    echo "  [2/2] Computing PESQ and STOI..."
    python python_eval.py \
        -d dns \
        -e exp/DNS-large-high-${exp}/speech/250k/ \
        -t ${TESTSET} \
        > results/${exp}_eval.log 2>&1
    
    # Extract and display results
    echo "  Results:"
    grep "pesq_wb\|pesq_nb\|stoi" results/${exp}_eval.log
    
done

echo ""
echo "=========================================="
echo "Generating comparison table..."
echo "=========================================="

# Create comparison table
cat > results/ablation_table.txt << 'EOF'
Ablation Study Results
======================

Exp | Method                      | PESQ-WB | PESQ-NB | STOI  | Δ from Baseline
----|----------------------------|---------|---------|-------|----------------
EOF

# Extract results for each experiment
for i in "${!experiments[@]}"; do
    exp="${experiments[$i]}"
    
    # Extract metrics from log file
    if [ -f "results/${exp}_eval.log" ]; then
        pesq_wb=$(grep "pesq_wb" results/${exp}_eval.log | grep -oP '\d+\.\d+' | head -1)
        pesq_nb=$(grep "pesq_nb" results/${exp}_eval.log | grep -oP '\d+\.\d+' | head -1)
        stoi=$(grep "stoi" results/${exp}_eval.log | grep -oP '0\.\d+' | head -1)
        
        # Calculate improvement over baseline
        if [ $i -eq 0 ]; then
            baseline_pesq=$pesq_wb
            delta="  -   "
        else
            delta=$(echo "$pesq_wb - $baseline_pesq" | bc -l | xargs printf "+%.2f")
        fi
        
        # Format experiment name
        case $exp in
            "exp0-baseline") method="CleanUNet (Baseline)";;
            "exp1-fixed-time") method="+ Fixed VUV (Time)";;
            "exp2-fixed-freq") method="+ Fixed VUV (Freq)";;
            "exp3-fixed-both") method="+ Fixed VUV (Both)";;
            "exp4-adaptive-replace") method="+ Adaptive VUV (Replace)";;
            "exp5-adaptive-add") method="+ Adaptive VUV (Add)";;
            "exp6-adaptive-full") method="+ Adaptive VUV (Full)";;
        esac
        
        printf " %d  | %-26s | %7s | %7s | %5s | %s\n" \
            "$i" "$method" "$pesq_wb" "$pesq_nb" "$stoi" "$delta" \
            >> results/ablation_table.txt
    else
        echo " $i  | ${exp} | ERROR: Log file not found" >> results/ablation_table.txt
    fi
done

echo "" >> results/ablation_table.txt
echo "Results saved to results/ablation_table.txt"

# Display the table
cat results/ablation_table.txt

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
echo ""
echo "Files generated:"
echo "  - results/*_eval.log      : Detailed metrics for each experiment"
echo "  - results/ablation_table.txt : Comparison table"
echo "  - exp/*/speech/250k/      : Denoised audio files"
