#!/bin/bash
# Evaluation Script for Small Model Ablation Study

echo "=========================================="
echo "Small Model Ablation Study Evaluation"
echo "=========================================="

# Test set path
TESTSET="dns/datasets/test_set/synthetic/no_reverb/"

# Create results directory
mkdir -p results_small

# Experiment names
experiments=(
    "exp0-baseline"
    "exp1-fixed-time"
    #"exp2-fixed-freq"
    #"exp3-fixed-both"
    "exp4-adaptive-replace"
    "exp5-adaptive-add"
    "exp6-adaptive-full"
)

# Evaluate each experiment
for exp in "${experiments[@]}"; do
    echo ""
    echo "Evaluating DNS-small-${exp}..."
    
    # Step 1: Generate denoised audio
    echo "  [1/2] Denoising test set..."
    python denoise.py -c configs/DNS-small-${exp}.json --ckpt_iter 30000
    
    # Step 2: Evaluate metrics
    echo "  [2/2] Computing PESQ and STOI..."
    python python_eval.py \
        -d dns \
        -e exp/DNS-small-${exp}/speech/30k/ \
        -t ${TESTSET} \
        > results_small/${exp}_eval.log 2>&1
    
    # Extract and display results
    echo "  Results:"
    grep "pesq_wb\|pesq_nb\|stoi" results_small/${exp}_eval.log || echo "  (evaluation in progress or failed)"
    
done

echo ""
echo "=========================================="
echo "Generating comparison table..."
echo "=========================================="

# Create comparison table
cat > results_small/ablation_table.txt << 'EOF'
Small Model Ablation Study Results
===================================

Exp | Method                      | PESQ-WB | PESQ-NB | STOI  | Δ from Baseline | Training Time
----|----------------------------|---------|---------|-------|-----------------|---------------
EOF

# Extract results for each experiment
for i in "${!experiments[@]}"; do
    exp="${experiments[$i]}"
    
    # Extract metrics from log file
    if [ -f "results_small/${exp}_eval.log" ]; then
        pesq_wb=$(grep "pesq_wb" results_small/${exp}_eval.log | grep -oP '\d+\.\d+' | head -1)
        pesq_nb=$(grep "pesq_nb" results_small/${exp}_eval.log | grep -oP '\d+\.\d+' | head -1)
        stoi=$(grep "stoi" results_small/${exp}_eval.log | grep -oP '0\.\d+' | head -1)
        
        # Calculate improvement over baseline
        if [ $i -eq 0 ]; then
            baseline_pesq=$pesq_wb
            delta="  -   "
        else
            if [ ! -z "$pesq_wb" ] && [ ! -z "$baseline_pesq" ]; then
                delta=$(echo "$pesq_wb - $baseline_pesq" | bc -l | xargs printf "+%.2f")
            else
                delta="  N/A "
            fi
        fi
        
        # Format experiment name
        case $exp in
            "exp0-baseline") method="CleanUNet-Small (Baseline)";;
            "exp1-fixed-time") method="+ Fixed VUV (Time)";;
            "exp2-fixed-freq") method="+ Fixed VUV (Freq)";;
            "exp3-fixed-both") method="+ Fixed VUV (Both)";;
            "exp4-adaptive-replace") method="+ Adaptive VUV (Replace)";;
            "exp5-adaptive-add") method="+ Adaptive VUV (Add)";;
            "exp6-adaptive-full") method="+ Adaptive VUV (Full)";;
        esac
        
        printf " %d  | %-26s | %7s | %7s | %5s | %s | ~6h\n" \
            "$i" "$method" "${pesq_wb:-N/A}" "${pesq_nb:-N/A}" "${stoi:-N/A}" "$delta" \
            >> results_small/ablation_table.txt
    else
        echo " $i  | ${exp} | ERROR: Log file not found" >> results_small/ablation_table.txt
    fi
done

echo "" >> results_small/ablation_table.txt
echo "Notes:" >> results_small/ablation_table.txt
echo "  - Model size: ~2.5M parameters (vs 40M for large model)" >> results_small/ablation_table.txt
echo "  - Training iterations: 30,000 (vs 250,000 for large model)" >> results_small/ablation_table.txt
echo "  - Purpose: Quick validation of method effectiveness" >> results_small/ablation_table.txt
echo "" >> results_small/ablation_table.txt
echo "Results saved to results_small/ablation_table.txt"

# Display the table
cat results_small/ablation_table.txt

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
echo ""
echo "Files generated:"
echo "  - results_small/*_eval.log        : Detailed metrics"
echo "  - results_small/ablation_table.txt : Comparison table"
echo "  - exp/DNS-small-*/speech/30k/     : Denoised audio"
echo ""
echo "Key Question: Does adaptive VUV show improvement trend?"
echo "  If Exp-4/6 > Exp-3 ✓ : Proceed to large model training"
echo "  If Exp-4/6 ≤ Exp-3 ✗ : Debug or adjust hyperparameters"
