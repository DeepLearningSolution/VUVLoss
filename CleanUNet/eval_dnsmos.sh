#!/bin/bash
# DNS-MOS Evaluation Script
# This script computes DNS-MOS (SIG/BAK/OVRL) scores for enhanced audio
# Prerequisite: Audio files must already be generated (run eval_pesq_stoi.sh first)

echo "=========================================="
echo "DNS-MOS Evaluation"
echo "=========================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cleanunet

# Create results directory
mkdir -p results_small/dnsmos

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

# Check if DNSMOS is properly set up
if [ ! -f "dnsmos_models/sig_bak_ovr.onnx" ]; then
    echo "❌ Error: DNSMOS model not found!"
    echo "Expected location: dnsmos_models/sig_bak_ovr.onnx"
    echo ""
    echo "To download the model, run:"
    echo "  bash setup_dnsmos.sh"
    echo ""
    echo "Or manually download from:"
    echo "  https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
    exit 1
fi

echo "✓ DNSMOS model found"
echo ""

# Evaluate each experiment
for exp in "${experiments[@]}"; do
    echo "=========================================="
    echo "Evaluating DNS-small-${exp}..."
    echo "=========================================="
    
    # Audio directory
    AUDIO_DIR="exp/DNS-small-${exp}/speech/30k/"
    
    if [ ! -d "$AUDIO_DIR" ]; then
        echo "  ⚠️  Audio directory not found: $AUDIO_DIR"
        echo "  Please run eval_pesq_stoi.sh first to generate audio files."
        continue
    fi
    
    # Count audio files
    num_files=$(ls -1 "$AUDIO_DIR"/*.wav 2>/dev/null | wc -l)
    if [ "$num_files" -eq 0 ]; then
        echo "  ⚠️  No audio files found in: $AUDIO_DIR"
        continue
    fi
    
    echo "  Found $num_files audio files"
    echo ""
    
    # Compute DNS-MOS
    python dnsmos_eval.py "$AUDIO_DIR" \
        -o "results_small/dnsmos/${exp}_dnsmos.txt"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "  ✓ DNS-MOS results saved to: results_small/dnsmos/${exp}_dnsmos.txt"
        
        # Display summary
        if [ -f "results_small/dnsmos/${exp}_dnsmos.txt" ]; then
            echo ""
            echo "  📊 DNS-MOS Summary:"
            grep -E "^SIG:|^BAK:|^OVRL:" "results_small/dnsmos/${exp}_dnsmos.txt" | sed 's/^/    /'
        fi
    else
        echo "  ✗ DNS-MOS evaluation failed for ${exp}"
    fi
    
    echo ""
done

echo "=========================================="
echo "Generating DNS-MOS Summary Table"
echo "=========================================="

# Generate DNS-MOS comparison table
cat > results_small/dnsmos_table.txt << 'TABLE_HEADER'
DNS-MOS Evaluation Results
==========================

Exp | Method                      | SIG   | BAK   | OVRL  | Δ OVRL  | Δ BAK
----|----------------------------|-------|-------|-------|---------|--------
TABLE_HEADER

baseline_ovrl=""
baseline_bak=""
for i in "${!experiments[@]}"; do
    exp="${experiments[$i]}"
    
    # Extract DNS-MOS scores
    if [ -f "results_small/dnsmos/${exp}_dnsmos.txt" ]; then
        sig=$(grep "^SIG:" results_small/dnsmos/${exp}_dnsmos.txt | grep -oP '\d+\.\d+' | head -1)
        bak=$(grep "^BAK:" results_small/dnsmos/${exp}_dnsmos.txt | grep -oP '\d+\.\d+' | head -1)
        ovrl=$(grep "^OVRL:" results_small/dnsmos/${exp}_dnsmos.txt | grep -oP '\d+\.\d+' | head -1)
        
        # Calculate improvement over baseline
        if [ $i -eq 0 ]; then
            baseline_ovrl=$ovrl
            baseline_bak=$bak
            delta_ovrl="   -   "
            delta_bak="   -   "
        else
            if [ ! -z "$ovrl" ] && [ ! -z "$baseline_ovrl" ]; then
                delta_ovrl=$(echo "$ovrl - $baseline_ovrl" | bc -l | xargs printf "%+.3f")
            else
                delta_ovrl="  N/A  "
            fi
            
            if [ ! -z "$bak" ] && [ ! -z "$baseline_bak" ]; then
                delta_bak=$(echo "$bak - $baseline_bak" | bc -l | xargs printf "%+.3f")
            else
                delta_bak="  N/A  "
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
        
        printf " %d  | %-26s | %5s | %5s | %5s | %-7s | %-7s\n" \
            "$i" "$method" "${sig:- N/A}" "${bak:- N/A}" "${ovrl:- N/A}" "$delta_ovrl" "$delta_bak" \
            >> results_small/dnsmos_table.txt
    else
        case $exp in
            "exp0-baseline") method="CleanUNet-Small (Baseline)";;
            "exp1-fixed-time") method="+ Fixed VUV (Time)";;
            "exp2-fixed-freq") method="+ Fixed VUV (Freq)";;
            "exp3-fixed-both") method="+ Fixed VUV (Both)";;
            "exp4-adaptive-replace") method="+ Adaptive VUV (Replace)";;
            "exp5-adaptive-add") method="+ Adaptive VUV (Add)";;
            "exp6-adaptive-full") method="+ Adaptive VUV (Full)";;
        esac
        printf " %d  | %-26s | %5s | %5s | %5s | %-7s | %-7s\n" \
            "$i" "$method" " N/A" " N/A" " N/A" "  N/A  " "  N/A  " \
            >> results_small/dnsmos_table.txt
    fi
done

echo "" >> results_small/dnsmos_table.txt
echo "DNS-MOS Metrics Explanation:" >> results_small/dnsmos_table.txt
echo "  - SIG:  Signal Quality (1-5, higher is better)" >> results_small/dnsmos_table.txt
echo "          Measures speech distortion and clarity" >> results_small/dnsmos_table.txt
echo "  - BAK:  Background Quality (1-5, higher is better)" >> results_small/dnsmos_table.txt
echo "          Measures noise suppression effectiveness" >> results_small/dnsmos_table.txt
echo "  - OVRL: Overall Quality (1-5, higher is better)" >> results_small/dnsmos_table.txt
echo "          Combined assessment of speech quality" >> results_small/dnsmos_table.txt
echo "" >> results_small/dnsmos_table.txt
echo "Quality Guidelines:" >> results_small/dnsmos_table.txt
echo "  4.5-5.0: Excellent" >> results_small/dnsmos_table.txt
echo "  4.0-4.5: Good" >> results_small/dnsmos_table.txt
echo "  3.5-4.0: Fair" >> results_small/dnsmos_table.txt
echo "  3.0-3.5: Poor" >> results_small/dnsmos_table.txt
echo "  < 3.0:   Bad" >> results_small/dnsmos_table.txt

# Display the table
echo ""
cat results_small/dnsmos_table.txt

echo ""
echo "=========================================="
echo "DNS-MOS Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  📊 results_small/dnsmos_table.txt"
echo "  📈 results_small/dnsmos/*_dnsmos.txt"
echo ""
echo "Detailed per-file results available in:"
echo "  results_small/dnsmos/exp*_dnsmos.txt"
echo ""
