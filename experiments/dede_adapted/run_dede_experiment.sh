#!/bin/bash

# Run complete DeDe-Adapted experiment pipeline
# Train on Exp1 baseline, detect on Exp3 GAN attack

set -e  # Exit on error

echo "=========================================="
echo "  DeDe-Adapted Experiment Pipeline"
echo "  Adapted from DeDe (CVPR 2025)"
echo "=========================================="
echo ""

# Configuration
CLEAN_DATA="datasets/splits/raw_scaled/exp1_baseline"
ADV_DATA="datasets/splits/raw_scaled/exp3_gan_attack"
MODEL_DIR="experiments/dede_adapted/models"
RESULTS_DIR="experiments/dede_adapted/results"

# Hyperparameters
EPOCHS=100
BATCH_SIZE=128
MASK_RATIO=0.5
LATENT_DIM=64
LEARNING_RATE=0.001
THRESHOLD_PCT=95

echo "Configuration:"
echo "  Clean data:     $CLEAN_DATA"
echo "  Adversarial:    $ADV_DATA"
echo "  Model dir:      $MODEL_DIR"
echo "  Results dir:    $RESULTS_DIR"
echo ""
echo "Hyperparameters:"
echo "  Epochs:         $EPOCHS"
echo "  Batch size:     $BATCH_SIZE"
echo "  Mask ratio:     $MASK_RATIO"
echo "  Latent dim:     $LATENT_DIM"
echo "  Learning rate:  $LEARNING_RATE"
echo "  Threshold:      ${THRESHOLD_PCT}th percentile"
echo ""

# Check if data exists
if [ ! -d "$CLEAN_DATA" ]; then
    echo "❌ Error: Clean data not found: $CLEAN_DATA"
    echo ""
    echo "Please run baseline experiment first:"
    echo "  python experiments/raw_scaled/exp1_baseline_raw.py"
    exit 1
fi

if [ ! -d "$ADV_DATA" ]; then
    echo "❌ Error: Adversarial data not found: $ADV_DATA"
    echo ""
    echo "Please generate adversarial samples first:"
    echo "  python generate_adversarial_raw_to_raw.py"
    exit 1
fi

echo "✅ Data directories exist"
echo ""

# ============================================
# STEP 1: Train DeDe-Adapted
# ============================================
echo "=========================================="
echo "[1/2] TRAINING DeDe-Adapted Model"
echo "=========================================="
echo ""

python experiments/dede_adapted/train_dede.py \
    --data-dir "$CLEAN_DATA" \
    --output-dir "$MODEL_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --mask-ratio $MASK_RATIO \
    --latent-dim $LATENT_DIM \
    --learning-rate $LEARNING_RATE

echo ""
echo "✅ Training completed!"
echo ""

# ============================================
# STEP 2: Detect Adversarial Samples
# ============================================
echo "=========================================="
echo "[2/2] DETECTING Adversarial Samples"
echo "=========================================="
echo ""

python experiments/dede_adapted/detect_adversarial.py \
    --model-dir "$MODEL_DIR" \
    --clean-data "$CLEAN_DATA" \
    --adv-data "$ADV_DATA" \
    --output-dir "$RESULTS_DIR" \
    --threshold-percentile $THRESHOLD_PCT

echo ""
echo "✅ Detection completed!"
echo ""

# ============================================
# Summary
# ============================================
echo "=========================================="
echo "  EXPERIMENT COMPLETED!"
echo "=========================================="
echo ""
echo "📊 Results saved in:"
echo "  Model:   $MODEL_DIR/"
echo "  Results: $RESULTS_DIR/"
echo ""
echo "📁 Directory structure:"
echo ""
echo "experiments/dede_adapted/"
echo "├── models/"
echo "│   ├── best_model.h5"
echo "│   ├── dede_final.h5"
echo "│   ├── training_history.png"
echo "│   └── training_config.json"
echo "└── results/"
echo "    ├── detection_results.json"
echo "    ├── detection_summary.csv"
echo "    ├── error_distributions.png"
echo "    ├── roc_curve.png"
echo "    └── confusion_matrix.png"
echo ""
echo "🎯 Key Files to Check:"
echo "  1. $RESULTS_DIR/detection_summary.csv"
echo "     → Accuracy, Precision, Recall, F1, AUC"
echo ""
echo "  2. $RESULTS_DIR/error_distributions.png"
echo "     → Clean vs Adversarial error comparison"
echo ""
echo "  3. $RESULTS_DIR/roc_curve.png"
echo "     → ROC curve & AUC score"
echo ""
echo "💡 Next steps:"
echo "  - Review detection performance in detection_summary.csv"
echo "  - Analyze error distributions in visualizations"
echo "  - Compare with baseline models (Exp3 results)"
echo "  - Try different hyperparameters (mask_ratio, latent_dim)"
echo ""
echo "=========================================="
echo "  DeDe-Adapted Experiment Finished! ✨"
echo "=========================================="
