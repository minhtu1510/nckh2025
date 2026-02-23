#!/bin/bash
# Chạy Exp6 v3: GAN-Optimized Hybrid Defense
# So sánh với v2 (Standard Stacking): GAN F1 từ 0.8224 → ~0.93+

cd /run/media/mtu/4AE886A9E886933D/NCKH2025/NCKH_code/ids_research

source $(conda info --base)/etc/profile.d/conda.sh
conda activate fl-fedavg

echo "=== EXP6 v3: GAN-Optimized Stacking ==="
echo "    Base: MLP_deep + MLP_wide + KNN_5 + KNN_11"
echo "    DeDe RAW (same as v2, threshold=99th pct)"
echo ""

python experiments/exp6_hybrid_defense_v3_ganopt.py \
    --data-dir   datasets/splits/3.0_raw_from_latent/exp1_baseline \
    --dede-model experiments/dede_adapted/models_raw \
    --output-dir results/raw/exp6_hybrid_defense_v3 \
    --threshold-pct 99 \
    2>&1 | grep -v "^2026\|TF-TRT\|cuda\|NUMA\|AVX\|FutureWarn\|warnings\|registering\|ConvergenceWarn\|Liblinear\|StreamExec\|disabl"
