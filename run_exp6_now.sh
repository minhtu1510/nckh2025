#!/bin/bash
# Chạy Exp6 Latent: Hybrid Defense — Latent Version

cd /run/media/mtu/4AE886A9E886933D/NCKH2025/NCKH_code/ids_research

source $(conda info --base)/etc/profile.d/conda.sh
conda activate fl-fedavg

echo "=== EXP6 LATENT: DeDe Latent (64-dim) + Stacking ==="
python experiments/latent/exp6_hybrid_defense_latent.py \
    --latent-dir   datasets/splits/3.1_latent \
    --raw-dir      datasets/splits/3.0_raw_from_latent \
    --dede-latent  experiments/dede_adapted/models_latent \
    --dede-raw     experiments/dede_adapted/models_raw \
    --output-dir   results/latent/exp6_hybrid_defense_latent \
    --threshold-pct 99 \
    --test-all-attacks \
    2>&1 | grep -v "^2026\|TF-TRT\|cuda\|NUMA\|AVX\|FutureWarn\|warnings\|dual\|StreamExec\|disabl\|registering\|ConvergenceWarning\|Liblinear"
