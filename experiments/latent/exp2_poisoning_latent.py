"""
Experiment 2: Data Poisoning Attack — LATENT Features

Threat Model (ĐÚNG):
  - Defender retrain trên poisoned latent data (không biết bị tấn công)
  - Test trên clean latent test data
  - Đo F1 giảm bao nhiêu

Data: datasets/splits/3.1_latent/exp2_poisoning/
"""
import sys, numpy as np, pandas as pd, subprocess
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

PYTHON = "/home/mtu/miniconda3/envs/fl-fedavg/bin/python"
RUNNER = str(BASE_DIR / "run_ensemble_evaluation.py")


def main():
    print("\n🚀 EXP2 LATENT: DATA POISONING — Retrain trên poisoned latent data")
    print("   Threat model ĐÚNG: defender train trên data bị nhiễm (không biết)")

    base = BASE_DIR / "datasets/splits/3.1_latent"
    models_dir  = BASE_DIR / "models/latent/exp1_baseline_latent"
    output_base = BASE_DIR / "results/latent/exp2_poisoning"

    if not models_dir.exists():
        print(f"❌ Models not found: {models_dir}")
        print("   Hãy chạy trước: python experiments/latent/exp1_baseline_latent.py")
        sys.exit(1)

    for rate in [5, 10, 15, 50]:
        rate_str = f"{rate:02d}"
        poison_dir = base / f"exp2_poisoning/poison_{rate_str}"
        out_dir    = output_base / f"poison_{rate_str}"

        if not poison_dir.exists():
            print(f"⚠️  Skipping poison_{rate}% — data not found at {poison_dir}")
            continue

        # Load để in thống kê
        X_tp = np.load(poison_dir / 'X_train.npy')
        y_tp = np.load(poison_dir / 'y_train.npy')
        y_cl = np.load(base / 'exp1_baseline_latent/y_train.npy')
        flips = (y_tp != y_cl).sum()

        print(f"\n{'='*70}")
        print(f"  Exp2 Latent — Poison {rate}%  ({flips:,} labels flipped)")
        print(f"  Train trên POISONED latent ({len(X_tp):,} × {X_tp.shape[1]})")
        print(f"  Test  trên CLEAN latent")
        print(f"{'='*70}\n")

        # NOTE: run_ensemble_evaluation.py chạy individual models (MLP,SVM,RF,KNN,NB)
        # với data từ poison_dir (X_train poisoned, X_test clean)
        cmd = [
            PYTHON, RUNNER,
            "--data-dir",       str(poison_dir),
            "--models-load-dir", str(models_dir),
            "--output-dir",      str(out_dir),
            "--retrain",                           # Retrain trên poisoned data!
            "--exp-name", f"EXP2 LATENT: POISONING {rate}% (retrain on poisoned)"
        ]
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n✅ Exp2 Latent hoàn thành!")
    print(f"📁 Results: {output_base}/")


if __name__ == '__main__':
    main()
