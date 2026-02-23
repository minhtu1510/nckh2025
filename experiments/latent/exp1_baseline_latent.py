#!/usr/bin/env python3
"""
Exp1 Baseline - LATENT Approach (Dual-Encoder)
Train models on clean dual-view latent features and SAVE for Exp3 reuse

Features: 64 latent dims (32 from benign_enc + 32 from malicious_enc)
Models: MLP, SVM, RF, KNN, NB
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def main():
    print("\n🚀 EXP1: BASELINE - DUAL-ENCODER LATENT APPROACH")
    print("   Pipeline: Raw → Filter → Scale → Select (50) → Dual-Encode (64)")
    print("   Features: [benign_enc(x), malicious_enc(x)] = 64 dims")
    print("   Train: Clean dual-view latent features")
    print("   Test:  Clean dual-view latent features\n")
    
    data_dir = BASE_DIR / "datasets" / "splits" / "3.1_latent" / "exp1_baseline_latent"
    output_dir = BASE_DIR / "results" / "latent" / "exp1_baseline_latent"
    models_save_dir = BASE_DIR / "models" / "latent" / "exp1_baseline_latent"
    
    # Check if data exists
    if not data_dir.exists():
        print(f"❌ Error: Data not found at {data_dir}")
        print("\n   Hãy chạy trước:")
        print("   python pipelines/preprocessing/prepare_latent_data.py\n")
        sys.exit(1)
    
    # Check required files
    required_files = ['X_train.npy', 'y_train.npy', 'X_test.npy', 'y_test.npy']
    missing = [f for f in required_files if not (data_dir / f).exists()]
    
    if missing:
        print(f"❌ Missing files: {missing}")
        print("\n   Hãy chạy trước:")
        print("   python pipelines/preprocessing/prepare_latent_data.py\n")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"  Running Exp1 Baseline (LATENT)")
    print(f"{'='*80}\n")
    
    # Run evaluation with model saving
    # Use fl-fedavg environment to ensure TensorFlow availability
    python_exec = "/home/mtu/miniconda3/envs/fl-fedavg/bin/python"
    cmd = [
        python_exec,
        str(BASE_DIR / "run_model_evaluation.py"),
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
        "--models-save-dir", str(models_save_dir),  # SAVE models for Exp3
        "--exp-name", "EXP1: BASELINE (LATENT - 64 dims dual-encoder)"
    ]
    
    subprocess.run(cmd, check=True)
    
    print(f"\n{'='*80}")
    print("✅ HOÀN THÀNH EXP1 LATENT!")
    print(f"{'='*80}")
    print(f"\n📊 Results: {output_dir}")
    print(f"💾 Models saved: {models_save_dir}")
    print(f"\n   Models will be REUSED in Exp3 for fair comparison!\n")

if __name__ == '__main__':
    main()
