#!/usr/bin/env python3
"""
Exp4: Ensemble Learning - LATENT Approach
Sử dụng các models đã train trước trên latent features để tạo ensemble classifier.

Ensemble strategies:
1. Soft voting (equal weights): Average probabilities từ tất cả models
2. Hard voting (equal weights): Majority vote từ tất cả models  
3. Weighted soft voting: Weighted average based on F1 scores
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def main():
    print("\n🚀 EXP4: ENSEMBLE LEARNING - LATENT APPROACH")
    print("   Pipeline: Load pre-trained models on latent features")
    print("   Features: 64 dims (32 benign_enc + 32 malicious_enc)")
    print("   Strategy: Soft voting, Hard voting, Weighted voting")
    print("   Models: MLP, SVM, RF, KNN, NB\n")
    
    data_dir = BASE_DIR / "datasets" / "splits" / "3.1_latent" / "exp1_baseline_latent"
    models_dir = BASE_DIR / "models" / "latent" / "exp1_baseline_latent"
    output_dir = BASE_DIR / "results" / "latent" / "exp4_ensemble_latent"
    
    # Check if data exists
    if not data_dir.exists():
        print(f"❌ Error: Data not found at {data_dir}")
        print("\n   Hãy chạy trước:")
        print("   python pipelines/preprocessing/prepare_latent_data.py\n")
        sys.exit(1)
    
    # Check if models exist
    if not models_dir.exists():
        print(f"❌ Error: Pre-trained models not found at {models_dir}")
        print("\n   Hãy chạy trước:")
        print("   python experiments/latent/exp1_baseline_latent.py")
        print("\n   → Script này sẽ train và save models cho ensemble\n")
        sys.exit(1)
    
    # Check required model files
    required_models = ['mlp.h5', 'svm.pkl', 'rf.pkl', 'knn.pkl', 'nb.pkl']
    missing = [f for f in required_models if not (models_dir / f).exists()]
    
    if missing:
        print(f"❌ Missing model files: {missing}")
        print("\n   Hãy chạy trước:")
        print("   python experiments/latent/exp1_baseline_latent.py\n")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"  Running Exp4: Ensemble Learning (LATENT)")
    print(f"{'='*80}\n")
    
    # Run ensemble evaluation
    python_exec = "/home/mtu/miniconda3/envs/fl-fedavg/bin/python"
    cmd = [
        python_exec,
        str(BASE_DIR / "run_ensemble_evaluation.py"),
        "--data-dir", str(data_dir),
        "--models-load-dir", str(models_dir),
        "--output-dir", str(output_dir),
        "--exp-name", "EXP4: ENSEMBLE LEARNING (LATENT - 64 dims)"
    ]
    
    subprocess.run(cmd, check=True)
    
    print(f"\n{'='*80}")
    print("✅ HOÀN THÀNH EXP4 ENSEMBLE!")
    print(f"{'='*80}")
    print(f"\n📊 Results: {output_dir}")
    print(f"\n💡 Compare with individual models:")
    print(f"   Individual results: results/latent/exp1_baseline_latent/")
    print(f"   Ensemble (this):    {output_dir}/")
    print(f"\n   → Check if ensemble improves over best individual model!\n")

if __name__ == '__main__':
    main()
