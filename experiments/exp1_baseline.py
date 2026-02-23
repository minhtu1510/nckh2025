"""
Thực nghiệm 1: Baseline (RAW Features - Fair Comparison)
Train models on clean 50-dim selected features
Same preprocessing as LATENT, but WITHOUT encoding layer
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# Use RAW data generated from same pipeline as LATENT (for fair comparison)
DATA_DIR = BASE_DIR / "datasets" / "splits" / "3.0_raw_from_latent" / "exp1_baseline"
OUTPUT_DIR = BASE_DIR / "results" / "raw_fair" / "exp1_baseline"

def main():
    print("\n🚀 THỰC NGHIỆM 1: BASELINE (RAW - 50 features)")
    print("   Pipeline: Raw → Filter → Scale → Select (50) → STOP")
    print("   (Same preprocessing as LATENT, WITHOUT encoding)")
    print("   Train: 300,000 samples (200k benign + 100k malicious)")
    print("   Test:  100,000 samples\n")
    
    # Check if data exists
    if not DATA_DIR.exists():
        print(f"❌ Error: Fair comparison RAW data not found at {DATA_DIR}")
        print("\n   Hãy chạy:")
        print("   python pipelines/preprocessing/prepare_latent_data.py")
        print("\n   → Script này sẽ tạo CẢ RAW (50 dims) VÀ LATENT (64 dims)")
        sys.exit(1)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    python_exec = "/home/mtu/miniconda3/envs/fl-fedavg/bin/python"
    cmd = [
        python_exec,
        str(BASE_DIR / "run_model_evaluation.py"),
        "--data-dir", str(DATA_DIR),
        "--output-dir", str(OUTPUT_DIR),
        "--exp-name", "EXP1: BASELINE (RAW - 50 dims, fair comparison)"
    ]
    
    subprocess.run(cmd, check=True)
    
    print(f"\n{'='*80}")
    print("✅ HOÀN THÀNH EXP1 RAW!")
    print(f"{'='*80}")
    print(f"\n📊 Results: {OUTPUT_DIR}")
    print(f"\n💡 Compare with LATENT:")
    print(f"   LATENT results: results/latent/exp1_baseline_latent/")
    print(f"   RAW (this):     {OUTPUT_DIR}/\n")

if __name__ == '__main__':
    main()
