"""
Thực nghiệm 3: GAN Adversarial Attack (RAW - Fair Comparison)
Train on clean 50-dim, test on adversarial 50-dim
Same preprocessing as LATENT, WITHOUT encoding layer
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# Use RAW adversarial data from same pipeline as LATENT
DATA_DIR = BASE_DIR / "datasets" / "splits" / "3.0_raw_from_latent" / "exp3_gan_attack"
OUTPUT_DIR = BASE_DIR / "results" / "raw_fair" / "exp3_gan_attack"

def main():
    print("\n🚀 THỰC NGHIỆM 3: GAN ADVERSARIAL ATTACK (RAW - Fair Comparison)")
    print("   Pipeline: Raw → Filter → Scale → Select (50) → STOP")
    print("   (Same preprocessing as LATENT, WITHOUT encoding)")
    print("   Train: Clean 50-dim features")
    print("   Test:  Adversarial 50-dim features (GAN generated)\n")
    
    # Check if adversarial data exists
    if not DATA_DIR.exists():
        print(f"❌ Error: Adversarial data not found at {DATA_DIR}")
        print("\n   Hãy chạy trước:")
        print("   python pipelines/attacks/generate_adversarial_samples.py")
        print("\n   → Script này sẽ:")
        print("   1. Train GAN on 50-dim RAW malicious samples")
        print("   2. Generate adversarial RAW samples")
        print("   3. Save RAW version")
        print("   4. Encode to LATENT version (for fair comparison)")
        sys.exit(1)
    
    # Check required files
    required_files = ['X_train.npy', 'y_train.npy', 'X_test.npy', 'y_test.npy']
    missing = [f for f in required_files if not (DATA_DIR / f).exists()]
    
    if missing:
        print(f"❌ Missing files: {missing}")
        print("\n   Hãy chạy trước:")
        print("   python pipelines/attacks/generate_adversarial_samples.py\n")
        sys.exit(1)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"  Running Exp3: GAN Attack (RAW)")
    print(f"  GAN trained on 50-dim RAW space")
    print(f"{'='*80}\n")
    
    # Run evaluation
    python_exec = "/home/mtu/miniconda3/envs/fl-fedavg/bin/python"
    cmd = [
        python_exec,
        str(BASE_DIR / "run_model_evaluation.py"),
        "--data-dir", str(DATA_DIR),
        "--output-dir", str(OUTPUT_DIR),
        "--exp-name", "EXP3: GAN ATTACK (RAW - 50 dims, fair comparison)"
    ]
    
    subprocess.run(cmd, check=True)
    
    print(f"\n{'='*80}")
    print("✅ HOÀN THÀNH EXP3 RAW!")
    print(f"{'='*80}")
    print(f"\n📊 Results: {OUTPUT_DIR}")
    print(f"\n💡 Compare with LATENT:")
    print(f"   LATENT results: results/latent/exp3_gan_attack/")
    print(f"   RAW (this):     {OUTPUT_DIR}/")
    print(f"\n   → Same adversarial source (GAN on 50-dim)")
    print(f"   → RAW uses directly, LATENT encodes to 64-dim\n")

if __name__ == '__main__':
    main()
