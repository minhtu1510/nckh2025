"""
Experiment 3: GAN Adversarial Attack — LATENT Features

Pipeline:
  - GAN được train trên malicious raw features → fake malicious samples (giống benign)
  - Fake malicious đã được encode sang latent space và lưu ở exp3_gan_attack/
  - Test set (latent): benign_clean + GAN_malicious (latent)
  - Evaluate individual latent models từ exp1

Data: datasets/splits/3.1_latent/exp3_gan_attack/
"""
import sys, numpy as np, subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

PYTHON = "/home/mtu/miniconda3/envs/fl-fedavg/bin/python"
RUNNER = str(BASE_DIR / "run_ensemble_evaluation.py")


def main():
    print("\n🚀 EXP3 LATENT: GAN ADVERSARIAL ATTACK — LATENT Features")
    print("   GAN samples encoded to latent → evaluate pre-trained latent models")

    base       = BASE_DIR / "datasets/splits/3.1_latent"
    gan_dir    = base / "exp3_gan_attack"
    models_dir = BASE_DIR / "models/latent/exp1_baseline_latent"
    out_dir    = BASE_DIR / "results/latent/exp3_gan_attack"

    if not gan_dir.exists():
        print(f"❌ GAN latent data not found: {gan_dir}")
        sys.exit(1)

    if not models_dir.exists():
        print(f"❌ Latent models not found: {models_dir}")
        print("   Hãy chạy: python experiments/latent/exp1_baseline_latent.py")
        sys.exit(1)

    X_g = np.load(gan_dir / 'X_test.npy')
    y_g = np.load(gan_dir / 'y_test.npy')
    print(f"\n  GAN latent test: {len(X_g):,} samples, dim={X_g.shape[1]}")
    print(f"  Benign: {(y_g==0).sum():,}  |  GAN Malicious: {(y_g==1).sum():,}")

    print(f"\n{'='*70}")
    print(f"  Exp3 Latent — GAN Adversarial Attack")
    print(f"  Models: Pre-trained on CLEAN latent data (Exp1)")
    print(f"  Test: GAN-generated malicious samples (latent space)")
    print(f"{'='*70}\n")

    cmd = [
        PYTHON, RUNNER,
        "--data-dir",        str(gan_dir),
        "--models-load-dir", str(models_dir),
        "--output-dir",      str(out_dir),
        "--exp-name",        "EXP3 LATENT: GAN ADVERSARIAL ATTACK"
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"❌ Error: {e}")

    print(f"\n✅ Exp3 Latent hoàn thành!")
    print(f"📁 Results: {out_dir}/")


if __name__ == '__main__':
    main()
