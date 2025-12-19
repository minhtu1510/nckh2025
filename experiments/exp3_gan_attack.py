"""
Thực nghiệm 3: GAN Attack
Dùng lại script training chung với adversarial dataset.
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets" / "splits" / "exp3_gan"
OUTPUT_DIR = BASE_DIR / "results" / "exp3_gan"

def main():
    print("\n🚀 THỰC NGHIỆM 3: GAN ATTACK")
    print("   Train: 150,000 Benign + 150,000 Malicious (clean)")
    print("   Test: 50,000 Benign (clean) + 50,000 Generated (adversarial)\n")
    
    # Check if adversarial data exists
    if not DATA_DIR.exists() or not (DATA_DIR / 'X_test.npy').exists():
        print(f"❌ Error: Adversarial data not found!")
        print("   Hãy chạy: python generate_adversarial_samples.py")
        print("   (Sẽ mất ~30-60 phút để generate adversarial samples)")
        sys.exit(1)
    
    # Run evaluation
    cmd = [
        sys.executable,
        str(BASE_DIR / "run_model_evaluation.py"),
        "--data-dir", str(DATA_DIR),
        "--output-dir", str(OUTPUT_DIR),
        "--exp-name", "THỰC NGHIỆM 3: GAN ATTACK"
    ]
    
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
