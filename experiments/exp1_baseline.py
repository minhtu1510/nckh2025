"""
Thực nghiệm 1: Baseline
Chỉ gọi script training chung với clean dataset.
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets" / "splits" / "exp1_baseline"
OUTPUT_DIR = BASE_DIR / "results" / "exp1_baseline"

def main():
    print("\n🚀 THỰC NGHIỆM 1: BASELINE")
    print("   Train: 150,000 Benign + 150,000 Malicious")
    print("   Test: 50,000 Benign + 50,000 Malicious\n")
    
    # Check if data exists
    if not DATA_DIR.exists():
        print(f"❌ Error: Data not found at {DATA_DIR}")
        print("   Hãy chạy: python prepare_experiment_data.py")
        sys.exit(1)
    
    # Run evaluation
    cmd = [
        sys.executable,
        str(BASE_DIR / "run_model_evaluation.py"),
        "--data-dir", str(DATA_DIR),
        "--output-dir", str(OUTPUT_DIR),
        "--exp-name", "THỰC NGHIỆM 1: BASELINE"
    ]
    
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
