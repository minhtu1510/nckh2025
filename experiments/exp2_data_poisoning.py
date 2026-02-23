"""
Thực nghiệm 2: Data Poisoning (RAW - Fair Comparison)
Train models on poisoned 50-dim features
Same preprocessing as LATENT, WITHOUT encoding layer
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# Use RAW poisoned data from same pipeline as LATENT
DATA_BASE = BASE_DIR / "datasets" / "splits" / "3.0_raw_from_latent" / "exp2_poisoning"
OUTPUT_BASE = BASE_DIR / "results" / "raw_fair" / "exp2_poisoning"

POISON_RATES = [5, 10, 15, 50]

def main():
    print("\n🚀 THỰC NGHIỆM 2: DATA POISONING (RAW - Fair Comparison)")
    print("   Pipeline: Raw → Filter → Scale → Select (50) → POISON → STOP")
    print("   (Same poisoning as LATENT, WITHOUT encoding)")
    print(f"   Poison rates: {POISON_RATES}%\n")
    
    # Check if base directory exists
    if not DATA_BASE.exists():
        print(f"❌ Error: Fair comparison poisoned data not found at {DATA_BASE}")
        print("\n   Hãy chạy:")
        print("   python pipelines/preprocessing/prepare_latent_data.py")
        print("\n   → Script này sẽ tạo CẢ RAW VÀ LATENT poisoned data")
        sys.exit(1)
    
    for rate in POISON_RATES:
        rate_str = f"{rate:02d}"
        data_dir = DATA_BASE / f"poison_{rate_str}"
        output_dir = OUTPUT_BASE / f"poison_{rate_str}"
        
        if not data_dir.exists():
            print(f"⚠  Skipping poison_{rate}% - data not found at {data_dir}")
            continue
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"  Running Exp2 with {rate}% Poisoning Rate (RAW)")
        print(f"  {rate}% malicious labels flipped to benign")
        print(f"{'='*80}\n")
        
        # Run evaluation
        python_exec = "/home/mtu/miniconda3/envs/fl-fedavg/bin/python"
        cmd = [
            python_exec,
            str(BASE_DIR / "run_model_evaluation.py"),
            "--data-dir", str(data_dir),
            "--output-dir", str(output_dir),
            "--exp-name", f"EXP2: POISONING {rate}% (RAW - 50 dims, fair)"
        ]
        
        subprocess.run(cmd, check=True)
        
        print(f"\n✅ Completed poison rate {rate}%\n")
    
    print(f"\n{'='*80}")
    print("✅ HOÀN THÀNH TẤT CẢ POISON RATES RAW!")
    print(f"{'='*80}")
    print(f"\n📊 Results: {OUTPUT_BASE}/")
    for rate in POISON_RATES:
        print(f"  - poison_{rate:02d}/")
    print(f"\n💡 Compare with LATENT:")
    print(f"   LATENT results: results/latent/exp2_poisoning/")
    print(f"   RAW (this):     {OUTPUT_BASE}/\n")

if __name__ == '__main__':
    main()
