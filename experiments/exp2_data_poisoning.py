"""
Thực nghiệm 2: Data Poisoning
Gọi script training chung cho từng poison rate (5%, 10%, 15%).
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
POISON_RATES = [5, 10, 15]

def main():
    print("\n🚀 THỰC NGHIỆM 2: DATA POISONING")
    print("   Sẽ chạy lần lượt với poison rates: 5%, 10%, 15%\n")
    
    for rate in POISON_RATES:
        rate_str = f"{rate:02d}"
        data_dir = BASE_DIR / "datasets" / "splits" / "exp2_poisoning" / f"poison_{rate_str}"
        output_dir = BASE_DIR / "results" / "exp2_poisoning" / f"poison_{rate_str}"
        
        # Check if data exists
        if not data_dir.exists():
            print(f"❌ Error: Data not found at {data_dir}")
            print("   Hãy chạy: python prepare_experiment_data.py")
            sys.exit(1)
        
        print(f"\n{'='*80}")
        print(f"  Running with POISON RATE = {rate}%")
        print(f"{'='*80}\n")
        
        # Run evaluation
        cmd = [
            sys.executable,
            str(BASE_DIR / "run_model_evaluation.py"),
            "--data-dir", str(data_dir),
            "--output-dir", str(output_dir),
            "--exp-name", f"THỰC NGHIỆM 2: DATA POISONING ({rate}%)"
        ]
        
        subprocess.run(cmd, check=True)
        
        print(f"\n✅ Completed poison rate {rate}%\n")
    
    print(f"\n{'='*80}")
    print("✅ HOÀN THÀNH TẤT CẢ POISON RATES!")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
