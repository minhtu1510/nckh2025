#!/usr/bin/env python3
"""
Preprocessing CIC-ToN-IoT với chunk processing
Tương tự CICIDS2018 nhưng cho CIC-ToN-IoT
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("CIC-ToN-IoT PREPROCESSING - CHUNK MODE")
print("="*80)

# Paths
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "datasets" / "raw"
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
ARTEFACTS_DIR = PROCESSED_DIR / "artefacts"

# Find raw file
raw_candidates = [
    RAW_DIR / "CIC-ToN-IoT.csv",
    RAW_DIR / "CIC-ToN-IoT-V2.parquet",
    RAW_DIR / "CIC-ToN-IoT-small.csv",
]

raw_file = None
for candidate in raw_candidates:
    if candidate.exists():
        raw_file = candidate
        print(f"\n✓ Found raw file: {candidate.name}")
        print(f"  Size: {candidate.stat().st_size / 1024 / 1024:.1f} MB")
        break

if not raw_file:
    print("\n❌ ERROR: No CIC-ToN-IoT raw file found!")
    print("\nSearched for:")
    for c in raw_candidates:
        print(f"  - {c}")
    print("\nPlease download CIC-ToN-IoT dataset first.")
    sys.exit(1)

# Output paths
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

output_csv = PROCESSED_DIR / "cic_ton_iot_processed.csv"

print(f"\nOutput: {output_csv}")

# Check if parquet
if raw_file.suffix == '.parquet':
    print("\n[INFO] Converting parquet to CSV first...")
    import pandas as pd
    df = pd.read_parquet(raw_file)
    temp_csv = RAW_DIR / "CIC-ToN-IoT.csv"
    df.to_csv(temp_csv, index=False)
    raw_file = temp_csv
    print(f"  ✓ Converted to {temp_csv}")
    del df

# Run preprocessing
print("\n" + "="*80)
print("STARTING CHUNK PROCESSING")
print("="*80)

import subprocess
cmd = [
    "python",
    "pipelines/preprocessing/cic_ton_iot.py",
    "--input", str(raw_file),
    "--output", str(output_csv),
    "--artefacts", str(ARTEFACTS_DIR)
]

print(f"\nCommand: {' '.join(cmd)}\n")

result = subprocess.run(cmd, cwd=str(BASE_DIR))

if result.returncode != 0:
    print("\n❌ Preprocessing failed!")
    sys.exit(1)

print("\n" + "="*80)
print("✓ PREPROCESSING DONE!")
print("="*80)

# Check output
if output_csv.exists():
    print(f"\n✓ Processed file created: {output_csv}")
    print(f"  Size: {output_csv.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Quick stats
    import pandas as pd
    print("\n[QUICK STATS]")
    df_sample = pd.read_csv(output_csv, nrows=1000)
    print(f"  Features: {len(df_sample.columns) - 1}")
    print(f"  Sample shape: {df_sample.shape}")
    
    if 'label' in df_sample.columns:
        unique_labels = sorted(df_sample['label'].unique())
        print(f"  Classes (in sample): {unique_labels}")
    
    print("\n🚀 Next step: python setup_toniot.py")
else:
    print("\n❌ Output file not found!")
