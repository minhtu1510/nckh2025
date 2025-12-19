#!/usr/bin/env python3
"""
Chuẩn bị dữ liệu cho 3 thực nghiệm từ CIC-ToN-IoT
Đọc theo chunk, sampling 200k benign + 200k malicious

Phân bổ:
    Benign: 200,000 (150,000 train + 50,000 test)
    Malicious: 200,000 (150,000 train + 50,000 test)
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "datasets" / "raw"
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
SPLITS_DIR = BASE_DIR / "datasets" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# Constants
N_BENIGN = 200_000
N_MALICIOUS = 200_000
N_TRAIN = 150_000
N_TEST = 50_000
RANDOM_STATE = 42
CHUNK_SIZE = 50_000

np.random.seed(RANDOM_STATE)


def find_raw_file():
    """Tìm file CIC-ToN-IoT."""
    candidates = [
        RAW_DIR / "CIC-ToN-IoT.csv",
        RAW_DIR / "CIC-ToN-IoT-V2.parquet",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    raise FileNotFoundError(
        f"❌ Không tìm thấy CIC-ToN-IoT file!\n"
        f"Cần có một trong: {[str(c) for c in candidates]}"
    )


def reservoir_sample_toniot():
    """
    Reservoir sampling từ CIC-ToN-IoT.
    Returns: (benign_samples, malicious_samples) as list of dicts
    """
    print("\n" + "="*80)
    print(" "*20 + "SAMPLING CIC-ToN-IoT - BINARY MODE")
    print("="*80)
    
    raw_file = find_raw_file()
    print(f"\n✓ Found: {raw_file.name}")
    print(f"  Size: {raw_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"\n[TARGET] {N_BENIGN:,} Benign + {N_MALICIOUS:,} Malicious")
    
    # Pass 1: Count attack types
    print("\n[PASS 1] Counting attack type distribution...")
    
    attack_counts = {}
    benign_count = 0
    chunk_num = 0
    
    # Read file
    if raw_file.suffix == '.parquet':
        print("  Reading parquet...")
        df = pd.read_parquet(raw_file)
        chunks = [df]
    else:
        print("  Reading CSV in chunks...")
        chunks = pd.read_csv(raw_file, chunksize=CHUNK_SIZE, low_memory=False)
    
    for chunk in chunks:
        chunk_num += 1
        
        # Lowercase columns
        chunk.columns = [c.lower() for c in chunk.columns]
        
        # Find label column
        label_col = None
        for possible in ['label', 'class', 'type']:
            if possible in chunk.columns:
                label_col = possible
                break
        
        if not label_col:
            continue
        
        # Clean labels
        chunk[label_col] = chunk[label_col].astype(str).str.strip().str.lower()
        
        # Count
        for label in chunk[label_col].unique():
            if label in ['benign', 'normal', '0']:
                benign_count += (chunk[label_col] == label).sum()
            else:
                attack_counts[label] = attack_counts.get(label, 0) + (chunk[label_col] == label).sum()
        
        if chunk_num % 10 == 0:
            print(f"  Processed {chunk_num} chunks...")
    
    total_malicious = sum(attack_counts.values())
    print(f"\n  Dataset statistics:")
    print(f"    Benign: {benign_count:,}")
    print(f"    Total Malicious: {total_malicious:,}")
    print(f"\n  Attack type distribution:")
    for attack in sorted(attack_counts.keys()):
        count = attack_counts[attack]
        pct = count / total_malicious * 100
        print(f"    {attack}: {count:,} ({pct:.1f}%)")
    
    # Calculate proportional targets
    print(f"\n[PASS 2] Calculating proportional targets for {N_MALICIOUS:,} malicious...")
    attack_targets = {}
    for attack, count in attack_counts.items():
        ratio = count / total_malicious
        target = int(N_MALICIOUS * ratio)
        attack_targets[attack] = target
        print(f"  {attack}: {target:,} ({ratio*100:.1f}%)")
    
    # Pass 2: Reservoir sampling
    print(f"\n[PASS 3] Reservoir sampling...")
    
    benign_samples = []
    attack_samples = {attack: [] for attack in attack_targets.keys()}
    benign_seen = 0
    attack_seen = {attack: 0 for attack in attack_targets.keys()}
    
    chunk_num = 0
    
    # Re-read
    if raw_file.suffix == '.parquet':
        df = pd.read_parquet(raw_file)
        chunks = [df]
    else:
        chunks = pd.read_csv(raw_file, chunksize=CHUNK_SIZE, low_memory=False)
    
    for chunk in chunks:
        chunk_num += 1
        
        chunk.columns = [c.lower() for c in chunk.columns]
        
        label_col = None
        for possible in ['label', 'class', 'type']:
            if possible in chunk.columns:
                label_col = possible
                break
        
        if not label_col:
            continue
        
        chunk[label_col] = chunk[label_col].astype(str).str.strip().str.lower()
        
        # Sample benign
        benign_chunk = chunk[chunk[label_col].isin(['benign', 'normal', '0'])]
        for _, row in benign_chunk.iterrows():
            benign_seen += 1
            if len(benign_samples) < N_BENIGN:
                benign_samples.append(row.to_dict())
            else:
                rand_idx = np.random.randint(0, benign_seen)
                if rand_idx < N_BENIGN:
                    benign_samples[rand_idx] = row.to_dict()
        
        # Sample each attack type
        for attack in attack_targets.keys():
            attack_chunk = chunk[chunk[label_col] == attack]
            target = attack_targets[attack]
            
            for _, row in attack_chunk.iterrows():
                attack_seen[attack] += 1
                if len(attack_samples[attack]) < target:
                    attack_samples[attack].append(row.to_dict())
                else:
                    rand_idx = np.random.randint(0, attack_seen[attack])
                    if rand_idx < target:
                        attack_samples[attack][rand_idx] = row.to_dict()
        
        if chunk_num % 10 == 0:
            total_mal = sum(len(attack_samples[a]) for a in attack_samples)
            print(f"  Chunk {chunk_num}: Benign={len(benign_samples):,}/{N_BENIGN:,}, Malicious={total_mal:,}/{N_MALICIOUS:,}")
    
    print(f"\n  ✓ Collected:")
    print(f"    Benign: {len(benign_samples):,}")
    for attack in sorted(attack_samples.keys()):
        print(f"    {attack}: {len(attack_samples[attack]):,}")
    
    return benign_samples, attack_samples


def preprocess_and_split():
    """Main function."""
    print("\n🚀 CHUẨN BỊ DỮ LIỆU CHO 3 THỰC NGHIỆM")
    print("   Dataset: CIC-ToN-IoT")
    print(f"   Scale: {N_BENIGN:,} Benign + {N_MALICIOUS:,} Malicious\n")
    
    # Sample data
    benign_samples, attack_samples = reservoir_sample_toniot()
    
    # Create DataFrames
    print("\n[STEP 1] Creating DataFrames...")
    benign_df = pd.DataFrame(benign_samples[:N_BENIGN])
    
    malicious_dfs = []
    for attack in attack_samples.keys():
        if len(attack_samples[attack]) > 0:
            malicious_dfs.append(pd.DataFrame(attack_samples[attack]))
    
    malicious_df = pd.concat(malicious_dfs, ignore_index=True) if malicious_dfs else pd.DataFrame()
    
    # Add binary labels
    benign_df['label'] = 0
    malicious_df['label'] = 1
    
    # Drop original label columns
    for col in ['Label', 'label', 'class', 'type', 'Type']:
        if col in benign_df.columns and col != 'label':
            benign_df = benign_df.drop(columns=[col], errors='ignore')
        if col in malicious_df.columns and col != 'label':
            malicious_df = malicious_df.drop(columns=[col], errors='ignore')
    
    print(f"  Benign: {len(benign_df):,}")
    print(f"  Malicious: {len(malicious_df):,}")
    
    # Basic preprocessing
    print("\n[STEP 2] Preprocessing...")
    
    # Select numeric columns only
    numeric_cols_benign = benign_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_mal = malicious_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Find common columns
    common_cols = list(set(numeric_cols_benign) & set(numeric_cols_mal))
    if 'label' in common_cols:
        common_cols.remove('label')
    
    print(f"  Common features: {len(common_cols)}")
    
    # Extract features
    benign_features = benign_df[common_cols].copy()
    malicious_features = malicious_df[common_cols].copy()
    
    # Handle inf/nan
    benign_features = benign_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    malicious_features = malicious_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale
    print("  Scaling features...")
    scaler = MinMaxScaler()
    
    # Fit on combined data
    combined_features = pd.concat([benign_features, malicious_features], ignore_index=True)
    scaler.fit(combined_features)
    
    benign_scaled = scaler.transform(benign_features)
    malicious_scaled = scaler.transform(malicious_features)
    
    # Create final arrays
    X_benign = benign_scaled
    y_benign = np.zeros(len(X_benign))
    
    X_malicious = malicious_scaled
    y_malicious = np.ones(len(X_malicious))
    
    print(f"  Final shapes: X_benign={X_benign.shape}, X_malicious={X_malicious.shape}")
    
    # Split train/test
    print(f"\n[STEP 3] Splitting train/test...")
    
    benign_train_X, benign_test_X, benign_train_y, benign_test_y = train_test_split(
        X_benign, y_benign,
        train_size=N_TRAIN,
        test_size=N_TEST,
        random_state=RANDOM_STATE
    )
    
    malicious_train_X, malicious_test_X, malicious_train_y, malicious_test_y = train_test_split(
        X_malicious, y_malicious,
        train_size=N_TRAIN,
        test_size=N_TEST,
        random_state=RANDOM_STATE
    )
    
    print(f"  Benign: {len(benign_train_X):,} train + {len(benign_test_X):,} test")
    print(f"  Malicious: {len( malicious_train_X):,} train + {len(malicious_test_X):,} test")
    
    # Save base splits
    print(f"\n[STEP 4] Saving base splits...")
    
    np.save(SPLITS_DIR / 'benign_train_X.npy', benign_train_X)
    np.save(SPLITS_DIR / 'benign_train_y.npy', benign_train_y)
    np.save(SPLITS_DIR / 'benign_test_X.npy', benign_test_X)
    np.save(SPLITS_DIR / 'benign_test_y.npy', benign_test_y)
    
    np.save(SPLITS_DIR / 'malicious_train_X.npy', malicious_train_X)
    np.save(SPLITS_DIR / 'malicious_train_y.npy', malicious_train_y)
    np.save(SPLITS_DIR / 'malicious_test_X.npy', malicious_test_X)
    np.save(SPLITS_DIR / 'malicious_test_y.npy', malicious_test_y)
    
    print(f"  ✓ Saved base splits to {SPLITS_DIR}/")
    
    # Create Exp1 data
    print(f"\n[STEP 5] Creating Exp1 (Baseline) data...")
    
    X_train = np.vstack([benign_train_X, malicious_train_X])
    y_train = np.hstack([benign_train_y, malicious_train_y])
    X_test = np.vstack([benign_test_X, malicious_test_X])
    y_test = np.hstack([benign_test_y, malicious_test_y])
    
    # Shuffle
    train_idx = np.random.permutation(len(X_train))
    test_idx = np.random.permutation(len(X_test))
    
    X_train = X_train[train_idx]
    y_train = y_train[train_idx]
    X_test = X_test[test_idx]
    y_test = y_test[test_idx]
    
    exp1_dir = SPLITS_DIR / 'exp1_baseline'
    exp1_dir.mkdir(exist_ok=True)
    
    np.save(exp1_dir / 'X_train.npy', X_train)
    np.save(exp1_dir / 'y_train.npy', y_train)
    np.save(exp1_dir / 'X_test.npy', X_test)
    np.save(exp1_dir / 'y_test.npy', y_test)
    
    print(f"  ✓ Exp1 data: Train={X_train.shape}, Test={X_test.shape}")
    
    # Create Exp2 data (poisoned)
    print(f"\n[STEP 6] Creating Exp2 (Poisoning) data...")
    
    exp2_dir = SPLITS_DIR / 'exp2_poisoning'
    exp2_dir.mkdir(exist_ok=True)
    
    # Save test (same for all poison rates)
    np.save(exp2_dir / 'X_test.npy', X_test)
    np.save(exp2_dir / 'y_test.npy', y_test)
    
    for poison_rate in [0.05, 0.10, 0.15, 0.50]:  # Added 0.50
        rate_str = f"{int(poison_rate*100):02d}"
        rate_dir = exp2_dir / f"poison_{rate_str}"
        rate_dir.mkdir(exist_ok=True)
        
        # Flip labels
        y_train_poisoned = y_train.copy()
        n_poison = int(len(y_train_poisoned) * poison_rate)
        poison_indices = np.random.choice(len(y_train_poisoned), n_poison, replace=False)
        y_train_poisoned[poison_indices] = 1 - y_train_poisoned[poison_indices]
        
        np.save(rate_dir / 'X_train.npy', X_train)
        np.save(rate_dir / 'y_train.npy', y_train_poisoned)
        np.save(rate_dir / 'X_test.npy', X_test)
        np.save(rate_dir / 'y_test.npy', y_test)
        
        print(f"  ✓ Poison {rate_str}%: Flipped {n_poison:,} labels")
    
    # Save metadata
    print(f"\n[STEP 7] Saving metadata...")
    
    metadata = {
        'dataset': 'CIC-ToN-IoT',
        'n_benign_total': N_BENIGN,
        'n_malicious_total': N_MALICIOUS,
        'n_train_per_class': N_TRAIN,
        'n_test_per_class': N_TEST,
        'random_state': RANDOM_STATE,
        'feature_dim': X_train.shape[1],
        'created_at': str(pd.Timestamp.now()),
    }
    
    with open(SPLITS_DIR / 'experiment_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Metadata saved")
    
    # Summary
    print("\n" + "="*80)
    print("✅ HOÀN THÀNH! Dữ liệu đã sẵn sàng cho 3 thực nghiệm")
    print("="*80)
    print(f"\n📊 TỔNG KẾT:")
    print(f"  • Dataset: CIC-ToN-IoT")
    print(f"  • Benign: {N_BENIGN:,} ({N_TRAIN:,} train + {N_TEST:,} test)")
    print(f"  • Malicious: {N_MALICIOUS:,} ({N_TRAIN:,} train + {N_TEST:,} test)")
    print(f"  • Features: {X_train.shape[1]}")
    print(f"\n💾 Files tại: {SPLITS_DIR}/")
    print(f"  ✓ exp1_baseline/")
    print(f"  ✓ exp2_poisoning/poison_05/, 10/, 15/")
    print(f"  ✓ Base splits: benign_*, malicious_*")
    print(f"\n📝 TIẾP THEO:")
    print(f"  1️⃣  python experiments/exp1_baseline.py")
    print(f"  2️⃣  python experiments/exp2_data_poisoning.py")
    print(f"  3️⃣  python generate_adversarial_samples.py")
    print(f"  4️⃣  python experiments/exp3_gan_attack.py")
    print("\n")


if __name__ == '__main__':
    preprocess_and_split()
