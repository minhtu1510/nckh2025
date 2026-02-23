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
from sklearn.feature_selection import SelectKBest, f_classif

# Go up to ids_research root (2 levels up from preprocessing)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "datasets" / "raw"
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
SPLITS_DIR = BASE_DIR / "datasets" / "splits" / "2.1"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# Constants - FLEXIBLE RATIOS
TOTAL_SAMPLES = 400_000  # Total samples to collect
BENIGN_RATIO = 2/3  # Ratio of benign samples (0.5 = 50%, 0.7 = 70%, etc.)
TRAIN_RATIO = 0.75  # Ratio for train split (0.75 = 75% train, 25% test)

N_BENIGN = int(TOTAL_SAMPLES * BENIGN_RATIO)
N_MALICIOUS = TOTAL_SAMPLES - N_BENIGN
N_TRAIN = int(TOTAL_SAMPLES * BENIGN_RATIO * TRAIN_RATIO)
N_TEST = N_BENIGN - N_TRAIN
N_TRAIN_MAL = int(N_MALICIOUS * TRAIN_RATIO)
N_TEST_MAL = N_MALICIOUS - N_TRAIN_MAL

RANDOM_STATE = 42
CHUNK_SIZE = 50_000

# Feature selection (set to None to disable, or number of features to keep)
N_FEATURES_SELECT = None  # e.g., 50 to keep top 50 features

np.random.seed(RANDOM_STATE)


def find_raw_file():
    """Tìm file CIC-ToN-IoT."""
    # User prefers CSV
    candidates = [
        RAW_DIR / "CIC-ToN-IoT.csv",
        RAW_DIR / "CIC-ToN-IoT-V2.parquet",
        RAW_DIR / "CIC-ToN-IoT.parquet",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            print(f"  Using: {candidate.name}")
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
        try:
            df = pd.read_parquet(raw_file)
            chunks = [df]
        except ImportError:
            print("  ❌ pyarrow not installed. Please install: pip install pyarrow")
            raise
    else:
        print("  Reading CSV in chunks...")
        try:
            chunks = pd.read_csv(raw_file, chunksize=CHUNK_SIZE, low_memory=False)
        except Exception as e:
            print(f"  ⚠️  C parser failed, trying Python engine (slower)...")
            chunks = pd.read_csv(raw_file, chunksize=CHUNK_SIZE, low_memory=False, engine='python')
    
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
        try:
            df = pd.read_parquet(raw_file)
            chunks = [df]
        except ImportError:
            print("  ❌ pyarrow not installed. Please install: pip install pyarrow")
            raise
    else:
        try:
            chunks = pd.read_csv(raw_file, chunksize=CHUNK_SIZE, low_memory=False)
        except Exception as e:
            print(f"  ⚠️  C parser failed, trying Python engine (slower)...")
            chunks = pd.read_csv(raw_file, chunksize=CHUNK_SIZE, low_memory=False, engine='python')
    
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
    
    # ========================================================================
    # STEP 2.5: REMOVE BIASED/SUBJECTIVE FEATURES
    # ========================================================================
    print("\n[STEP 2.5] Removing biased/subjective features...")
    
    # Features to exclude (case-insensitive matching)
    EXCLUDE_FEATURES = [
        'flow id', 'flow_id', 'flowid',
        'src ip', 'src_ip', 'srcip', 'source ip', 'source_ip',
        'dst ip', 'dst_ip', 'dstip', 'destination ip', 'destination_ip',
        'src port', 'src_port', 'srcport', 'source port', 'source_port',
        'dst port', 'dst_port', 'dstport', 'destination port', 'destination_port',
        'timestamp', 'time', 'datetime',
        'protocol',  # Can be kept as one-hot encoded, but raw protocol ID is biased
        'attack',  # Attack type (we only need binary label)
    ]
    
    print(f"  Features to exclude: {len(EXCLUDE_FEATURES)}")
    
    # Basic preprocessing
    print("\n[STEP 2] Preprocessing...")
    
    # Select numeric columns only
    numeric_cols_benign = benign_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_mal = malicious_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Find common columns
    common_cols = list(set(numeric_cols_benign) & set(numeric_cols_mal))
    if 'label' in common_cols:
        common_cols.remove('label')
    
    # Remove biased features
    common_cols_filtered = []
    excluded_count = 0
    for col in common_cols:
        col_lower = col.lower().strip()
        if col_lower not in EXCLUDE_FEATURES:
            common_cols_filtered.append(col)
        else:
            excluded_count += 1
    
    common_cols = common_cols_filtered
    
    print(f"  Original common features: {len(numeric_cols_benign)}")
    print(f"  After removing biased features: {len(common_cols)} (excluded: {excluded_count})")

    
    # Extract features
    benign_features = benign_df[common_cols].copy()
    malicious_features = malicious_df[common_cols].copy()
    
    # Handle inf/nan
    benign_features = benign_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    malicious_features = malicious_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Convert to numpy arrays
    X_benign = benign_features.values
    y_benign = np.zeros(len(X_benign))
    
    X_malicious = malicious_features.values
    y_malicious = np.ones(len(X_malicious))
    
    print(f"  Feature shapes: X_benign={X_benign.shape}, X_malicious={X_malicious.shape}")
    
    # ========================================================================
    # STEP 3: SPLIT FIRST (before scaling to prevent data leakage!)
    # ========================================================================
    print(f"\n[STEP 3] Splitting train/test BEFORE scaling...")
    
    benign_train_X, benign_test_X, benign_train_y, benign_test_y = train_test_split(
        X_benign, y_benign,
        train_size=N_TRAIN,
        test_size=N_TEST,
        random_state=RANDOM_STATE
    )
    
    malicious_train_X, malicious_test_X, malicious_train_y, malicious_test_y = train_test_split(
        X_malicious, y_malicious,
        train_size=N_TRAIN_MAL,
        test_size=N_TEST_MAL,
        random_state=RANDOM_STATE
    )
    
    print(f"  Benign: {len(benign_train_X):,} train + {len(benign_test_X):,} test")
    print(f"  Malicious: {len(malicious_train_X):,} train + {len(malicious_test_X):,} test")
    
    # ========================================================================
    # STEP 4: FIT SCALER ONLY ON TRAIN SET (Fix data leakage!)
    # ========================================================================
    print(f"\n[STEP 4] Scaling - FIT ONLY ON TRAIN SET...")
    
    scaler = MinMaxScaler()
    
    # Combine train data for fitting
    X_train_combined = np.vstack([benign_train_X, malicious_train_X])
    
    # FIT scaler on TRAIN ONLY
    scaler.fit(X_train_combined)
    print(f"  ✓ Scaler fitted on {len(X_train_combined):,} train samples")
    
    # Transform all splits
    benign_train_X_scaled = scaler.transform(benign_train_X)
    benign_test_X_scaled = scaler.transform(benign_test_X)
    malicious_train_X_scaled = scaler.transform(malicious_train_X)
    malicious_test_X_scaled = scaler.transform(malicious_test_X)
    
    print(f"  ✓ Transformed train and test sets")
    
    # ========================================================================
    # STEP 5: FEATURE SELECTION (Optional, fit only on train!)
    # ========================================================================
    if N_FEATURES_SELECT is not None and N_FEATURES_SELECT < benign_train_X_scaled.shape[1]:
        print(f"\n[STEP 5] Feature Selection - FIT ONLY ON TRAIN SET...")
        print(f"  Selecting top {N_FEATURES_SELECT} features from {benign_train_X_scaled.shape[1]}")
        
        selector = SelectKBest(f_classif, k=N_FEATURES_SELECT)
        
        # ✅ FIX: Combine SCALED train data (not raw!)
        X_train_combined_scaled = np.vstack([benign_train_X_scaled, malicious_train_X_scaled])
        y_train_combined = np.hstack([benign_train_y, malicious_train_y])
        
        # FIT selector on SCALED TRAIN data
        selector.fit(X_train_combined_scaled, y_train_combined)
        print(f"  ✓ Selector fitted on SCALED train set")
        
        # Transform all splits
        benign_train_X_scaled = selector.transform(benign_train_X_scaled)
        benign_test_X_scaled = selector.transform(benign_test_X_scaled)
        malicious_train_X_scaled = selector.transform(malicious_train_X_scaled)
        malicious_test_X_scaled = selector.transform(malicious_test_X_scaled)
        
        print(f"  ✓ Selected features: {benign_train_X_scaled.shape[1]}")
    else:
        print(f"\n[STEP 5] Feature Selection - SKIPPED (N_FEATURES_SELECT={N_FEATURES_SELECT})")
    
    # Update variables with scaled data
    benign_train_X = benign_train_X_scaled
    benign_test_X = benign_test_X_scaled
    malicious_train_X = malicious_train_X_scaled
    malicious_test_X = malicious_test_X_scaled
    
    print(f"  Final shapes: X_train={benign_train_X.shape}, X_test={benign_test_X.shape}")
    
    # ========================================================================
    # STEP 6: SAVE BASE SPLITS
    # ========================================================================
    print(f"\n[STEP 6] Saving base splits...")
    
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
    print(f"\n[STEP 7] Creating Exp1 (Baseline) data...")
    
    X_train = np.vstack([benign_train_X, malicious_train_X])
    y_train = np.hstack([benign_train_y, malicious_train_y])
    X_test = np.vstack([benign_test_X, malicious_test_X])
    y_test = np.hstack([benign_test_y, malicious_test_y])
    
    # ✅ FIX: Shuffle with explicit seed for reproducibility
    rng_shuffle = np.random.RandomState(RANDOM_STATE)
    train_idx = rng_shuffle.permutation(len(X_train))
    test_idx = rng_shuffle.permutation(len(X_test))
    
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
    print(f"\n[STEP 8] Creating Exp2 (Poisoning) data...")
    
    exp2_dir = SPLITS_DIR / 'exp2_poisoning'
    exp2_dir.mkdir(exist_ok=True)
    
    # Save test (same for all poison rates)
    np.save(exp2_dir / 'X_test.npy', X_test)
    np.save(exp2_dir / 'y_test.npy', y_test)
    
    for poison_rate in [0.05, 0.10, 0.15, 0.50]:  # Added 0.50
        rate_str = f"{int(poison_rate*100):02d}"
        rate_dir = exp2_dir / f"poison_{rate_str}"
        rate_dir.mkdir(exist_ok=True)
        
        # ✅ FIX: Use dedicated RNG for each poison rate (reproducible!)
        # Different seed for each rate to ensure independence
        poison_seed = RANDOM_STATE + int(poison_rate * 1000)
        rng_poison = np.random.RandomState(poison_seed)
        
        # ✅ ONLY FLIP MALICIOUS → BENIGN (not both ways!)
        y_train_poisoned = y_train.copy()
        
        # Find indices of MALICIOUS samples only (y == 1)
        malicious_indices = np.where(y_train == 1)[0]
        n_malicious = len(malicious_indices)
        
        # Calculate number to poison based on malicious count
        n_poison = int(n_malicious * poison_rate)
        
        # Randomly select malicious samples to flip
        poison_indices = rng_poison.choice(malicious_indices, n_poison, replace=False)
        
        # Flip ONLY malicious → benign (1 → 0)
        y_train_poisoned[poison_indices] = 0
        
        np.save(rate_dir / 'X_train.npy', X_train)
        np.save(rate_dir / 'y_train.npy', y_train_poisoned)
        np.save(rate_dir / 'X_test.npy', X_test)
        np.save(rate_dir / 'y_test.npy', y_test)
        
        print(f"  ✓ Poison {rate_str}%: Flipped {n_poison:,}/{n_malicious:,} malicious→benign (seed={poison_seed})")
    
    # Save metadata
    print(f"\n[STEP 9] Saving metadata...")
    
    metadata = {
        'dataset': 'CIC-ToN-IoT',
        'total_samples': TOTAL_SAMPLES,
        'benign_ratio': BENIGN_RATIO,
        'malicious_ratio': 1 - BENIGN_RATIO,
        'train_ratio': TRAIN_RATIO,
        'n_benign_total': N_BENIGN,
        'n_malicious_total': N_MALICIOUS,
        'n_train_benign': N_TRAIN,
        'n_test_benign': N_TEST,
        'n_train_malicious': N_TRAIN_MAL,
        'n_test_malicious': N_TEST_MAL,
        'random_state': RANDOM_STATE,
        'feature_dim': X_train.shape[1],
        'n_features_selected': N_FEATURES_SELECT,
        'data_leakage_fixed': True,
        'scaler_fit_on': 'train_only',
        'feature_selection_fit_on': 'train_only',
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
    print(f"  • Total: {TOTAL_SAMPLES:,} samples")
    print(f"  • Benign: {N_BENIGN:,} ({BENIGN_RATIO*100:.0f}%) - {N_TRAIN:,} train + {N_TEST:,} test")
    print(f"  • Malicious: {N_MALICIOUS:,} ({(1-BENIGN_RATIO)*100:.0f}%) - {N_TRAIN_MAL:,} train + {N_TEST_MAL:,} test")
    print(f"  • Features: {X_train.shape[1]}")
    if N_FEATURES_SELECT is not None:
        print(f"  • Feature Selection: {N_FEATURES_SELECT} features selected")
    print(f"  • ✅ NO DATA LEAKAGE: Scaler & Feature Selection fit on train only")
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
