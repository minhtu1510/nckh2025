#!/usr/bin/env python3
"""
Chuẩn bị dữ liệu với Dual-Encoder Latent Features
- Feature selection
- 2 Autoencoder riêng biệt: 1 cho benign, 1 cho malicious
- Extract dual-view features: z = [benign_enc(x), malicious_enc(x)]
- Final features: 64 dims (32 from each encoder)

Author: Research Team
Date: 2026-01-24
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "datasets" / "raw"
SPLITS_DIR = BASE_DIR / "datasets" / "splits" / "3.1_latent"  # New version with latent
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# Constants
TOTAL_SAMPLES = 400_000
BENIGN_RATIO = 2/3
TRAIN_RATIO = 0.75

N_BENIGN = int(TOTAL_SAMPLES * BENIGN_RATIO)
N_MALICIOUS = TOTAL_SAMPLES - N_BENIGN
N_TRAIN = int(TOTAL_SAMPLES * BENIGN_RATIO * TRAIN_RATIO)
N_TEST = N_BENIGN - N_TRAIN
N_TRAIN_MAL = int(N_MALICIOUS * TRAIN_RATIO)
N_TEST_MAL = N_MALICIOUS - N_TRAIN_MAL

RANDOM_STATE = 42
CHUNK_SIZE = 50_000

# Feature selection - ENABLED for latent approach
N_FEATURES_SELECT = 50  # Select top 100 features before AE

# Autoencoder configuration
LATENT_DIM = 32  # Latent dimension per encoder (final: 32*2 = 64 dims after concat)
AE_EPOCHS = 100
AE_BATCH_SIZE = 256
AE_VERBOSE = 1

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def find_raw_file():
    """Tìm file CIC-ToN-IoT."""
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


def build_autoencoder(input_dim, latent_dim, name_prefix="ae"):
    """
    Build Autoencoder for feature extraction
    
    Args:
        input_dim: Input feature dimension
        latent_dim: Latent dimension (bottleneck)
        name_prefix: Prefix for model name
    
    Returns:
        encoder, decoder, autoencoder models
    """
    # Encoder
    encoder_input = Input(shape=(input_dim,), name=f"{name_prefix}_input")
    x = Dense(256, activation='relu', name=f"{name_prefix}_enc1")(encoder_input)
    x = Dense(128, activation='relu', name=f"{name_prefix}_enc2")(x)
    x = Dense(64, activation='relu', name=f"{name_prefix}_enc3")(x)
    latent = Dense(latent_dim, activation='relu', name=f"{name_prefix}_latent")(x)
    
    encoder = Model(encoder_input, latent, name=f"{name_prefix}_encoder")
    
    # Decoder
    decoder_input = Input(shape=(latent_dim,), name=f"{name_prefix}_latent_input")
    x = Dense(64, activation='relu', name=f"{name_prefix}_dec1")(decoder_input)
    x = Dense(128, activation='relu', name=f"{name_prefix}_dec2")(x)
    x = Dense(256, activation='relu', name=f"{name_prefix}_dec3")(x)
    decoder_output = Dense(input_dim, activation='sigmoid', name=f"{name_prefix}_output")(x)
    
    decoder = Model(decoder_input, decoder_output, name=f"{name_prefix}_decoder")
    
    # Autoencoder
    autoencoder_output = decoder(encoder(encoder_input))
    autoencoder = Model(encoder_input, autoencoder_output, name=f"{name_prefix}_autoencoder")
    
    autoencoder.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
    
    return encoder, decoder, autoencoder


def main():
    """Main preprocessing pipeline with dual-encoder concatenation"""
    print("\n" + "="*80)
    print(" "*15 + "🚀 CHUẨN BỊ DỮ LIỆU VỚI DUAL-VIEW LATENT FEATURES")
    print(" "*20 + "Dual-Encoder Concatenation")
    print("="*80)
    print(f"\n  Dataset: CIC-ToN-IoT")
    print(f"  Scale: {N_BENIGN:,} Benign + {N_MALICIOUS:,} Malicious")
    print(f"  Feature Selection: Top {N_FEATURES_SELECT} features")
    print(f"  Latent Dimension: {LATENT_DIM} per encoder → {LATENT_DIM*2} total")
    print(f"  Strategy: z = [benign_enc(x), malicious_enc(x)]")
    print("")
    
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
    # STEP 3: SPLIT FIRST (before any transformation!)
    # ========================================================================
    print(f"\n[STEP 3] Splitting train/test...")
    
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
    # IMPORTANT: Save RAW malicious BEFORE any preprocessing for GAN
    # ========================================================================
    print(f"\n[STEP 3.5] Saving RAW data for GAN generation...")
    
    raw_dir = SPLITS_DIR / 'raw_for_gan'
    raw_dir.mkdir(exist_ok=True)
    
    # Save RAW malicious train (before scaling, selection, encoding)
    np.save(raw_dir / 'malicious_train_X_raw.npy', malicious_train_X)
    np.save(raw_dir / 'malicious_train_y.npy', malicious_train_y)
    
    # Save RAW benign train too (for GAN discriminator training)
    np.save(raw_dir / 'benign_train_X_raw.npy', benign_train_X)
    np.save(raw_dir / 'benign_train_y.npy', benign_train_y)
    
    # Save feature names for reference
    raw_metadata = {
        'features': common_cols,
        'n_features': len(common_cols),
        'n_malicious_samples': len(malicious_train_X),
        'n_benign_samples': len(benign_train_X),
        'note': 'RAW train data BEFORE any preprocessing - for GAN generation',
    }
    
    with open(raw_dir / 'raw_metadata.json', 'w') as f:
        json.dump(raw_metadata, f, indent=2)
    
    print(f"  ✓ Saved RAW malicious train: {malicious_train_X.shape}")
    print(f"  ✓ Saved RAW benign train: {benign_train_X.shape}")
    print(f"  ✓ Location: {raw_dir}/")
    
    # ========================================================================
    # STEP 4: SCALING (fit on train)
    # ========================================================================
    print(f"\n[STEP 4] Scaling - FIT ONLY ON TRAIN SET...")
    
    scaler = MinMaxScaler()
    
    # Combine train data for fitting
    X_train_combined = np.vstack([benign_train_X, malicious_train_X])
    
    # FIT scaler on TRAIN ONLY
    scaler.fit(X_train_combined)
    print(f"  ✓ Scaler fitted on {len(X_train_combined):,} train samples")
    
    # Transform all splits
    benign_train_X = scaler.transform(benign_train_X)
    benign_test_X = scaler.transform(benign_test_X)
    malicious_train_X = scaler.transform(malicious_train_X)
    malicious_test_X = scaler.transform(malicious_test_X)
    
    print(f"  ✓ Transformed train and test sets")
    
    # ========================================================================
    # STEP 5: FEATURE SELECTION (fit on train)
    # ========================================================================
    if N_FEATURES_SELECT is not None and N_FEATURES_SELECT < benign_train_X.shape[1]:
        print(f"\n[STEP 5] Feature Selection - FIT ONLY ON TRAIN SET...")
        print(f"  Selecting top {N_FEATURES_SELECT} features from {benign_train_X.shape[1]}")
        
        selector = SelectKBest(f_classif, k=N_FEATURES_SELECT)
        
        # Combine SCALED train data
        X_train_combined = np.vstack([benign_train_X, malicious_train_X])
        y_train_combined = np.hstack([benign_train_y, malicious_train_y])
        
        # FIT selector on SCALED TRAIN data
        selector.fit(X_train_combined, y_train_combined)
        print(f"  ✓ Selector fitted on SCALED train set")
        
        # Transform all splits
        benign_train_X = selector.transform(benign_train_X)
        benign_test_X = selector.transform(benign_test_X)
        malicious_train_X = selector.transform(malicious_train_X)
        malicious_test_X = selector.transform(malicious_test_X)
        
        print(f"  ✓ Selected features: {benign_train_X.shape[1]}")
    else:
        print(f"\n[STEP 5] Feature Selection - SKIPPED")
    
    input_dim = benign_train_X.shape[1]
    print(f"\n  Input dimension for AE: {input_dim}")
    
    # ========================================================================
    # STEP 5.5: SAVE RAW DATA (50 dims) FOR FAIR COMPARISON WITH LATENT
    # ========================================================================
    print(f"\n[STEP 5.5] Saving RAW data (50 dims) before encoding...")
    
    raw_exp_dir = BASE_DIR / "datasets" / "splits" / "3.0_raw_from_latent"
    raw_exp_dir.mkdir(exist_ok=True)
    
    # Combine train (for Exp1 baseline)
    X_train_raw = np.vstack([benign_train_X, malicious_train_X])
    y_train_raw = np.hstack([benign_train_y, malicious_train_y])
    
    # Combine test
    X_test_raw = np.vstack([benign_test_X, malicious_test_X])
    y_test_raw = np.hstack([benign_test_y, malicious_test_y])
    
    # Shuffle
    rng_raw = np.random.RandomState(RANDOM_STATE)
    train_idx_raw = rng_raw.permutation(len(X_train_raw))
    test_idx_raw = rng_raw.permutation(len(X_test_raw))
    
    X_train_raw = X_train_raw[train_idx_raw]
    y_train_raw = y_train_raw[train_idx_raw]
    X_test_raw = X_test_raw[test_idx_raw]
    y_test_raw = y_test_raw[test_idx_raw]
    
    # Save RAW baseline
    raw_baseline_dir = raw_exp_dir / "exp1_baseline"
    raw_baseline_dir.mkdir(exist_ok=True)
    
    np.save(raw_baseline_dir / 'X_train.npy', X_train_raw)
    np.save(raw_baseline_dir / 'y_train.npy', y_train_raw)
    np.save(raw_baseline_dir / 'X_test.npy', X_test_raw)
    np.save(raw_baseline_dir / 'y_test.npy', y_test_raw)
    
    print(f"  ✓ Saved RAW data: Train={X_train_raw.shape}, Test={X_test_raw.shape}")
    print(f"  ✓ Location: {raw_baseline_dir}/")
    print(f"  → This allows FAIR comparison: RAW (50 dims) vs LATENT (64 dims)")
    
    # Save metadata for RAW
    raw_metadata = {
        'dataset': 'CIC-ToN-IoT',
        'approach': 'raw_selected_features',
        'pipeline': 'Raw → Filter → Scale → Select(50)',
        'feature_dim': input_dim,
        'total_samples': TOTAL_SAMPLES,
        'benign_ratio': BENIGN_RATIO,
        'train_ratio': TRAIN_RATIO,
        'random_state': RANDOM_STATE,
        'note': 'Same preprocessing as LATENT, but WITHOUT encoding layer',
        'created_at': str(pd.Timestamp.now()),
    }
    
    with open(raw_exp_dir / 'raw_metadata.json', 'w') as f:
        json.dump(raw_metadata, f, indent=2)
    
    print(f"  ✓ Saved RAW metadata")
    
    # ========================================================================
    # STEP 6: TRAIN BENIGN AUTOENCODER
    # ========================================================================
    print("\n" + "="*80)
    print(" "*20 + "🧠 TRAINING BENIGN AUTOENCODER")
    print("="*80)
    
    benign_encoder, benign_decoder, benign_ae = build_autoencoder(
        input_dim=input_dim,
        latent_dim=LATENT_DIM,
        name_prefix="benign"
    )
    
    print(f"\n  Model: {benign_ae.name}")
    print(f"  Params: {benign_ae.count_params():,}")
    print(f"  Training on {len(benign_train_X):,} benign samples...")
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    benign_history = benign_ae.fit(
        benign_train_X, benign_train_X,
        epochs=AE_EPOCHS,
        batch_size=AE_BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=AE_VERBOSE
    )
    
    benign_loss = benign_ae.evaluate(benign_test_X, benign_test_X, verbose=0)
    print(f"\n  ✓ Benign AE Test Loss: {benign_loss[0]:.6f}")
    
    # ========================================================================
    # STEP 7: TRAIN MALICIOUS AUTOENCODER
    # ========================================================================
    print("\n" + "="*80)
    print(" "*18 + "🧠 TRAINING MALICIOUS AUTOENCODER")
    print("="*80)
    
    malicious_encoder, malicious_decoder, malicious_ae = build_autoencoder(
        input_dim=input_dim,
        latent_dim=LATENT_DIM,
        name_prefix="malicious"
    )
    
    print(f"\n  Model: {malicious_ae.name}")
    print(f"  Params: {malicious_ae.count_params():,}")
    print(f"  Training on {len(malicious_train_X):,} malicious samples...")
    
    malicious_history = malicious_ae.fit(
        malicious_train_X, malicious_train_X,
        epochs=AE_EPOCHS,
        batch_size=AE_BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=AE_VERBOSE
    )
    
    malicious_loss = malicious_ae.evaluate(malicious_test_X, malicious_test_X, verbose=0)
    print(f"\n  ✓ Malicious AE Test Loss: {malicious_loss[0]:.6f}")
    
    # ========================================================================
    # STEP 8: EXTRACT DUAL-VIEW LATENT FEATURES (CONCAT BOTH ENCODERS)
    # ========================================================================
    print("\n" + "="*80)
    print(" "*15 + "🔬 EXTRACTING DUAL-VIEW LATENT FEATURES")
    print("="*80)
    
    # KEY CHANGE: Pass ALL samples through BOTH encoders and concat
    print("\nExtracting with DUAL encoders (benign + malicious)...")
    print(f"  Strategy: z = [benign_enc(x), malicious_enc(x)] → {LATENT_DIM*2} dims\n")
    
    # Benign samples: pass through BOTH encoders
    print("  Processing benign samples...")
    benign_train_z_b = benign_encoder.predict(benign_train_X, verbose=0)      # 32 dims
    benign_train_z_m = malicious_encoder.predict(benign_train_X, verbose=0)  # 32 dims
    benign_train_latent = np.hstack([benign_train_z_b, benign_train_z_m])    # 64 dims
    
    benign_test_z_b = benign_encoder.predict(benign_test_X, verbose=0)
    benign_test_z_m = malicious_encoder.predict(benign_test_X, verbose=0)
    benign_test_latent = np.hstack([benign_test_z_b, benign_test_z_m])
    
    # Malicious samples: pass through BOTH encoders
    print("  Processing malicious samples...")
    malicious_train_z_b = benign_encoder.predict(malicious_train_X, verbose=0)      # 32 dims
    malicious_train_z_m = malicious_encoder.predict(malicious_train_X, verbose=0)  # 32 dims
    malicious_train_latent = np.hstack([malicious_train_z_b, malicious_train_z_m])  # 64 dims
    
    malicious_test_z_b = benign_encoder.predict(malicious_test_X, verbose=0)
    malicious_test_z_m = malicious_encoder.predict(malicious_test_X, verbose=0)
    malicious_test_latent = np.hstack([malicious_test_z_b, malicious_test_z_m])
    
    print(f"\n  ✓ Benign train latent: {benign_train_latent.shape}  (was {benign_train_z_b.shape} + {benign_train_z_m.shape})")
    print(f"  ✓ Benign test latent:  {benign_test_latent.shape}")
    print(f"  ✓ Malicious train latent: {malicious_train_latent.shape}")
    print(f"  ✓ Malicious test latent:  {malicious_test_latent.shape}")
    
    # ========================================================================
    # STEP 9: SAVE LATENT DATA
    # ========================================================================
    print(f"\n[STEP 9] Saving latent data...")
    
    # Save benign/malicious latent separately
    np.save(SPLITS_DIR / 'benign_train_latent.npy', benign_train_latent)
    np.save(SPLITS_DIR / 'benign_train_y.npy', benign_train_y)
    np.save(SPLITS_DIR / 'benign_test_latent.npy', benign_test_latent)
    np.save(SPLITS_DIR / 'benign_test_y.npy', benign_test_y)
    
    np.save(SPLITS_DIR / 'malicious_train_latent.npy', malicious_train_latent)
    np.save(SPLITS_DIR / 'malicious_train_y.npy', malicious_train_y)
    np.save(SPLITS_DIR / 'malicious_test_latent.npy', malicious_test_latent)
    np.save(SPLITS_DIR / 'malicious_test_y.npy', malicious_test_y)
    
    print(f"  ✓ Saved class-specific latent features")
    
    # Create combined datasets for experiments
    print(f"\n[STEP 10] Creating combined datasets...")
    
    # Combine train
    X_train_latent = np.vstack([benign_train_latent, malicious_train_latent])
    y_train = np.hstack([benign_train_y, malicious_train_y])
    
    # Combine test
    X_test_latent = np.vstack([benign_test_latent, malicious_test_latent])
    y_test = np.hstack([benign_test_y, malicious_test_y])
    
    # Shuffle
    rng_shuffle = np.random.RandomState(RANDOM_STATE)
    train_idx = rng_shuffle.permutation(len(X_train_latent))
    test_idx = rng_shuffle.permutation(len(X_test_latent))
    
    X_train_latent = X_train_latent[train_idx]
    y_train = y_train[train_idx]
    X_test_latent = X_test_latent[test_idx]
    y_test = y_test[test_idx]
    
    # Save Exp1 baseline with latent features
    exp1_dir = SPLITS_DIR / 'exp1_baseline_latent'
    exp1_dir.mkdir(exist_ok=True)
    
    np.save(exp1_dir / 'X_train.npy', X_train_latent)
    np.save(exp1_dir / 'y_train.npy', y_train)
    np.save(exp1_dir / 'X_test.npy', X_test_latent)
    np.save(exp1_dir / 'y_test.npy', y_test)
    
    print(f"  ✓ Exp1 latent data: Train={X_train_latent.shape}, Test={X_test_latent.shape}")
    
    # ========================================================================
    # STEP 10.5: CREATE EXP2 POISONED DATA (RAW + LATENT)
    # ========================================================================
    print(f"\n[STEP 10.5] Creating Exp2 poisoned data (RAW + LATENT)...")
    print("  Strategy: Poison at 50-dim level, then encode for LATENT")
    
    poison_rates = [5, 10, 15, 50]
    
    for poison_rate in poison_rates:
        print(f"\n  Creating {poison_rate}% poisoned data...")
        
        # Calculate number of malicious samples to flip
        malicious_indices = np.where(y_train_raw == 1)[0]
        n_poison = int(len(malicious_indices) * (poison_rate / 100.0))
        
        # Randomly select malicious samples to poison
        rng_poison = np.random.RandomState(RANDOM_STATE + poison_rate)
        poison_indices = rng_poison.choice(malicious_indices, n_poison, replace=False)
        
        # Create poisoned labels (flip malicious→benign)
        y_train_poisoned = y_train_raw.copy()
        y_train_poisoned[poison_indices] = 0  # Flip to benign
        
        print(f"    Flipped {n_poison:,} malicious → benign")
        
        # ---- RAW POISONED DATA (50 dims) ----
        raw_poison_dir = raw_exp_dir / "exp2_poisoning" / f"poison_{poison_rate:02d}"
        raw_poison_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(raw_poison_dir / 'X_train.npy', X_train_raw)  # No change to features
        np.save(raw_poison_dir / 'y_train.npy', y_train_poisoned)  # Poisoned labels
        np.save(raw_poison_dir / 'X_test.npy', X_test_raw)    # Clean test
        np.save(raw_poison_dir / 'y_test.npy', y_test_raw)    # Clean test labels
        
        print(f"    ✓ RAW poisoned saved: {raw_poison_dir}/")
        
        # ---- LATENT POISONED DATA (64 dims) ----
        # KEY FIX: Encode ALL samples, preserve EXACT poisoned labels
        # This ensures RAW and LATENT have IDENTICAL labels
        
        latent_poison_dir = SPLITS_DIR / "exp2_poisoning" / f"poison_{poison_rate:02d}"
        latent_poison_dir.mkdir(parents=True, exist_ok=True)
        
        # CRITICAL: Encode ALL X_train_raw (not split by label!)
        # Then use the EXACT SAME y_train_poisoned labels
        print(f"    Encoding ALL train samples (preserving poisoned labels)...")
        
        # Encode through both encoders
        X_train_z_b = benign_encoder.predict(X_train_raw, verbose=0)
        X_train_z_m = malicious_encoder.predict(X_train_raw, verbose=0)
        X_train_latent_poison = np.hstack([X_train_z_b, X_train_z_m])
        
        # Use EXACT same labels as RAW (already poisoned!)
        y_train_latent_poison = y_train_poisoned.copy()
        
        # Verify fairness
        assert np.array_equal(y_train_latent_poison, y_train_poisoned), \
            f"CRITICAL BUG: LATENT labels don't match RAW labels!"
        assert len(X_train_latent_poison) == len(y_train_latent_poison), \
            f"Shape mismatch: X={len(X_train_latent_poison)}, y={len(y_train_latent_poison)}"
        
        # Save latent poisoned data
        np.save(latent_poison_dir / 'X_train.npy', X_train_latent_poison)
        np.save(latent_poison_dir / 'y_train.npy', y_train_latent_poison)
        np.save(latent_poison_dir / 'X_test.npy', X_test_latent)  # Clean test (reuse from Exp1)
        np.save(latent_poison_dir / 'y_test.npy', y_test)         # Clean test labels
        
        print(f"    ✓ LATENT poisoned saved: {latent_poison_dir}/")
        print(f"    → Fair comparison: Same samples poisoned in both RAW & LATENT")
    
    print(f"\n  ✓ Exp2 poisoned data created for all rates: {poison_rates}")
    
    # ========================================================================
    print(f"\n[STEP 11] Saving models and transformers...")
    
    models_dir = SPLITS_DIR / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Save encoders/autoencoders
    benign_encoder.save(models_dir / 'benign_encoder.h5')
    benign_ae.save(models_dir / 'benign_autoencoder.h5')
    malicious_encoder.save(models_dir / 'malicious_encoder.h5')
    malicious_ae.save(models_dir / 'malicious_autoencoder.h5')
    
    # Save preprocessing transformers for GAN generation
    import joblib
    
    # Save scaler
    joblib.dump(scaler, models_dir / 'scaler.pkl')
    print(f"  ✓ Saved scaler")
    
    # Save selector (if used)
    if N_FEATURES_SELECT is not None:
        joblib.dump(selector, models_dir / 'selector.pkl')
        print(f"  ✓ Saved feature selector")
    
    # Save feature names and metadata
    preprocessing_info = {
        'common_features': common_cols,  # Features after filtering biased ones
        'n_features_after_filter': len(common_cols),
        'n_features_selected': N_FEATURES_SELECT,
        'latent_dim': LATENT_DIM,
        'exclude_features': EXCLUDE_FEATURES,
    }
    
    with open(models_dir / 'preprocessing_info.json', 'w') as f:
        json.dump(preprocessing_info, f, indent=2)
    
    print(f"  ✓ Saved preprocessing info")
    print(f"  ✓ All artifacts saved to {models_dir}/")
    
    # Save metadata
    print(f"\n[STEP 12] Saving metadata...")
    
    metadata = {
        'dataset': 'CIC-ToN-IoT',
        'approach': 'dual_encoder_concatenation',
        'total_samples': TOTAL_SAMPLES,
        'benign_ratio': BENIGN_RATIO,
        'train_ratio': TRAIN_RATIO,
        'n_benign_total': N_BENIGN,
        'n_malicious_total': N_MALICIOUS,
        'n_train_benign': N_TRAIN,
        'n_test_benign': N_TEST,
        'n_train_malicious': N_TRAIN_MAL,
        'n_test_malicious': N_TEST_MAL,
        'random_state': RANDOM_STATE,
        'original_feature_dim': input_dim,
        'n_features_selected': N_FEATURES_SELECT,
        'latent_dim_per_encoder': LATENT_DIM,
        'final_latent_dim': LATENT_DIM * 2,  # 64 dims after concat
        'ae_epochs': AE_EPOCHS,
        'ae_batch_size': AE_BATCH_SIZE,
        'benign_ae_test_loss': float(benign_loss[0]),
        'malicious_ae_test_loss': float(malicious_loss[0]),
        'note': 'Dual-view encoding: feature = [benign_enc(x), malicious_enc(x)] → 64 dims',
        'created_at': str(pd.Timestamp.now()),
    }
    
    with open(SPLITS_DIR / 'experiment_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Metadata saved")
    
    # Summary
    print("\n" + "="*80)
    print("✅ HOÀN THÀNH! Dual-View Latent Features đã sẵn sàng")
    print("="*80)
    print(f"\n📊 TỔNG KẾT:")
    print(f"  • Dataset: CIC-ToN-IoT")
    print(f"  • Approach: Dual-Encoder Concatenation")
    print(f"  • Total: {TOTAL_SAMPLES:,} samples")
    print(f"  • Benign: {N_BENIGN:,} - {N_TRAIN:,} train + {N_TEST:,} test")
    print(f"  • Malicious: {N_MALICIOUS:,} - {N_TRAIN_MAL:,} train + {N_TEST_MAL:,} test")
    print(f"  • Original features: {input_dim} (after selection from {len(common_cols)})")
    print(f"  • Latent features: {LATENT_DIM} (per encoder) × 2 = {LATENT_DIM*2} dims TOTAL")
    print(f"  • Feature composition: [benign_enc(x), malicious_enc(x)]")
    print(f"  • Benign AE loss: {benign_loss[0]:.6f}")
    print(f"  • Malicious AE loss: {malicious_loss[0]:.6f}")
    print(f"\n💾 Files tại: {SPLITS_DIR}/")
    print(f"  ✓ benign_*_latent.npy, malicious_*_latent.npy (64 dims each)")
    print(f"  ✓ exp1_baseline_latent/ (64 dims)")
    print(f"  ✓ models/ (benign_encoder.h5, malicious_encoder.h5, ...)")
    print("\n")


if __name__ == '__main__':
    main()
