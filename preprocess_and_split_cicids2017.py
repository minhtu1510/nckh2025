#!/usr/bin/env python3
"""
Preprocessing + Split CICIDS2017 (Chỉ dùng numpy + pandas)

Quy trình:
1. Load raw data
2. Feature selection (manual implementation)
3. Normalization (Min-Max)
4. Split train/test
5. Save
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

np.random.seed(42)

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "datasets" / "raw"
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
SPLITS_DIR = BASE_DIR / "datasets" / "splits" / "cicids2017"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("🔧 PREPROCESSING + SPLIT CICIDS2017")
print("="*80)
print()

# ===== 1. LOAD RAW DATA =====
print("📂 Step 1: Loading raw data...")
df = pd.read_csv(RAW_DIR / "CICIDS2017_dataset.csv")
print(f"✓ Loaded: {len(df):,} samples, {df.shape[1]} columns")
print()

# ===== 2. PREPARE LABELS =====
print("🏷️  Step 2: Processing labels...")
if 'class' in df.columns:
    label_col = 'class'
elif 'Label' in df.columns:
    label_col = 'Label'
else:
    raise ValueError("No label column found!")

# Check if labels are numeric
if pd.api.types.is_numeric_dtype(df[label_col]):
    # Already numeric (0/1)
    df['label'] = df[label_col].astype(int)
    print(f"✓ Found numeric labels in '{label_col}' column")
else:
    # String labels, convert benign=0, others=1
    df['label'] = (df[label_col].astype(str).str.lower() != 'benign').astype(int)
    print(f"✓ Converted string labels in '{label_col}' column")

print(f"✓ Label distribution:")
print(f"   Benign (0):    {(df['label']==0).sum():,}")
print(f"   Malicious (1): {(df['label']==1).sum():,}")
print()

# Drop label columns, keep features
features_to_drop = []
if 'Label' in df.columns:
    features_to_drop.append('Label')
if 'class' in df.columns:
    features_to_drop.append('class')

X = df.drop(columns=features_to_drop + ['label'])
y = df['label'].values

print(f"✓ Initial features: {X.shape[1]}")
print()

# ===== 3. HANDLE MISSING VALUES =====
print("🧹 Step 3: Handling missing values...")
X = X.replace([np.inf, -np.inf], np.nan)
missing_before = X.isnull().sum().sum()
X = X.fillna(0)
print(f"✓ Filled {missing_before:,} missing values with 0")
print()

# ===== 4. FEATURE SELECTION =====
print("✂️  Step 4: Feature Selection...")
print()

# 4.1: Remove constant features
print("   4.1: Removing constant features...")
variances = X.var()
constant_features = variances[variances == 0].index.tolist()
X = X.drop(columns=constant_features)
print(f"   ✓ Removed {len(constant_features)} constant features")
print(f"   ✓ Remaining: {X.shape[1]} features")
print()

# 4.2: Remove low variance features (< 0.01)
print("   4.2: Removing low variance features (< 0.01)...")
variances = X.var()
low_var_features = variances[variances < 0.01].index.tolist()
X = X.drop(columns=low_var_features)
print(f"   ✓ Removed {len(low_var_features)} low variance features")
print(f"   ✓ Remaining: {X.shape[1]} features")
print()

# 4.3: Remove highly correlated features (> 0.95)
print("   4.3: Removing highly correlated features (>0.95)...")
corr_matrix = X.corr().abs()

# Find pairs of correlated features
to_drop = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.95:
            colname = corr_matrix.columns[j]
            to_drop.add(colname)

to_drop = list(to_drop)
X = X.drop(columns=to_drop)
print(f"   ✓ Removed {len(to_drop)} highly correlated features")
print(f"   ✓ Remaining: {X.shape[1]} features")
print()

print(f"📊 Feature selection summary:")
print(f"   Initial:    80 features")
print(f"   Constant:   -{len(constant_features)}")
print(f"   Low var:    -{len(low_var_features)}")
print(f"   Correlated: -{len(to_drop)}")
print(f"   ------------")
print(f"   Final:      {X.shape[1]} features")
print()

# ===== 5. NORMALIZATION (Min-Max) =====
print("📏 Step 5: Min-Max Normalization...")
X_min = X.min()
X_max = X.max()
X_normalized = (X - X_min) / (X_max - X_min + 1e-8)  # Avoid division by zero
X = X_normalized
print(f"✓ All features scaled to [0, 1]")
print()

# ===== 6. SPLIT TRAIN/TEST (Stratified) =====
print("✂️  Step 6: Splitting train/test (80/20, stratified)...")

# Manual stratified split
benign_idx = np.where(y == 0)[0]
malicious_idx = np.where(y == 1)[0]

np.random.shuffle(benign_idx)
np.random.shuffle(malicious_idx)

# 80/20 split
n_benign_train = int(len(benign_idx) * 0.8)
n_malicious_train = int(len(malicious_idx) * 0.8)

train_idx = np.concatenate([
    benign_idx[:n_benign_train],
    malicious_idx[:n_malicious_train]
])
test_idx = np.concatenate([
    benign_idx[n_benign_train:],
    malicious_idx[n_malicious_train:]
])

# Shuffle
np.random.shuffle(train_idx)
np.random.shuffle(test_idx)

# Split
X_values = X.values
X_train = X_values[train_idx]
y_train = y[train_idx]
X_test = X_values[test_idx]
y_test = y[test_idx]

print(f"✓ Train: {X_train.shape}")
print(f"   Benign:    {(y_train==0).sum():,} ({(y_train==0).sum()/len(y_train)*100:.2f}%)")
print(f"   Malicious: {(y_train==1).sum():,} ({(y_train==1).sum()/len(y_train)*100:.2f}%)")
print()
print(f"✓ Test: {X_test.shape}")
print(f"   Benign:    {(y_test==0).sum():,} ({(y_test==0).sum()/len(y_test)*100:.2f}%)")
print(f"   Malicious: {(y_test==1).sum():,} ({(y_test==1).sum()/len(y_test)*100:.2f}%)")
print()

# ===== 7. SAVE SPLITS =====
print("💾 Step 7: Saving processed splits...")

# Full train/test
np.save(SPLITS_DIR / "train_X.npy", X_train)
np.save(SPLITS_DIR / "train_y.npy", y_train)
np.save(SPLITS_DIR / "test_X.npy", X_test)
np.save(SPLITS_DIR / "test_y.npy", y_test)
print(f"✓ Saved full splits")

# Separate benign/malicious
train_benign_mask = y_train == 0
train_malicious_mask = y_train == 1
test_benign_mask = y_test == 0
test_malicious_mask = y_test == 1

np.save(SPLITS_DIR / "train_benign_X.npy", X_train[train_benign_mask])
np.save(SPLITS_DIR / "train_benign_y.npy", y_train[train_benign_mask])
np.save(SPLITS_DIR / "train_malicious_X.npy", X_train[train_malicious_mask])
np.save(SPLITS_DIR / "train_malicious_y.npy", y_train[train_malicious_mask])

np.save(SPLITS_DIR / "test_benign_X.npy", X_test[test_benign_mask])
np.save(SPLITS_DIR / "test_benign_y.npy", y_test[test_benign_mask])
np.save(SPLITS_DIR / "test_malicious_X.npy", X_test[test_malicious_mask])
np.save(SPLITS_DIR / "test_malicious_y.npy", y_test[test_malicious_mask])
print(f"✓ Saved benign/malicious splits")
print()

# ===== 8. SAVE METADATA =====
print("📝 Step 8: Saving metadata...")

metadata = {
    'raw_samples': len(df),
    'raw_features': 80,
    'final_features': X.shape[1],
    'feature_names': X.columns.tolist(),
    'removed_features': {
        'constant': len(constant_features),
        'low_variance': len(low_var_features),
        'highly_correlated': len(to_drop),
    },
    'preprocessing': {
        'missing_filled': int(missing_before),
        'normalization': 'MinMaxScaler [0, 1]',
        'variance_threshold': 0.01,
        'correlation_threshold': 0.95,
    },
    'split': {
        'test_ratio': 0.2,
        'random_state': 42,
        'stratified': True,
    },
    'train_size': len(X_train),
    'test_size': len(X_test),
    'class_distribution': {
        'train_benign': int((y_train==0).sum()),
        'train_malicious': int((y_train==1).sum()),
        'test_benign': int((y_test==0).sum()),
        'test_malicious': int((y_test==1).sum()),
    }
}

with open(SPLITS_DIR / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Saved metadata.json")
print()

# ===== 9. SUMMARY =====
print("="*80)
print("✅ PREPROCESSING + SPLIT COMPLETE!")
print("="*80)
print()
print(f"📊 Summary:")
print(f"   Original features:  80")
print(f"   Final features:     {X.shape[1]}")
print(f"   Reduction:          {80 - X.shape[1]} features ({(80-X.shape[1])/80*100:.1f}%)")
print()
print(f"   Dataset:")
print(f"   - Total samples:    {len(df):,}")
print(f"   - Train:            {len(X_train):,}")
print(f"   - Test:             {len(X_test):,}")
print()
print(f"📁 Output location: {SPLITS_DIR}/")
print()
print(f"✨ Data is now ready for experiments!")
print(f"   - Cleaned (no missing/inf)")
print(f"   - Optimized ({X.shape[1]} features)")
print(f"   - Normalized [0, 1]")
print(f"   - Stratified split")
print()
