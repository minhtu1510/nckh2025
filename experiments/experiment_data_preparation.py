"""
OPTIMIZED Data preparation - Minimal RAM usage
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
SPLITS_DIR = BASE_DIR / "datasets" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def create_balanced_splits_multiclass(
    processed_csv: Path,
    train_ratio: float = 0.75,
    random_state: int = 42,
    balance_to_min_class: bool = True,
):
    """MULTI-CLASS version - split each class separately."""
    
    print("="*70)
    print("CICIDS2018 Data Preparation - MULTI-CLASS MODE")
    print("="*70)
    
    print(f"\n[1] Loading data...")
    df = pd.read_csv(processed_csv)
    
    print(f"    Total: {len(df):,} samples, {df.shape[1]-1} features")
    
    # Get all unique classes
    unique_labels = sorted(df['label'].unique())
    n_classes = len(unique_labels)
    
    print(f"\n[2] Distribution ({n_classes} classes):")
    class_dfs = {}
    class_counts = {}
    for label in unique_labels:
        class_df = df[df['label'] == label]
        class_dfs[label] = class_df
        class_counts[label] = len(class_df)
        print(f"    Class {label}: {len(class_df):,}")
    
    # Balance to minimum class size if requested
    if balance_to_min_class:
        min_count = min(class_counts.values())
        print(f"\n[3] Balancing to min class size: {min_count:,} per class")
        
        np.random.seed(random_state)
        for label in unique_labels:
            if len(class_dfs[label]) > min_count:
                class_dfs[label] = class_dfs[label].sample(n=min_count, random_state=random_state)
                print(f"    Sampled class {label}: {min_count:,}")
    
    # Split each class
    print(f"\n[4] Splitting {train_ratio*100:.0f}%/{(1-train_ratio)*100:.0f}% per class...")
    
    train_dfs = []
    test_dfs = []
    
    for label in unique_labels:
        train_part, test_part = train_test_split(
            class_dfs[label], 
            train_size=train_ratio, 
            random_state=random_state
        )
        train_dfs.append(train_part)
        test_dfs.append(test_part)
        print(f"    Class {label}: {len(train_part):,} train, {len(test_part):,} test")
    
    # Combine all classes
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    print(f"\n    Total - Train: {len(train_df):,}, Test: {len(test_df):,}")
    
    # Shuffle
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Save
    print(f"\n[5] Saving to {SPLITS_DIR}...")
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    
    train_path = SPLITS_DIR / 'cicids2018_multiclass_train.csv'
    test_path = SPLITS_DIR / 'cicids2018_multiclass_test.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Metadata
    metadata = {
        'n_classes': n_classes,
        'class_labels': [int(x) for x in unique_labels],
        'n_train': len(train_df),
        'n_test': len(test_df),
        'train_distribution': {int(k): int(v) for k, v in train_df['label'].value_counts().to_dict().items()},
        'test_distribution': {int(k): int(v) for k, v in test_df['label'].value_counts().to_dict().items()},
    }
    
    (SPLITS_DIR / 'multiclass_metadata.json').write_text(json.dumps(metadata, indent=2))
    
    print(f"\n{'='*70}")
    print("✓ Done!")
    print(f"{'='*70}\n")
    
    return {'train': train_path, 'test': test_path}, metadata


def load_experiment_data(experiment_type='baseline'):
    """Load data for experiments - MULTI-CLASS version."""
    
    # Load from multi-class splits
    train_path = SPLITS_DIR / 'cicids2018_multiclass_train.csv'
    test_path = SPLITS_DIR / 'cicids2018_multiclass_test.csv'
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Multi-class splits not found! Run: python reprocess_multiclass.py"
        )
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Shuffle
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values
    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values
    
    return X_train, y_train, X_test, y_test


def main():
    processed_csv = PROCESSED_DIR / 'cicids2018_processed.csv'
    
    if not processed_csv.exists():
        print(f"ERROR: {processed_csv} not found!")
        print("Run: python test_cicids2018_preprocessing.py")
        return
    
    create_balanced_splits_fast(processed_csv, balance_ratio=1.0)


if __name__ == '__main__':
    main()
