"""
Helper để load dữ liệu cho các thực nghiệm.
"""

import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
SPLITS_DIR = BASE_DIR / "datasets" / "splits"


def load_exp1_data():
    """
    Load dữ liệu cho Thực nghiệm 1: Baseline
    Returns: X_train, y_train, X_test, y_test
    """
    exp1_dir = SPLITS_DIR / 'exp1_baseline'
    
    if not exp1_dir.exists():
        raise FileNotFoundError(
            f"❌ Chưa chuẩn bị dữ liệu Experiment 1!\n"
            f"Hãy chạy: python prepare_experiment_data.py"
        )
    
    X_train = np.load(exp1_dir / 'X_train.npy')
    y_train = np.load(exp1_dir / 'y_train.npy')
    X_test = np.load(exp1_dir / 'X_test.npy')
    y_test = np.load(exp1_dir / 'y_test.npy')
    
    print(f"📊 Loaded Experiment 1 data:")
    print(f"  Train: {X_train.shape[0]:,} samples")
    print(f"    - Benign: {(y_train==0).sum():,}")
    print(f"    - Malicious: {(y_train==1).sum():,}")
    print(f"  Test: {X_test.shape[0]:,} samples")
    print(f"    - Benign: {(y_test==0).sum():,}")
    print(f"    - Malicious: {(y_test==1).sum():,}")
    
    return X_train, y_train, X_test, y_test


def load_exp2_data(poison_rate=0.05):
    """
    Load dữ liệu cho Thực nghiệm 2: Data Poisoning
    
    Args:
        poison_rate: 0.05, 0.10, hoặc 0.15
    
    Returns: X_train, y_train, X_test, y_test
    """
    exp2_dir = SPLITS_DIR / 'exp2_poisoning'
    
    if not exp2_dir.exists():
        raise FileNotFoundError(
            f"❌ Chưa chuẩn bị dữ liệu Experiment 2!\n"
            f"Hãy chạy: python prepare_experiment_data.py"
        )
    
    rate_str = f"{int(poison_rate*100):02d}"
    
    X_train = np.load(exp2_dir / f'X_train_poison_{rate_str}.npy')
    y_train = np.load(exp2_dir / f'y_train_poison_{rate_str}.npy')
    X_test = np.load(exp2_dir / 'X_test.npy')
    y_test = np.load(exp2_dir / 'y_test.npy')
    
    print(f"📊 Loaded Experiment 2 data (poison rate = {poison_rate*100:.0f}%):")
    print(f"  Train: {X_train.shape[0]:,} samples (poisoned)")
    print(f"    - Benign: {(y_train==0).sum():,}")
    print(f"    - Malicious: {(y_train==1).sum():,}")
    print(f"  Test: {X_test.shape[0]:,} samples (clean)")
    print(f"    - Benign: {(y_test==0).sum():,}")
    print(f"    - Malicious: {(y_test==1).sum():,}")
    
    return X_train, y_train, X_test, y_test


def load_exp3_train_data():
    """
    Load training data cho Thực nghiệm 3: GAN Attack
    (Giống như Exp1, nhưng riêng hàm cho rõ ràng)
    
    Returns: X_train, y_train
    """
    exp1_dir = SPLITS_DIR / 'exp1_baseline'
    
    if not exp1_dir.exists():
        raise FileNotFoundError(
            f"❌ Chưa chuẩn bị dữ liệu!\n"
            f"Hãy chạy: python prepare_experiment_data.py"
        )
    
    X_train = np.load(exp1_dir / 'X_train.npy')
    y_train = np.load(exp1_dir / 'y_train.npy')
    
    print(f"📊 Loaded Experiment 3 training data:")
    print(f"  Train: {X_train.shape[0]:,} samples")
    print(f"    - Benign: {(y_train==0).sum():,}")
    print(f"    - Malicious: {(y_train==1).sum():,}")
    
    return X_train, y_train


def load_base_data():
    """
    Load base data đã được chia sẵn (cho GAN training).
    
    Returns: dict with keys ['benign_train', 'benign_test', 'malicious_train', 'malicious_test']
    Each value is tuple (X, y)
    """
    data = {}
    for name in ['benign_train', 'benign_test', 'malicious_train', 'malicious_test']:
        X_path = SPLITS_DIR / f'{name}_X.npy'
        y_path = SPLITS_DIR / f'{name}_y.npy'
        
        if not X_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"❌ Không tìm thấy {name} data!\n"
                f"Hãy chạy: python prepare_experiment_data.py"
            )
        
        data[name] = (np.load(X_path), np.load(y_path))
    
    return data


# Backward compatibility
def load_experiment_data(experiment_type='baseline'):
    """
    Deprecated: Dùng load_exp1_data(), load_exp2_data(), hoặc load_exp3_train_data() thay thế.
    
    Giữ lại để tương thích với code cũ.
    """
    if experiment_type == 'baseline':
        return load_exp1_data()
    elif experiment_type.startswith('poison'):
        # Extract poison rate if specified
        if '_' in experiment_type:
            rate = float(experiment_type.split('_')[1]) / 100
        else:
            rate = 0.05
        return load_exp2_data(poison_rate=rate)
    else:
        return load_exp1_data()
