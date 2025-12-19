"""
Helper functions to load pre-split CICIDS2017 data
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict

BASE_DIR = Path(__file__).resolve().parents[2]
SPLITS_DIR = BASE_DIR / "datasets" / "splits" / "cicids2017"


def load_cicids2017_splits(
    return_dataframe: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pre-split CICIDS2017 train/test data.
    
    Parameters
    ----------
    return_dataframe : bool
        If True, return pandas DataFrames/Series instead of numpy arrays
        
    Returns
    -------
    X_train, X_test, y_train, y_test
        Training and test features and labels
    """
    if not SPLITS_DIR.exists():
        raise FileNotFoundError(
            f"Splits directory not found: {SPLITS_DIR}\n"
            "Run split_cicids2017.py first to create train/test splits."
        )
    
    X_train = np.load(SPLITS_DIR / "train_X.npy")
    y_train = np.load(SPLITS_DIR / "train_y.npy")
    X_test = np.load(SPLITS_DIR / "test_X.npy")
    y_test = np.load(SPLITS_DIR / "test_y.npy")
    
    if return_dataframe:
        # Convert to DataFrame for sklearn compatibility
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        y_train = pd.Series(y_train)
        y_test = pd.Series(y_test)
    
    return X_train, X_test, y_train, y_test


def get_splits_metadata() -> Dict:
    """Get metadata about the splits."""
    import json
    
    metadata_path = SPLITS_DIR / "metadata.json"
    if not metadata_path.exists():
        return {}
    
    with open(metadata_path) as f:
        return json.load(f)


def splits_exist() -> bool:
    """Check if pre-split files exist."""
    required_files = ['train_X.npy', 'train_y.npy', 'test_X.npy', 'test_y.npy']
    return all((SPLITS_DIR / f).exists() for f in required_files)


def load_benign_malicious_splits(
    split: str = 'test',
    return_dataframe: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load benign and malicious samples separately.
    
    Parameters
    ----------
    split : str
        'train' or 'test'
    return_dataframe : bool
        If True, return pandas DataFrames/Series
        
    Returns
    -------
    X_benign, y_benign, X_malicious, y_malicious
    """
    if not SPLITS_DIR.exists():
        raise FileNotFoundError(f"Splits directory not found: {SPLITS_DIR}")
    
    X_benign = np.load(SPLITS_DIR / f"{split}_benign_X.npy")
    y_benign = np.load(SPLITS_DIR / f"{split}_benign_y.npy")
    X_malicious = np.load(SPLITS_DIR / f"{split}_malicious_X.npy")
    y_malicious = np.load(SPLITS_DIR / f"{split}_malicious_y.npy")
    
    if return_dataframe:
        X_benign = pd.DataFrame(X_benign)
        y_benign = pd.Series(y_benign)
        X_malicious = pd.DataFrame(X_malicious)
        y_malicious = pd.Series(y_malicious)
    
    return X_benign, y_benign, X_malicious, y_malicious


def prepare_gan_attack_data(return_dataframe: bool = False) -> Dict:
    """
    Prepare data for GAN attack workflow.
    
    Returns dict with:
    - train_malicious: For training GAN
    - test_benign: Original benign test samples
    - test_malicious: Malicious test samples (to be replaced by adversarial)
    
    Usage for GAN attack:
    1. Train GAN on train_malicious
    2. Generate adversarial from test_malicious
    3. Evaluate on: test_benign + adversarial_samples
    """
    # Load separated data
    train_benign_X, train_benign_y, train_mal_X, train_mal_y = \
        load_benign_malicious_splits('train', return_dataframe)
    test_benign_X, test_benign_y, test_mal_X, test_mal_y = \
        load_benign_malicious_splits('test', return_dataframe)
    
    return {
        # For training classifier
        'train_X': np.concatenate([train_benign_X, train_mal_X]) if not return_dataframe 
                   else pd.concat([train_benign_X, train_mal_X]),
        'train_y': np.concatenate([train_benign_y, train_mal_y]) if not return_dataframe
                   else pd.concat([train_benign_y, train_mal_y]),
        
        # For training GAN
        'train_malicious_X': train_mal_X,
        'train_malicious_y': train_mal_y,
        
        # For evaluation
        'test_benign_X': test_benign_X,
        'test_benign_y': test_benign_y,
        'test_malicious_X': test_mal_X,
        'test_malicious_y': test_mal_y,
    }


__all__ = [
    'load_cicids2017_splits', 
    'get_splits_metadata', 
    'splits_exist',
    'load_benign_malicious_splits',
    'prepare_gan_attack_data',
]

