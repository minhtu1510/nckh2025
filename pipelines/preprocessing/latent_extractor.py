"""
Preprocessing pipeline that extracts latent features from an Autoencoder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from .cicids2017 import preprocess_cicids2017, PreprocessResult
from ids_research.models.advanced.autoencoder import train_autoencoder_detector

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
ARTEFACTS_DIR = PROCESSED_DIR / "artefacts"


def extract_latent_features(
    raw_file: Optional[Path] = None,
    output_name: str = "cicids2017_latent.csv",
    ae_epochs: int = 20,
    run_id: str = "latent_extraction",
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreprocessResult:
    """
    1. Preprocess CICIDS2017 (standard scaling/encoding).
    2. Split into train/test (SAME random_state as baseline).
    3. Train Autoencoder on TRAIN set.
    4. Extract latent features for BOTH train and test.
    5. Save maintaining the SAME split order.
    """
    from sklearn.model_selection import train_test_split
    
    # 1. Standard Preprocessing
    print("Step 1: Standard Preprocessing...")
    base_result = preprocess_cicids2017(raw_file=raw_file)
    processed_csv = base_result.data_path
    
    # 2. Load and Split (SAME as baseline)
    print(f"Step 2: Splitting data (random_state={random_state})...")
    df = pd.read_csv(processed_csv)
    label_col = "label"
    
    X = df.drop(columns=[label_col]).values.astype(np.float32)
    y = df[label_col].values
    
    # IMPORTANT: Use SAME random_state as baseline models!
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,  # ← Must match baseline!
        stratify=y
    )
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # 3. Train Autoencoder on TRAIN set
    print("Step 3: Training Autoencoder on TRAIN set...")
    ae_results = train_autoencoder_detector(
        processed_csv=processed_csv,
        run_id=run_id,
        epochs=ae_epochs,
        test_size=test_size,
        random_state=random_state  # ← Same split!
    )
    encoder_path = ae_results["encoder_path"]
    
    # 4. Load Encoder and Extract Latent
    print("Step 4: Extracting Latent Features...")
    encoder = load_model(encoder_path)
    
    # Extract for train and test separately
    X_train_latent = encoder.predict(X_train, verbose=0)
    X_test_latent = encoder.predict(X_test, verbose=0)
    print(f"   Latent dim: {X_train_latent.shape[1]}")
    
    # 5. Combine back in SAME order as split
    # Concatenate train then test (same as baseline split)
    X_latent_all = np.vstack([X_train_latent, X_test_latent])
    y_all = np.concatenate([y_train, y_test])
    
    # Create DataFrame
    latent_cols = [f"latent_{i}" for i in range(X_latent_all.shape[1])]
    latent_df = pd.DataFrame(X_latent_all, columns=latent_cols)
    latent_df[label_col] = y_all
    
    # 6. Save
    out_path = PROCESSED_DIR / output_name
    latent_df.to_csv(out_path, index=False)
    print(f"✅ Latent dataset saved: {out_path}")
    print(f"   Shape: {latent_df.shape}")
    print(f"   Maintains train/test split order (train first, then test)")
    
    # Save metadata
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = ARTEFACTS_DIR / f"{output_name.replace('.csv', '')}_meta.json"
    metadata = {
        "original_dataset": str(processed_csv),
        "encoder_path": str(encoder_path),
        "n_samples": len(latent_df),
        "n_features": len(latent_cols),
        "latent_dim": X_latent_all.shape[1],
        "train_size": len(X_train),
        "test_size": len(X_test),
        "random_state": random_state,
        "split_method": "train_test_split with stratify",
        "note": "Data order: train samples first, then test samples"
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"✅ Metadata saved: {metadata_path}")
    
    return PreprocessResult(
        data_path=out_path,
        scaler_path=None,
        metadata_path=metadata_path
    )

__all__ = ["extract_latent_features"]
