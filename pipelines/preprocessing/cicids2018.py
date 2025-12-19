"""
Preprocessing pipeline for CICIDS 2018 dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from .utils import load_tabular_dataframe

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "datasets" / "raw"
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
ARTEFACTS_DIR = PROCESSED_DIR / "artefacts"


@dataclass
class PreprocessResult:
    data_path: Path
    scaler_path: Optional[Path]
    metadata_path: Optional[Path]


def preprocess_cicids2018(
    raw_file: Optional[Path] = None,
    label_column: str = "Label",
    drop_cols: Optional[list[str]] = None,
    output_name: str = "cicids2018_processed.csv",
    save_scaler: bool = True,
    chunk_size: int = 50000,
    remove_bias_columns: bool = True,
) -> PreprocessResult:
    """
    Preprocess CICIDS2018 dataset with chunk processing to avoid memory issues.
    
    Args:
        raw_file: Path to raw CICIDS2018 CSV file. Defaults to datasets/raw/cicids2018.csv
        label_column: Name of the label column. Defaults to "Label"
        drop_cols: Optional list of additional columns to drop before processing
        output_name: Name of the output processed CSV file
        save_scaler: Whether to save the scaler and metadata artifacts
        chunk_size: Number of rows to process per chunk (default: 50000)
        remove_bias_columns: Whether to remove bias-prone columns (default: True)
        
    Returns:
        PreprocessResult containing paths to processed data, scaler, and metadata
    """
    raw_path = Path(raw_file or RAW_DIR / "cicids2018.csv")
    if not raw_path.exists():
        raise FileNotFoundError(f"CICIDS2018 dataset not found: {raw_path}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Starting CICIDS2018 preprocessing with chunk size: {chunk_size}")
    
    # Define bias-prone columns to remove
    # These columns can leak information and cause model to learn shortcuts
    bias_columns = [
        "Timestamp",  # Temporal information can cause overfitting
        "Dst Port",   # Specific ports might be too obvious indicators
        "Protocol",

    ]
    
    # Combine with user-specified drop columns
    drop_cols = drop_cols or []
    if remove_bias_columns:
        drop_cols = list(set(drop_cols + bias_columns))
    
    print(f"Columns to remove: {drop_cols}")
    
    # First pass: determine columns, label encoder, and collect samples for scaler fitting
    print("First pass: analyzing dataset structure...")
    first_chunk = pd.read_csv(raw_path, nrows=100000)
    
    if label_column not in first_chunk.columns:
        raise ValueError(f"Label column '{label_column}' missing from dataset.")
    
    # Drop bias columns
    first_chunk = first_chunk.drop(
        columns=[c for c in drop_cols if c in first_chunk.columns], 
        errors="ignore"
    )
    
    # Clean data
    first_chunk = first_chunk.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Prepare label encoder
    label_series = first_chunk[label_column]
    label_encoder = None
    if not np.issubdtype(label_series.dtype, np.number):
        # Need to scan entire dataset for all unique labels
        print("Scanning for unique labels...")
        unique_labels = set()
        for chunk in pd.read_csv(raw_path, chunksize=chunk_size):
            unique_labels.update(chunk[label_column].unique())
        
        label_encoder = LabelEncoder()
        label_encoder.fit(list(unique_labels))
        print(f"Found {len(label_encoder.classes_)} unique labels: {label_encoder.classes_}")
    
    # Get feature columns
    feature_cols = [c for c in first_chunk.columns if c != label_column]
    
    # Handle categorical columns in first chunk
    X_sample = first_chunk[feature_cols].copy()
    categorical_cols = X_sample.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        print(f"Converting categorical columns: {list(categorical_cols)}")
        for col in categorical_cols:
            X_sample[col], _ = pd.factorize(X_sample[col])
    X_sample = X_sample.astype(np.float32)
    
    # Fit scaler on sample
    print("Fitting scaler on sample data...")
    scaler = MinMaxScaler()
    scaler.fit(X_sample)
    
    # Second pass: process in chunks and save
    print(f"\nSecond pass: processing data in chunks...")
    out_path = PROCESSED_DIR / output_name
    
    chunk_num = 0
    total_samples = 0
    first_write = True
    
    for chunk in pd.read_csv(raw_path, chunksize=chunk_size):
        chunk_num += 1
        
        # Drop bias columns
        chunk = chunk.drop(
            columns=[c for c in drop_cols if c in chunk.columns],
            errors="ignore"
        )
        
        # Clean data
        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(chunk) == 0:
            continue
        
        # === MULTI-CLASS LABEL ENCODING ===
        # Keep multi-class labels using label encoder
        chunk_df = chunk.copy()
        
        # Encode labels using the fitted label encoder
        if label_encoder is not None:
            # String labels → numeric via encoder
            y_encoded = label_encoder.transform(chunk_df[label_column])
        else:
            # Already numeric, use as-is
            y_encoded = chunk_df[label_column].values
        
        # Process features (excluding label column)
        X = chunk_df[feature_cols].copy()
        
        # Handle categorical columns
        for col in categorical_cols:
            if col in X.columns:
                X[col], _ = pd.factorize(X[col])
        X = X.astype(np.float32)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Create output dataframe for this chunk
        scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
        scaled_df["label"] = y_encoded  # Use multi-class labels!
        
        # Save chunk
        scaled_df.to_csv(
            out_path, 
            mode='w' if first_write else 'a',
            header=first_write,
            index=False
        )
        
        total_samples += len(scaled_df)
        first_write = False
        
        if chunk_num % 10 == 0:
            print(f"  Processed chunk {chunk_num}, total samples: {total_samples:,}")
    
    print(f"\nCompleted! Total samples processed: {total_samples:,}")
    print(f"Output saved to: {out_path}")
    
    # Save artifacts
    scaler_path = metadata_path = None
    if save_scaler:
        import joblib

        base_name = output_name.replace(".csv", "")
        scaler_path = ARTEFACTS_DIR / f"{base_name}_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        
        encoder_path = None
        if label_encoder is not None:
            encoder_path = ARTEFACTS_DIR / f"{base_name}_label_encoder.pkl"
            joblib.dump(label_encoder, encoder_path)

        metadata_path = ARTEFACTS_DIR / f"{base_name}_meta.json"
        metadata = {
            "raw_path": str(raw_path.resolve()),
            "output_path": str(out_path.resolve()),
            "label_column": label_column,
            "n_samples": total_samples,
            "n_features": len(feature_cols),
            "feature_columns": feature_cols,
            "dropped_columns": drop_cols,
            "label_encoder": str(encoder_path) if encoder_path else None,
            "label_classes": label_encoder.classes_.tolist() if label_encoder else None,
            "chunk_size": chunk_size,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))
        print(f"Artifacts saved to: {ARTEFACTS_DIR}")

    return PreprocessResult(
        data_path=out_path,
        scaler_path=scaler_path,
        metadata_path=metadata_path,
    )


__all__ = ["preprocess_cicids2018", "PreprocessResult"]
