"""
Preprocessing pipeline for CICIDS 2017 dataset.
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


def preprocess_cicids2017(
    raw_file: Optional[Path] = None,
    label_column: str = "class",
    drop_cols: Optional[list[str]] = None,
    output_name: str = "cicids2017_processed.csv",
    save_scaler: bool = True,
) -> PreprocessResult:
    raw_path = Path(raw_file or RAW_DIR / "CICIDS2017_dataset.csv")
    if not raw_path.exists():
        raise FileNotFoundError(f"CICIDS2017 dataset not found: {raw_path}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_tabular_dataframe(raw_path)
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' missing from dataset.")

    drop_cols = drop_cols or []
    # Don't drop the label column itself!
    if label_column in drop_cols:
        drop_cols = [c for c in drop_cols if c != label_column]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    label_series = df[label_column]
    label_encoder = None
    if not np.issubdtype(label_series.dtype, np.number):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(label_series.values)
    else:
        y = label_series.astype(int).values

    # Drop label column AND any other label-like columns to prevent data leakage
    label_like_cols = [label_column]
    for col in df.columns:
        if col != label_column and col.lower() in ['label', 'class', 'target']:
            label_like_cols.append(col)
    
    X = df.drop(columns=label_like_cols)
    print(f"\n✓ Dropped label columns: {label_like_cols}")
    print(f"✓ Remaining features: {len(X.columns)}")
    
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            X[col], _ = pd.factorize(X[col])
    X = X.astype(np.float32)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    scaled_df["label"] = y

    out_path = PROCESSED_DIR / output_name
    scaled_df.to_csv(out_path, index=False)

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
            "n_samples": int(len(df)),
            "n_features": int(X.shape[1]),
            "label_encoder": str(encoder_path) if encoder_path else None,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))

    return PreprocessResult(
        data_path=out_path,
        scaler_path=scaler_path,
        metadata_path=metadata_path,
    )


__all__ = ["preprocess_cicids2017", "PreprocessResult"]
