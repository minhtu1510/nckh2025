"""
Preprocessing pipeline for TRAbID 2017 dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "datasets" / "raw"
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
ARTEFACTS_DIR = PROCESSED_DIR / "artefacts"


@dataclass
class PreprocessResult:
    data_path: Path
    scaler_path: Optional[Path]
    metadata_path: Optional[Path]


def preprocess_trabid(
    feature_file: Optional[Path] = None,
    label_file: Optional[Path] = None,
    output_name: str = "trabid2017_processed.csv",
    save_scaler: bool = True,
) -> PreprocessResult:
    feature_path = Path(feature_file or RAW_DIR / "TRAbID2017_dataset.arff")
    label_path = Path(label_file or RAW_DIR / "TRAbID2017_dataset_Y_class.csv")

    if not feature_path.exists():
        raise FileNotFoundError(f"TRAbID feature file not found: {feature_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"TRAbID label file not found: {label_path}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

    data = arff.loadarff(str(feature_path))
    dataset = pd.DataFrame(data[0])
    labels = pd.read_csv(label_path, header=None)[0]

    if len(dataset) != len(labels):
        raise ValueError("Feature and label files have mismatched lengths.")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(dataset.values.astype(np.float32))
    scaled_df = pd.DataFrame(scaled, columns=dataset.columns)
    scaled_df["label"] = labels.values

    out_path = PROCESSED_DIR / output_name
    scaled_df.to_csv(out_path, index=False)

    scaler_path = metadata_path = None
    if save_scaler:
        import joblib

        scaler_path = ARTEFACTS_DIR / f"{output_name.replace('.csv', '')}_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        metadata_path = ARTEFACTS_DIR / f"{output_name.replace('.csv', '')}_meta.json"
        metadata = {
            "feature_file": str(feature_path.resolve()),
            "label_file": str(label_path.resolve()),
            "output_path": str(out_path.resolve()),
            "n_samples": int(len(dataset)),
            "n_features": int(dataset.shape[1]),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))

    return PreprocessResult(
        data_path=out_path,
        scaler_path=scaler_path,
        metadata_path=metadata_path,
    )


__all__ = ["preprocess_trabid", "PreprocessResult"]
