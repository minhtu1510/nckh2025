#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ----------------------------
# Helpers
# ----------------------------
def load_schema_feature_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "feature_names" in obj:
        names = obj["feature_names"]
    elif isinstance(obj, list):
        names = obj
    else:
        raise ValueError(f"Schema file {path} must be a list[str] or dict with key 'feature_names'")
    if not names:
        raise ValueError(f"Schema {path} has empty feature_names. Fill it first.")
    if len(names) != len(set(names)):
        raise ValueError(f"Schema {path} has duplicated feature names.")
    return names


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_str(x) -> str:
    return str(x).strip().lower()


def binarize_label(label_val) -> int:
    """
    Make y_binary from Label column.
    Accept common variants: Benign/Attack, 0/1, True/False, etc.
    """
    s = normalize_str(label_val)
    if s in {"benign", "normal", "0", "false", "f", "no"}:
        return 0
    if s in {"attack", "malicious", "1", "true", "t", "yes"}:
        return 1
    # fallback: if it looks like a number
    try:
        v = float(s)
        return 1 if v != 0 else 0
    except Exception:
        # if unknown string, treat as attack (safer for IDS training)
        return 1


def auto_family_map(attack_names: List[str]) -> Dict[str, str]:
    """
    Automatically map attack labels to 7 families using keywords.
    Anything not matched -> Other.

    Families (7):
      - DDoS
      - DoS
      - BruteForce
      - WebAttack
      - Bot
      - Infiltration
      - Other
    """
    mapping = {}
    for a in attack_names:
        s = normalize_str(a)

        # benign
        if s in {"benign", "normal"}:
            mapping[a] = "Benign"
            continue

        # DDoS
        if "ddos" in s or "hoic" in s or "loic" in s:
            mapping[a] = "DDoS"
            continue

        # DoS
        if "dos" in s or "hulk" in s or "goldeneye" in s or "slowloris" in s or "slowhttptest" in s:
            mapping[a] = "DoS"
            continue

        # Brute force
        if "brute" in s or "bruteforce" in s or "ssh" in s or "ftp" in s:
            mapping[a] = "BruteForce"
            continue

        # Web
        if "xss" in s or "sql" in s or "injection" in s or "web" in s:
            mapping[a] = "WebAttack"
            continue

        # Bot
        if "bot" in s:
            mapping[a] = "Bot"
            continue

        # Infiltration
        if "infil" in s:
            mapping[a] = "Infiltration"
            continue

        mapping[a] = "Other"

    # remove Benign from the 7 families used for attack-only training,
    # but keep mapping key for completeness
    return mapping


def clean_features(df: pd.DataFrame, feature_names: List[str], fillna: str = "zero") -> np.ndarray:
    """
    Extract and clean features in the exact order of feature_names.
    - replace inf/-inf -> NaN
    - fill NaN (zero or median)
    - cast to float32
    """
    Xdf = df[feature_names].copy()

    # convert everything numeric if possible
    for c in feature_names:
        if not np.issubdtype(Xdf[c].dtype, np.number):
            Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")

    Xdf = Xdf.replace([np.inf, -np.inf], np.nan)

    if fillna == "median":
        med = Xdf.median(numeric_only=True)
        Xdf = Xdf.fillna(med)
    else:
        Xdf = Xdf.fillna(0)

    return Xdf.to_numpy(dtype=np.float32, copy=False)


@dataclass
class BuildConfig:
    csv_path: str
    schema_binary: str
    schema_family: str
    out_root: str
    test_size: float
    seed: int
    label_col: str
    attack_col: str
    fillna: str
    family_map_path: Optional[str]
    save_family_map_to: Optional[str]
    attack_only_family: bool


def save_split(out_dir: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> None:
    ensure_outdir(out_dir)
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to NF-CICIDS2018-v3.csv")
    ap.add_argument("--schema-binary", required=True, help="schemas/v2/binary.json")
    ap.add_argument("--schema-family", required=True, help="schemas/v2/family.json")
    ap.add_argument("--out-root", required=True, help="datasets/v2")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label-col", default="Label")
    ap.add_argument("--attack-col", default="Attack")
    ap.add_argument("--fillna", choices=["zero", "median"], default="zero")

    ap.add_argument("--family-map", default=None, help="Optional JSON mapping Attack->Family. If omitted, auto-map by keywords.")
    ap.add_argument("--save-family-map-to", default=None, help="Save the mapping JSON to this path (recommended).")
    ap.add_argument("--attack-only-family", action="store_true", help="Train family only on attack rows (default).")

    args = ap.parse_args()
    cfg = BuildConfig(
        csv_path=args.csv,
        schema_binary=args.schema_binary,
        schema_family=args.schema_family,
        out_root=args.out_root,
        test_size=args.test_size,
        seed=args.seed,
        label_col=args.label_col,
        attack_col=args.attack_col,
        fillna=args.fillna,
        family_map_path=args.family_map,
        save_family_map_to=args.save_family_map_to,
        attack_only_family=True if args.attack_only_family else True,  # default True
    )

    # Load schemas
    binary_feats = load_schema_feature_names(cfg.schema_binary)
    family_feats = load_schema_feature_names(cfg.schema_family)

    # Read CSV (only needed columns for memory efficiency)
    needed_cols = set(binary_feats) | set(family_feats) | {cfg.label_col, cfg.attack_col}
    print(f"[BUILD] Loading CSV: {cfg.csv_path}")
    df = pd.read_csv(cfg.csv_path, usecols=lambda c: c in needed_cols)

    # Build binary labels
    if cfg.label_col not in df.columns:
        raise ValueError(f"Label column '{cfg.label_col}' not found in CSV.")
    y_binary = df[cfg.label_col].apply(binarize_label).to_numpy(dtype=np.int64)

    # ----------------------------
    # Binary dataset
    # ----------------------------
    print(f"[BINARY] Extracting {len(binary_feats)} features...")
    Xb = clean_features(df, binary_feats, fillna=cfg.fillna)

    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        Xb, y_binary, test_size=cfg.test_size, random_state=cfg.seed, stratify=y_binary
    )

    out_bin = os.path.join(cfg.out_root, "binary")
    save_split(out_bin, Xb_train, yb_train, Xb_test, yb_test)
    meta_bin = {
        "task": "binary",
        "schema": cfg.schema_binary,
        "n_features": int(Xb.shape[1]),
        "n_train": int(Xb_train.shape[0]),
        "n_test": int(Xb_test.shape[0]),
        "label_counts_train": {str(k): int(v) for k, v in zip(*np.unique(yb_train, return_counts=True))},
        "label_counts_test": {str(k): int(v) for k, v in zip(*np.unique(yb_test, return_counts=True))},
        "fillna": cfg.fillna,
        "seed": cfg.seed,
    }
    with open(os.path.join(out_bin, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta_bin, f, indent=2)
    print(f"[BINARY] Saved to {out_bin}")

    # ----------------------------
    # Family dataset (attack-only)
    # ----------------------------
    if cfg.attack_col not in df.columns:
        raise ValueError(f"Attack column '{cfg.attack_col}' not found in CSV.")

    # build mapping Attack -> Family
    if cfg.family_map_path:
        with open(cfg.family_map_path, "r", encoding="utf-8") as f:
            family_map = json.load(f)
        if not isinstance(family_map, dict):
            raise ValueError("--family-map must be a JSON object mapping attack label -> family")
    else:
        uniq_attacks = sorted(df[cfg.attack_col].astype(str).unique().tolist())
        family_map = auto_family_map(uniq_attacks)

    if cfg.save_family_map_to:
        ensure_outdir(os.path.dirname(cfg.save_family_map_to))
        with open(cfg.save_family_map_to, "w", encoding="utf-8") as f:
            json.dump(family_map, f, indent=2, ensure_ascii=False)
        print(f"[FAMILY] Saved family map to {cfg.save_family_map_to}")

    # filter attack rows for attack-only
    attack_mask = (y_binary == 1)
    df_a = df.loc[attack_mask].copy()
    if df_a.empty:
        raise RuntimeError("No attack rows found after binarization. Check Label column mapping.")

    # map Attack -> Family
    fam = df_a[cfg.attack_col].astype(str).map(lambda x: family_map.get(x, "Other"))
    # remove accidental 'Benign' in attack-only set
    fam = fam.map(lambda x: "Other" if normalize_str(x) == "benign" else x)

    # extract family features
    print(f"[FAMILY] Attack-only rows: {df_a.shape[0]}")
    print(f"[FAMILY] Extracting {len(family_feats)} features...")
    Xf = clean_features(df_a, family_feats, fillna=cfg.fillna)

    # encode families
    le = LabelEncoder()
    y_family = le.fit_transform(fam.to_numpy(dtype=str))
    family_classes = le.classes_.tolist()
    print(f"[FAMILY] Classes ({len(family_classes)}): {family_classes}")

    Xf_train, Xf_test, yf_train, yf_test = train_test_split(
        Xf, y_family, test_size=cfg.test_size, random_state=cfg.seed, stratify=y_family
    )

    out_fam = os.path.join(cfg.out_root, "family")
    save_split(out_fam, Xf_train, yf_train, Xf_test, yf_test)

    # save label encoder
    import joblib
    joblib.dump(le, os.path.join(out_fam, "label_encoder.pkl"))

    meta_fam = {
        "task": "family_attack_only",
        "schema": cfg.schema_family,
        "n_features": int(Xf.shape[1]),
        "n_train": int(Xf_train.shape[0]),
        "n_test": int(Xf_test.shape[0]),
        "classes": family_classes,
        "fillna": cfg.fillna,
        "seed": cfg.seed,
        "note": "Family classifier is trained on ATTACK rows only; run it only when binary p_attack is high.",
    }
    with open(os.path.join(out_fam, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta_fam, f, indent=2, ensure_ascii=False)

    print(f"[FAMILY] Saved to {out_fam}")
    print("[DONE] Build features v2 complete.")


if __name__ == "__main__":
    main()

