#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.lib.format import open_memmap

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib


def load_schema_feature_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "feature_names" in obj:
        names = obj["feature_names"]
    elif isinstance(obj, list):
        names = obj
    else:
        raise ValueError(f"Schema {path} must be list[str] or dict{{feature_names:[]}}")
    if not names:
        raise ValueError(f"Schema {path} has empty feature_names")
    if len(names) != len(set(names)):
        raise ValueError(f"Schema {path} has duplicate feature names")
    return names


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def normalize_str(x) -> str:
    return str(x).strip().lower()


def binarize_label(label_val) -> int:
    s = normalize_str(label_val)
    if s in {"benign", "normal", "0", "false", "f", "no"}:
        return 0
    if s in {"attack", "malicious", "1", "true", "t", "yes"}:
        return 1
    try:
        v = float(s)
        return 1 if v != 0 else 0
    except Exception:
        return 1


def auto_family_map(attack_names: List[str]) -> Dict[str, str]:
    mapping = {}
    for a in attack_names:
        s = normalize_str(a)
        if s in {"benign", "normal"}:
            mapping[a] = "Benign"
            continue

        if "ddos" in s or "hoic" in s or "loic" in s:
            mapping[a] = "DDoS"
        elif "dos" in s or "hulk" in s or "goldeneye" in s or "slowloris" in s or "slowhttptest" in s:
            mapping[a] = "DoS"
        elif "brute" in s or "bruteforce" in s or "ssh" in s or "ftp" in s:
            mapping[a] = "BruteForce"
        elif "xss" in s or "sql" in s or "injection" in s or "web" in s:
            mapping[a] = "WebAttack"
        elif "bot" in s:
            mapping[a] = "Bot"
        elif "infil" in s:
            mapping[a] = "Infiltration"
        else:
            mapping[a] = "Other"
    return mapping


def clean_features_chunk(df: pd.DataFrame, feature_names: List[str], fillna: str) -> np.ndarray:
    Xdf = df[feature_names].copy()

    # numeric coercion
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


def streaming_stratified_split_decider(
    y: np.ndarray,
    remain_total: np.ndarray,
    remain_test: np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Decide test/train for each sample in y (int class ids) in a streaming stratified way:
    prob(test | class c) = remain_test[c] / remain_total[c]
    Ensures near-target class proportions without holding all rows.
    """
    out_is_test = np.zeros(len(y), dtype=bool)
    for i, c in enumerate(y):
        c = int(c)
        if remain_total[c] <= 0:
            out_is_test[i] = False
            continue
        if remain_test[c] <= 0:
            out_is_test[i] = False
        elif remain_test[c] == remain_total[c]:
            out_is_test[i] = True
        else:
            p = remain_test[c] / remain_total[c]
            out_is_test[i] = (rng.random() < p)

        # update remaining
        remain_total[c] -= 1
        if out_is_test[i]:
            remain_test[c] -= 1
    return out_is_test


def pass_collect_attacks(csv_path: str, label_col: str, attack_col: str, usecols: List[str], chunksize: int) -> List[str]:
    attacks = set()
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        if attack_col in chunk.columns:
            attacks.update(chunk[attack_col].astype(str).unique().tolist())
    return sorted(attacks)


def pass_count_classes(
    csv_path: str,
    label_col: str,
    attack_col: str,
    family_map: Dict[str, str],
    usecols: List[str],
    chunksize: int,
    attack_only_family: bool
) -> Tuple[np.ndarray, Dict[str, int]]:
    # binary counts
    bin_counts = np.zeros(2, dtype=np.int64)

    # family counts by name (string)
    fam_counts: Dict[str, int] = {}

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        yb = chunk[label_col].apply(binarize_label).to_numpy(dtype=np.int64)
        bin_counts[0] += int((yb == 0).sum())
        bin_counts[1] += int((yb == 1).sum())

        if attack_only_family:
            chunk = chunk.loc[yb == 1]
            if chunk.empty:
                continue

        fam = chunk[attack_col].astype(str).map(lambda x: family_map.get(x, "Other"))
        fam = fam.map(lambda x: "Other" if normalize_str(x) == "benign" else x)
        vc = fam.value_counts()
        for k, v in vc.items():
            fam_counts[str(k)] = fam_counts.get(str(k), 0) + int(v)

    return bin_counts, fam_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--schema-binary", required=True)
    ap.add_argument("--schema-family", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label-col", default="Label")
    ap.add_argument("--attack-col", default="Attack")
    ap.add_argument("--fillna", choices=["zero", "median"], default="zero")
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--family-map", default=None)
    ap.add_argument("--save-family-map-to", default=None)
    ap.add_argument("--attack-only-family", action="store_true")
    args = ap.parse_args()

    binary_feats = load_schema_feature_names(args.schema_binary)
    family_feats = load_schema_feature_names(args.schema_family)

    # only read needed columns to reduce RAM
    needed_cols = sorted(set(binary_feats) | set(family_feats) | {args.label_col, args.attack_col})

    rng = np.random.default_rng(args.seed)

    print(f"[PASS1] collect unique Attack labels (chunksize={args.chunksize})")
    if args.family_map:
        with open(args.family_map, "r", encoding="utf-8") as f:
            family_map = json.load(f)
        if not isinstance(family_map, dict):
            raise ValueError("--family-map must be a JSON object Attack->Family")
    else:
        attacks = pass_collect_attacks(args.csv, args.label_col, args.attack_col, needed_cols, args.chunksize)
        family_map = auto_family_map(attacks)

    if args.save_family_map_to:
        ensure_dir(os.path.dirname(args.save_family_map_to))
        with open(args.save_family_map_to, "w", encoding="utf-8") as f:
            json.dump(family_map, f, indent=2, ensure_ascii=False)
        print(f"[PASS1] saved family map -> {args.save_family_map_to}")

    print("[PASS2] count classes for stratified streaming split")
    bin_counts, fam_counts = pass_count_classes(
        args.csv, args.label_col, args.attack_col, family_map,
        needed_cols, args.chunksize,
        attack_only_family=True if args.attack_only_family else True  # default True
    )

    # family label encoder (7 groups typically)
    fam_labels = sorted(fam_counts.keys())
    le = LabelEncoder()
    le.fit(fam_labels)

    out_bin = os.path.join(args.out_root, "binary")
    out_fam = os.path.join(args.out_root, "family")
    ensure_dir(out_bin); ensure_dir(out_fam)

    # desired test counts
    bin_test_target = np.floor(bin_counts * args.test_size).astype(np.int64)
    fam_counts_arr = np.array([fam_counts[k] for k in le.classes_], dtype=np.int64)
    fam_test_target = np.floor(fam_counts_arr * args.test_size).astype(np.int64)

    # train sizes
    bin_train_n = int(bin_counts.sum() - bin_test_target.sum())
    bin_test_n  = int(bin_test_target.sum())
    fam_train_n = int(fam_counts_arr.sum() - fam_test_target.sum())
    fam_test_n  = int(fam_test_target.sum())

    print(f"[ALLOC] binary: total={bin_counts.sum()} train={bin_train_n} test={bin_test_n} features={len(binary_feats)}")
    print(f"[ALLOC] family(attack-only): total={fam_counts_arr.sum()} train={fam_train_n} test={fam_test_n} features={len(family_feats)}")
    print(f"[FAMILY] classes={list(le.classes_)}")

    # allocate memmaps (no RAM explosion)
    Xb_train = open_memmap(os.path.join(out_bin, "X_train.npy"), mode="w+", dtype=np.float32,
                        shape=(bin_train_n, len(binary_feats)))
    yb_train = open_memmap(os.path.join(out_bin, "y_train.npy"), mode="w+", dtype=np.int64,
                        shape=(bin_train_n,))
    Xb_test  = open_memmap(os.path.join(out_bin, "X_test.npy"), mode="w+", dtype=np.float32,
                        shape=(bin_test_n,  len(binary_feats)))
    yb_test  = open_memmap(os.path.join(out_bin, "y_test.npy"), mode="w+", dtype=np.int64,
                        shape=(bin_test_n,))

    Xf_train = open_memmap(os.path.join(out_fam, "X_train.npy"), mode="w+", dtype=np.float32,
                        shape=(fam_train_n, len(family_feats)))
    yf_train = open_memmap(os.path.join(out_fam, "y_train.npy"), mode="w+", dtype=np.int64,
                        shape=(fam_train_n,))
    Xf_test  = open_memmap(os.path.join(out_fam, "X_test.npy"), mode="w+", dtype=np.float32,
                        shape=(fam_test_n,  len(family_feats)))
    yf_test  = open_memmap(os.path.join(out_fam, "y_test.npy"), mode="w+", dtype=np.int64,
                        shape=(fam_test_n,))

    # remaining counters for streaming stratified
    bin_rem_total = bin_counts.astype(np.int64).copy()
    bin_rem_test  = bin_test_target.astype(np.int64).copy()

    fam_rem_total = fam_counts_arr.astype(np.int64).copy()
    fam_rem_test  = fam_test_target.astype(np.int64).copy()

    # write offsets
    btr = bte = 0
    ftr = fte = 0

    print("[PASS3] stream CSV -> write memmaps")
    for chunk in pd.read_csv(args.csv, usecols=needed_cols, chunksize=args.chunksize):
        # binary
        yb = chunk[args.label_col].apply(binarize_label).to_numpy(dtype=np.int64)
        Xb = clean_features_chunk(chunk, binary_feats, fillna=args.fillna)
        is_test_b = streaming_stratified_split_decider(yb, bin_rem_total, bin_rem_test, rng)

        # write binary rows
        if is_test_b.any():
            idx = np.where(is_test_b)[0]
            n = len(idx)
            Xb_test[bte:bte+n] = Xb[idx]
            yb_test[bte:bte+n] = yb[idx]
            bte += n
        if (~is_test_b).any():
            idx = np.where(~is_test_b)[0]
            n = len(idx)
            Xb_train[btr:btr+n] = Xb[idx]
            yb_train[btr:btr+n] = yb[idx]
            btr += n

        # family (attack-only)
        attack_mask = (yb == 1)
        if attack_mask.any():
            ca = chunk.loc[attack_mask]
            fam = ca[args.attack_col].astype(str).map(lambda x: family_map.get(x, "Other"))
            fam = fam.map(lambda x: "Other" if normalize_str(x) == "benign" else x)

            yfam = le.transform(fam.to_numpy(dtype=str))
            Xf = clean_features_chunk(ca, family_feats, fillna=args.fillna)

            is_test_f = streaming_stratified_split_decider(yfam, fam_rem_total, fam_rem_test, rng)

            if is_test_f.any():
                idx = np.where(is_test_f)[0]
                n = len(idx)
                Xf_test[fte:fte+n] = Xf[idx]
                yf_test[fte:fte+n] = yfam[idx]
                fte += n
            if (~is_test_f).any():
                idx = np.where(~is_test_f)[0]
                n = len(idx)
                Xf_train[ftr:ftr+n] = Xf[idx]
                yf_train[ftr:ftr+n] = yfam[idx]
                ftr += n

        # progress
        if (btr + bte) % (args.chunksize * 5) < args.chunksize:
            print(f"  progress: binary_written={btr+bte}/{bin_counts.sum()}  family_written={ftr+fte}/{fam_counts_arr.sum()}")

    # flush
    Xb_train.flush(); yb_train.flush(); Xb_test.flush(); yb_test.flush()
    Xf_train.flush(); yf_train.flush(); Xf_test.flush(); yf_test.flush()

    # save label encoder for family
    joblib.dump(le, os.path.join(out_fam, "label_encoder.pkl"))

    # metadata
    with open(os.path.join(out_bin, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "task": "binary",
            "schema": args.schema_binary,
            "n_features": len(binary_feats),
            "total": int(bin_counts.sum()),
            "train": int(bin_train_n),
            "test": int(bin_test_n),
            "test_size": args.test_size,
            "seed": args.seed,
            "fillna": args.fillna
        }, f, indent=2)

    with open(os.path.join(out_fam, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "task": "family_attack_only",
            "schema": args.schema_family,
            "n_features": len(family_feats),
            "classes": list(le.classes_),
            "total": int(fam_counts_arr.sum()),
            "train": int(fam_train_n),
            "test": int(fam_test_n),
            "test_size": args.test_size,
            "seed": args.seed,
            "fillna": args.fillna
        }, f, indent=2, ensure_ascii=False)

    print("[DONE] Chunked build complete.")
    print(f"Binary saved -> {out_bin}")
    print(f"Family saved -> {out_fam}")


if __name__ == "__main__":
    main()
