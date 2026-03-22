#!/usr/bin/env python3
"""
CSV flow sender -> backend ingest endpoint (bulk)

This script is the "offline" counterpart of live_to_backend.py.

Problem it fixes
----------------
The original script sends *all* CSV columns as the flow payload:
  flow = flow_df.iloc[i].to_dict()

That causes feature mismatch/drift when:
- the CSV contains extra columns not used by the model,
- column naming differs (e.g., "Bwd IAT Max" vs "bwd iat max"),
- the backend builds vectors by a fixed schema.

Solution
--------
- Whitelist the exact expected feature names (CIC-style, with spaces)
- Map CSV column names to expected names using normalization & synonyms
- Fill missing features with 0.0 (or fail fast with --strict)
- Keep only optional metadata keys if present, for tracing

Usage
-----
python3 flow_csv_to_backend.py --csv flows.csv --backend http://127.0.0.1:5000
python3 flow_csv_to_backend.py --csv flows.csv --backend http://127.0.0.1:5000 --dry-run
"""

import argparse
import re
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# EXPECTED FEATURE SCHEMA (from your Exp9 config)
# IMPORTANT: Keep these names EXACTLY (lowercase with spaces), because your
# backend / ML pipeline expects them.
# ---------------------------------------------------------------------------
FEATURE_NAMES: List[str] = [
    "bwd iat max",
    "bwd iat mean",
    "tot bwd pkts",
    "bwd urg flags",
    "active mean",
    "fwd pkts/b avg",
    "fwd blk rate avg",
    "fwd byts/b avg",
    "bwd byts/b avg",
    "fwd iat min",
    "init fwd win byts",
    "flow iat mean",
    "idle max",
    "fwd pkt len max",
    "flow duration",
    "totlen fwd pkts",
    "subflow bwd byts",
    "flow byts/s",
    "bwd iat std",
    "fin flag cnt",
    "urg flag cnt",
    "bwd pkt len max",
    "active std",
    "fwd urg flags",
    "bwd pkts/b avg",
    "bwd header len",
    "totlen bwd pkts",
    "fwd iat tot",
    "cwe flag count",
    "pkt len min",
    "bwd blk rate avg",
    "pkt size avg",
    "bwd iat tot",
    "active max",
    "flow iat max",
    "pkt len mean",
    "subflow bwd pkts",
    "ack flag cnt",
    "fwd pkt len std",
    "fwd seg size avg",
    "idle mean",
    "pkt len std",
    "syn flag cnt",
    "fwd act data pkts",
    "fwd header len",
    "active min",
    "fwd pkts/s",
    "psh flag cnt",
    "idle std",
    "tot fwd pkts",
    "pkt len var",
    "idle min",
    "subflow fwd byts",
    "bwd pkts/s",
    "bwd iat min",
    "fwd pkt len mean",
    "flow pkts/s",
    "fwd iat std",
    "flow iat min",
    "bwd psh flags",
    "bwd pkt len min",
    "rst flag cnt",
    "down/up ratio",
    "init bwd win byts",
    "subflow fwd pkts",
    "fwd psh flags",
    "flow iat std",
    "fwd pkt len min",
    "bwd seg size avg",
    "fwd iat max",
    "pkt len max",
    "fwd iat mean",
    "fwd seg size min",
    "bwd pkt len std",
    "ece flag cnt",
    "bwd pkt len mean",
]


# Optional metadata keys (kept only if present in CSV). Not part of ML schema.
META_KEYS: List[str] = [
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "label",
    "timestamp",
    "time_start",
    "time_end",
]


def _to_float(x: Any) -> float:
    """Convert to float safely; NaN/inf -> 0.0."""
    try:
        if x is None:
            return 0.0
        if isinstance(x, (np.integer, np.floating)):
            x = x.item()
        v = float(x)
        if not np.isfinite(v):
            return 0.0
        return v
    except Exception:
        return 0.0


def _norm_key(k: str) -> str:
    """Normalize keys to match across spaces/underscores/slashes/case."""
    return re.sub(r"[^a-z0-9]+", "", (k or "").lower())


def _synonym_norms(feature_name: str) -> List[str]:
    """
    Generate normalized variants to bridge common naming differences.
    Example: 'pkts' <-> 'packets', 'byts' <-> 'bytes', 'pkt' <-> 'packet'
    """
    s = (feature_name or "").lower()
    variants = {s}

    variants.add(s.replace("pkts", "packets"))
    variants.add(s.replace("byts", "bytes"))
    variants.add(s.replace("pkt", "packet"))
    variants.add(s.replace("len", "length"))
    variants.add(s.replace(" tot", " total"))
    variants.add(s.replace("tot ", "total "))

    out = []
    for v in variants:
        out.append(_norm_key(v))
        out.append(_norm_key(v.replace("/", " per ")))
    return list(dict.fromkeys(out))


def build_column_mapping(csv_columns: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Build mapping:
      expected feature name -> actual CSV column name
    Returns (feat_to_col, feat_to_debug_source)
    """
    # normalized csv col -> original csv col (first wins)
    norm_to_col: Dict[str, str] = {}
    for c in csv_columns:
        norm_to_col.setdefault(_norm_key(c), c)

    feat_to_col: Dict[str, str] = {}
    feat_to_src: Dict[str, str] = {}

    for feat in FEATURE_NAMES:
        # 1) exact column
        if feat in norm_to_col.values():
            feat_to_col[feat] = feat
            feat_to_src[feat] = "exact"
            continue

        # 2) normalized direct
        nfeat = _norm_key(feat)
        if nfeat in norm_to_col:
            feat_to_col[feat] = norm_to_col[nfeat]
            feat_to_src[feat] = "normalized"
            continue

        # 3) synonym variants
        found = None
        for nvar in _synonym_norms(feat):
            if nvar in norm_to_col:
                found = norm_to_col[nvar]
                break
        if found is not None:
            feat_to_col[feat] = found
            feat_to_src[feat] = "synonym"
            continue

        # not found
        feat_to_src[feat] = "missing"

    return feat_to_col, feat_to_src


def row_to_flow_payload(row: pd.Series, feat_to_col: Dict[str, str], strict: bool) -> Dict[str, Any]:
    """
    Convert one CSV row to payload with:
      - whitelisted FEATURE_NAMES (fill missing with 0.0, or fail if strict)
      - optional META_KEYS if present
    """
    flow: Dict[str, Any] = {}
    
    # meta keys (optional)
    for k in META_KEYS:
        if k in row.index:
            v = row[k]
            if isinstance(v, (np.integer, np.floating)):
                v = v.item()
            flow[k] = v
        # --- META REMAP: lấy các cột meta kiểu CIC (Title Case) -> key chuẩn underscore ---
    META_REMAP = {
        "Src IP": "src_ip",
        "Dst IP": "dst_ip",
        "Src Port": "src_port",
        "Dst Port": "dst_port",
        "Protocol": "protocol",
        "Timestamp": "timestamp",
        # tùy dataset có/không:
        "Flow ID": "flow_id",
    }

    for src_col, dst_key in META_REMAP.items():
        if src_col in row.index and dst_key not in flow:
            v = row[src_col]
            if isinstance(v, (np.integer, np.floating)):
                v = v.item()
            flow[dst_key] = v
    # ML features
    missing = []
    for feat in FEATURE_NAMES:
        if feat in feat_to_col:
            col = feat_to_col[feat]
            flow[feat] = _to_float(row.get(col, 0.0))
        else:
            flow[feat] = 0.0
            missing.append(feat)

    if strict and missing:
        raise ValueError(f"Missing {len(missing)} required features, e.g. {missing[:10]}")

    return flow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV file path")
    ap.add_argument("--backend", default="http://127.0.0.1:5000", help="Backend base URL")
    ap.add_argument("--endpoint", default="/api/ingest_bulk", help="Bulk ingest endpoint path")
    ap.add_argument("--batch", type=int, default=100, help="Rows per POST")
    ap.add_argument("--nrows", type=int, default=1000, help="Max rows to send")
    ap.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout seconds")
    ap.add_argument("--strict", action="store_true", help="Fail if any required feature is missing")
    ap.add_argument("--dry-run", action="store_true", help="Only print mapping/missing, do not POST")
    args = ap.parse_args()

    url = args.backend.rstrip("/") + args.endpoint

    df = pd.read_csv(args.csv, nrows=args.nrows)
    print(f"[CSV] loaded: {args.csv} rows={len(df)} cols={len(df.columns)} (limit={args.nrows})")

    feat_to_col, feat_to_src = build_column_mapping(list(df.columns))

    # report mapping
    missing = [f for f, src in feat_to_src.items() if src == "missing"]
    mapped = [(f, feat_to_col.get(f, None), feat_to_src[f]) for f in FEATURE_NAMES]

    print(f"[SCHEMA] expected features: {len(FEATURE_NAMES)}")
    print(f"[MAP] mapped={len(FEATURE_NAMES) - len(missing)} missing={len(missing)}")
    if missing:
        print("[MAP] missing examples:", missing[:20])

    # show some mappings
    print("[MAP] sample mappings:")
    for f, c, src in mapped[:20]:
        print(f"  {f!r:25s} -> {c!r:25s}  ({src})")

    if args.dry_run:
        print("[DRY-RUN] Not posting.")
        return

    # send rows
    sent = 0
    items: List[Dict[str, Any]] = []
    max_rows = min(args.nrows, len(df))

    for i in range(max_rows):
        row = df.iloc[i]
        flow = row_to_flow_payload(row, feat_to_col, strict=args.strict)
        items.append(flow)

        if len(items) >= args.batch:
            r = requests.post(url, json={"items": items, "ts0": time.time()}, timeout=args.timeout)
            r.raise_for_status()
            sent += len(items)
            print(f"[POST] {sent}/{max_rows} ok")
            items = []

    if items:
        r = requests.post(url, json={"items": items, "ts0": time.time()}, timeout=args.timeout)
        r.raise_for_status()
        sent += len(items)
        print(f"[POST] {sent}/{max_rows} ok")


if __name__ == "__main__":
    main()