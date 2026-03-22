#!/usr/bin/env python3
"""
Live flow collector (NFStreamer) -> POST to backend ingest endpoint.

Why this file exists
--------------------
Your ML pipeline expects a FIXED feature schema (CIC-style names with spaces, e.g. "bwd iat max").
NFStreamer exposes attributes with different naming conventions (usually snake_case, sometimes with unit suffixes).

To avoid "feature drift / lệch feature":
- We DO NOT dump every numeric attribute anymore.
- We emit ONLY the expected feature names (whitelist) and fill missing ones with 0.0.
- We keep some metadata (IPs/ports/protocol/timestamps) for tracing, but they don't affect the model.

Example:
  sudo python3 live_to_backend.py \
    --iface eth0 \
    --backend http://127.0.0.1:5000 \
    --endpoint /api/ingest_bulk \
    --batch 200 \
    --flush-interval 1.0 \
    --idle 2 --active 10
"""

import argparse
import sys
import time
import re
from typing import Any, Dict, List

import requests

try:
    from nfstream import NFStreamer
except Exception as e:
    print("ERROR: nfstream not installed. Try: pip install nfstream", file=sys.stderr)
    raise

import numpy as np


# ---------------------------------------------------------------------------
# EXPECTED FEATURE SCHEMA (from your Exp9 config)
# NOTE: Even if input_dim=50, your upstream pipeline may expect a larger raw
# feature set (e.g., before SelectKBest). So we whitelist EXACTLY these names.
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


# Metadata (debug / tracing). These keys are not part of the ML feature schema.
META_KEYS: List[str] = [
    "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
    "time_start", "time_end", "duration",
    "bidirectional_packets", "bidirectional_bytes",
    "src2dst_packets", "src2dst_bytes",
    "dst2src_packets", "dst2src_bytes",
    "application_name", "requested_server_name",
]


def _to_py_scalar(v: Any):
    """Convert numpy scalars to Python scalars so JSON can serialize them."""
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return v


def _norm_key(k: str) -> str:
    """Normalize keys to match across 'spaces/underscores/slashes/case'."""
    return re.sub(r"[^a-z0-9]+", "", (k or "").lower())


def _synonym_norms(feature_name: str) -> List[str]:
    """
    Generate a few normalized variants to bridge common naming differences.
    This does NOT guarantee perfect mapping, but helps a lot in practice.
    """
    s = (feature_name or "").lower()

    variants = {s}

    # common swaps
    variants.add(s.replace("pkts", "packets"))
    variants.add(s.replace("byts", "bytes"))
    variants.add(s.replace("pkt", "packet"))
    variants.add(s.replace("len", "length"))
    variants.add(s.replace("tot ", "total "))
    variants.add(s.replace(" tot", " total"))

    # remove spaces/slashes etc via _norm_key at the end
    out = []
    for v in variants:
        out.append(_norm_key(v))
        out.append(_norm_key(v.replace("/", " per ")))
    return list(dict.fromkeys(out))


def _extract_source_dict(nf: Any) -> Dict[str, Any]:
    """
    Try to extract a flat dict of attributes from NFStreamer flow.
    Prefer nf.to_dict() if available; otherwise fallback to getattr/dir.
    """
    src: Dict[str, Any] = {}

    # Best effort: use to_dict (usually includes computed stats cleanly)
    try:
        if hasattr(nf, "to_dict"):
            d0 = nf.to_dict()
            if isinstance(d0, dict):
                for k, v in d0.items():
                    src[str(k)] = _to_py_scalar(v)
    except Exception:
        pass

    # Ensure META_KEYS are present if available
    for k in META_KEYS:
        if hasattr(nf, k) and k not in src:
            try:
                src[k] = _to_py_scalar(getattr(nf, k))
            except Exception:
                pass

    # Fallback: scan numeric attributes (but DO NOT export all of them!)
    # This is only to help mapping into FEATURE_NAMES.
    try:
        for k in dir(nf):
            if k.startswith("_") or k in src or k in ("to_dict", "to_pandas"):
                continue
            try:
                v = getattr(nf, k)
                if isinstance(v, (int, float, np.integer, np.floating)):
                    src[k] = _to_py_scalar(v)
            except (AttributeError, TypeError):
                continue
    except Exception:
        pass

    return src


def _maybe_convert_time_units(target_feature: str, source_key: str, v: float) -> float:
    """
    CIC-style time features are often in microseconds.
    NFStreamer sometimes provides *_ms or *_s values. We convert ONLY when the
    source key explicitly contains a unit suffix.

    - *_ms  -> microseconds (x1000)
    - *_s   -> microseconds (x1e6)
    - *_us  -> unchanged
    """
    tf = (target_feature or "").lower()
    if "/s" in tf:
        return v  # rate, not time
    is_time_like = any(tok in tf for tok in ("iat", "duration", "idle", "active"))
    if not is_time_like:
        return v

    sk = (source_key or "").lower()
    if sk.endswith("_ms"):
        return v * 1000.0
    if sk.endswith("_s"):
        return v * 1_000_000.0
    if sk.endswith("_us"):
        return v
    return v


def nf_to_flow_dict(nf: Any, include_raw: bool = False) -> Dict[str, Any]:
    """
    Convert NFStreamer flow object into JSON dict:
    - Metadata fields (IPs/ports/protocol/timestamps)
    - ML features: EXACT whitelist FEATURE_NAMES (fill missing with 0.0)
    """
    src = _extract_source_dict(nf)

    # Build a normalized lookup from src keys
    norm_to_key: Dict[str, str] = {}
    for k in src.keys():
        nk = _norm_key(k)
        # keep first occurrence
        norm_to_key.setdefault(nk, k)

    out: Dict[str, Any] = {}

    # 1) Metadata
    for k in META_KEYS:
        if k in src:
            v = src[k]
            if isinstance(v, (int, float, str)):
                out[k] = v

    # Convenience timestamps (seconds)
    # Keep raw time_start/time_end as-is, plus derived seconds.
    if "time_start" in out:
        try:
            out["ts_start"] = float(out["time_start"]) / 1000.0
        except Exception:
            pass
    if "time_end" in out:
        try:
            out["ts_end"] = float(out["time_end"]) / 1000.0
        except Exception:
            pass
    if "duration" in out:
        # Do NOT overwrite duration; keep both ms and seconds for debugging.
        try:
            out["duration_ms"] = float(out["duration"])
            out["duration_s"] = float(out["duration"]) / 1000.0
        except Exception:
            pass

    # 2) ML features (whitelist)
    missing: List[str] = []
    matched: Dict[str, str] = {}

    for feat in FEATURE_NAMES:
        val = None
        src_key = None

        # Exact match
        if feat in src:
            src_key = feat
            val = src[feat]
        else:
            # Normalized match (spaces/underscores/etc)
            nfeat = _norm_key(feat)
            if nfeat in norm_to_key:
                src_key = norm_to_key[nfeat]
                val = src.get(src_key)

            # Synonym variants
            if val is None:
                for nvar in _synonym_norms(feat):
                    if nvar in norm_to_key:
                        src_key = norm_to_key[nvar]
                        val = src.get(src_key)
                        break

        # Convert & cast
        if isinstance(val, (np.integer, np.floating)):
            val = val.item()

        if isinstance(val, (int, float)):
            v = float(val)
            if src_key is not None:
                v = _maybe_convert_time_units(feat, src_key, v)
                matched[feat] = src_key
            out[feat] = v
        else:
            out[feat] = 0.0
            missing.append(feat)

    if include_raw:
        # Keep raw dump (stringified) + mapping debug for quick diagnosis.
        raw = {}
        for k, v in src.items():
            try:
                raw[k] = str(v)
            except Exception:
                pass
        out["_raw"] = raw
        out["_matched_features"] = matched
        out["_missing_features"] = missing

    return out


def post_bulk(url: str, items: List[Dict[str, Any]], timeout_s: float) -> None:
    if not items:
        return
    r = requests.post(url, json={"items": items}, timeout=timeout_s)
    r.raise_for_status()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="eth0", help="Interface name to capture (e.g., eth0)")
    ap.add_argument("--backend", default="http://127.0.0.1:5000", help="Backend base URL")
    ap.add_argument("--endpoint", default="/api/ingest_bulk", help="Ingest bulk endpoint path")
    ap.add_argument("--batch", type=int, default=200, help="Max items per POST")
    ap.add_argument("--flush-interval", type=float, default=1.0, help="Flush every N seconds")
    ap.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout seconds")

    # Make flow emit faster:
    ap.add_argument("--idle", type=int, default=2, help="idle_timeout seconds (flow expires sooner)")
    ap.add_argument("--active", type=int, default=10, help="active_timeout seconds")

    # Performance toggles:
    ap.add_argument("--no-promisc", action="store_true", help="Disable promiscuous mode")
    ap.add_argument("--no-tunnels", action="store_true", help="Disable tunnel decoding")
    ap.add_argument("--no-stats", action="store_true", help="Disable statistical_analysis")
    ap.add_argument("--no-l7", action="store_true", help="Disable L7 dissections (n_dissections=0)")
    ap.add_argument("--include-raw", action="store_true", help="Attach _raw + mapping debug fields (heavy)")

    args = ap.parse_args()

    ingest_url = args.backend.rstrip("/") + args.endpoint
    print(f"[live_to_backend] Capturing iface={args.iface}")
    print(f"[live_to_backend] Posting to {ingest_url}")
    print(f"[live_to_backend] batch={args.batch} flush_interval={args.flush_interval}s idle={args.idle}s active={args.active}s")
    print(f"[live_to_backend] feature_schema={len(FEATURE_NAMES)} keys (whitelist)")

    streamer = NFStreamer(
        source=args.iface,
        decode_tunnels=not args.no_tunnels,
        promiscuous_mode=not args.no_promisc,
        idle_timeout=args.idle,
        active_timeout=args.active,
        statistical_analysis=not args.no_stats,
        n_dissections=0 if args.no_l7 else 20,  # 0 = off
    )

    buf: List[Dict[str, Any]] = []
    last_flush = time.time()

    try:
        for nf in streamer:
            buf.append(nf_to_flow_dict(nf, include_raw=args.include_raw))
            now = time.time()

            if len(buf) >= args.batch or (now - last_flush) >= args.flush_interval:
                try:
                    post_bulk(ingest_url, buf, timeout_s=args.timeout)
                    print(f"[live_to_backend] sent {len(buf)} flows")
                except Exception as e:
                    print(f"[live_to_backend] POST failed: {e}", file=sys.stderr)
                buf.clear()
                last_flush = now

    except KeyboardInterrupt:
        print("\n[live_to_backend] stopping...")

    # final flush
    if buf:
        try:
            post_bulk(ingest_url, buf, timeout_s=args.timeout)
            print(f"[live_to_backend] final sent {len(buf)} flows")
        except Exception as e:
            print(f"[live_to_backend] final POST failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()