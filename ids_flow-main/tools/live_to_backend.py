#!/usr/bin/env python3
"""
Live flow collector (NFStreamer) -> POST to backend ingest endpoint.

- Captures LIVE traffic from a network interface (e.g., eth0).
- Emits flows quickly by using small idle_timeout/active_timeout.
- Sends to backend in batches or every flush_interval seconds.

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
import json
import sys
import time
from typing import Any, Dict, List, Optional

import requests

try:
    from nfstream import NFStreamer
except Exception as e:
    print("ERROR: nfstream not installed. Try: pip install nfstream", file=sys.stderr)
    raise


def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


import numpy as np

def nf_to_flow_dict(nf: Any, include_raw: bool = False) -> Dict[str, Any]:
    """
    Convert an NFStreamer flow object into a JSON-serializable dict.
    Automatically includes all numeric features from statistical_analysis.
    """
    d: Dict[str, Any] = {}

    # Standard attributes to prioritize
    standard_keys = [
        "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
        "time_start", "time_end", "duration",
        "bidirectional_packets", "bidirectional_bytes",
        "src2dst_packets", "src2dst_bytes",
        "dst2src_packets", "dst2src_bytes",
        "application_name", "requested_server_name"
    ]
    
    for k in standard_keys:
        if hasattr(nf, k):
            val = getattr(nf, k)
            if isinstance(val, (int, float, str)):
                d[k] = val

    # Automatically grab all statistical features (usually floats/ints)
    # Since NFlow uses __slots__, we use dir() to find attributes
    for k in dir(nf):
        if k.startswith("_") or k in d or k in ["to_dict", "to_pandas"]:
            continue
        try:
            v = getattr(nf, k)
            if isinstance(v, (int, float)):
                d[k] = v
            elif isinstance(v, (np.integer, np.floating)):
                d[k] = v.item()
        except (AttributeError, TypeError):
            continue

    # Normalize time
    if "time_start" in d: d["ts_start"] = float(d["time_start"]) / 1000.0
    if "time_end" in d: d["ts_end"] = float(d["time_end"]) / 1000.0
    if "duration" in d: d["duration"] = float(d["duration"]) / 1000.0 # NFStream is ms

    if include_raw:
        raw = {}
        for k in dir(nf):
            if not k.startswith("__"):
                try: raw[k] = str(getattr(nf, k))
                except: pass
        d["_raw"] = raw

    return d


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
    ap.add_argument("--include-raw", action="store_true", help="Attach _raw flow dict (debug)")

    args = ap.parse_args()

    ingest_url = args.backend.rstrip("/") + args.endpoint
    print(f"[live_to_backend] Capturing iface={args.iface}")
    print(f"[live_to_backend] Posting to {ingest_url}")
    print(f"[live_to_backend] batch={args.batch} flush_interval={args.flush_interval}s idle={args.idle}s active={args.active}s")

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
