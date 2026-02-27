import argparse
import json
import time
from typing import Dict, Any, List

import requests
from nfstream import NFStreamer

# ---------------- helpers ----------------

def load_schema_features(schema_path: str) -> List[str]:
    with open(schema_path, "r") as f:
        obj = json.load(f)
    feats = obj.get("feature_names") or []
    if not isinstance(feats, list) or not feats:
        raise ValueError(f"schema missing feature_names: {schema_path}")
    return feats

def base_flow(feats: List[str]) -> Dict[str, Any]:
    # ensure all schema keys exist
    return {k: 0.0 for k in feats}

def set_if_exists(d: Dict[str, Any], key: str, val: Any):
    if key in d:
        d[key] = val

def build_flow_from_nfstream(nf, feats: List[str]) -> Dict[str, Any]:
    f = base_flow(feats)

    # timing
    first_ms = getattr(nf, "bidirectional_first_seen_ms", None)
    last_ms  = getattr(nf, "bidirectional_last_seen_ms", None)
    dur_ms   = getattr(nf, "bidirectional_duration_ms", 0) or 0

    # map "CIC-like" core features (nếu schema có)
    proto = getattr(nf, "protocol", 0) or 0
    s2d_pkts = getattr(nf, "src2dst_packets", 0) or 0
    d2s_pkts = getattr(nf, "dst2src_packets", 0) or 0
    s2d_bytes = getattr(nf, "src2dst_bytes", 0) or 0
    d2s_bytes = getattr(nf, "dst2src_bytes", 0) or 0

    set_if_exists(f, "PROTOCOL", int(proto))
    set_if_exists(f, "IN_PKTS", float(s2d_pkts))
    set_if_exists(f, "OUT_PKTS", float(d2s_pkts))
    set_if_exists(f, "IN_BYTES", float(s2d_bytes))
    set_if_exists(f, "OUT_BYTES", float(d2s_bytes))
    set_if_exists(f, "FLOW_DURATION_MILLISECONDS", float(dur_ms))

    # ports (nếu schema có)
    set_if_exists(f, "SRC_PORT", float(getattr(nf, "src_port", 0) or 0))
    set_if_exists(f, "DST_PORT", float(getattr(nf, "dst_port", 0) or 0))

    # throughput (bytes/s) -> nếu schema có
    dur_s = max(1e-6, float(dur_ms) / 1000.0)
    set_if_exists(f, "SRC_TO_DST_AVG_THROUGHPUT", float(s2d_bytes) / dur_s)
    set_if_exists(f, "DST_TO_SRC_AVG_THROUGHPUT", float(d2s_bytes) / dur_s)

    # ts (seconds) để backend/UI dùng (nếu schema có key ts thì thôi; thường schema không có)
    # backend ingest thường tự gắn ts, nhưng gửi kèm cũng OK
    if last_ms is not None:
        f["ts"] = float(last_ms) / 1000.0
    elif first_ms is not None:
        f["ts"] = float(first_ms) / 1000.0
    else:
        f["ts"] = time.time()

    return f

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--backend", default="http://127.0.0.1:5000")
    ap.add_argument("--endpoint", default="/api/ingest_bulk")
    ap.add_argument("--binary-schema", default="schemas/v2/binary.json")
    ap.add_argument("--max-flows", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()

    feats = load_schema_features(args.binary_schema)

    # build flows from pcap
    streamer = NFStreamer(source=args.pcap, decode_tunnels=True)
    flows: List[Dict[str, Any]] = []
    for nf in streamer:
        flows.append(build_flow_from_nfstream(nf, feats))
        if len(flows) >= args.max_flows:
            break

    url = args.backend.rstrip("/") + args.endpoint
    posted = 0

    # POST in batches
    for i in range(0, len(flows), args.batch):
        chunk = flows[i:i + args.batch]
        payload = {"items": chunk}
        r = requests.post(url, json=payload, timeout=args.timeout)
        r.raise_for_status()
        posted += len(chunk)

    print(f"[DONE] flows={len(flows)} posted={posted} -> {url}")

if __name__ == "__main__":
    main()
