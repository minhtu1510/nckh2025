from flask import Flask, request, jsonify
from flask_cors import CORS
from backend.store import STORE
from backend.config import CONFIG
from ids.runtime.predictor_v2 import IDSRuntimeV2
import time

def _get_first(flow: dict, *keys, default=None):
    # 1. Thử khớp chính xác trước
    for k in keys:
        if k in flow and flow[k] not in (None, "", "NA", "N/A"):
            return flow[k]
    
    # 2. Thử khớp mờ (lowercase, không gạch dưới đầu)
    norm_flow = {str(k).strip().lower().lstrip("_"): v for k, v in flow.items()}
    for k in keys:
        kn = str(k).strip().lower().lstrip("_")
        if kn in norm_flow and norm_flow[kn] not in (None, "", "NA", "N/A"):
            return norm_flow[kn]
    return default

def _extract_meta(flow: dict) -> dict:
    proto = _get_first(flow, "PROTOCOL", "protocol", default=None)

    # IP variations
    src_ip = _get_first(flow, "SRC_IP", "src_ip", "IPV4_SRC_ADDR", "source_ip", default=None)
    dst_ip = _get_first(flow, "DST_IP", "dst_ip", "IPV4_DST_ADDR", "dest_ip", "destination_ip", default=None)

    # Port variations
    src_port = _get_first(flow, "SRC_PORT", "src_port", "L4_SRC_PORT", "source_port", "sport", default=None)
    dst_port = _get_first(flow, "DST_PORT", "dst_port", "L4_DST_PORT", "dest_port", "destination_port", "dport", default=None)

    pkts = _get_first(flow, "IN_PKTS", "src2dst_packets", "total_fwd_packets", "tot_fwd_pkts", default=0)
    bytes = _get_first(flow, "IN_BYTES", "src2dst_bytes", "total_length_of_fwd_packets", "totlen_fwd_pkts", default=0)
    dur_ms = _get_first(flow, "FLOW_DURATION_MILLISECONDS", "flow_duration", "duration_ms", default=None)

    def _to_int(x, d=None):
        try: return int(float(x))
        except: return d

    def _to_float(x, d=None):
        try: return float(x)
        except: return d

    return {
        "proto": _to_int(proto, None),
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": _to_int(src_port, None),
        "dst_port": _to_int(dst_port, None),
        "pkts": _to_float(pkts, 0.0),
        "bytes": _to_float(bytes, 0.0),
        "dur_ms": _to_float(dur_ms, None),
    }
app = Flask(__name__)
CORS(app)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/api/config")
def get_config():
    return CONFIG

@app.post("/api/config")
def set_config():
    data = request.get_json(force=True) or {}
    # allow updating thresholds + toggles
    for k in ("tau_low", "tau_high", "enable_rules"):
        if k in data:
            CONFIG[k] = data[k]
    return CONFIG

@app.post("/api/event")
def ingest_event():
    ev = request.get_json(force=True) or {}
    # accept event contract from collector/predictor
    STORE.add(ev)
    return {"ok": True}

@app.get("/api/attacks")
def get_attacks():
    limit = int(request.args.get("limit", 200))
    verdict = request.args.get("verdict", "").strip().lower()

    def ffloat(name):
        v = request.args.get(name, "")
        if not v:
            return None
        try:
            return float(v)
        except:
            return None

    since = ffloat("since")
    until = ffloat("until")
    src_ip = request.args.get("src_ip", "").strip() or None
    dst_ip = request.args.get("dst_ip", "").strip() or None
    q = request.args.get("q", "")

    return jsonify(STORE.get_events(
        limit=limit, verdict=verdict,
        since=since, until=until,
        src_ip=src_ip, dst_ip=dst_ip,
        q=q
    ))

@app.get("/api/stats")
def get_stats():
    return STORE.get_stats()

@app.get("/api/flows")
def get_flows():
    limit = int(request.args.get("limit", 500))
    verdict = request.args.get("verdict", "").strip().lower()

    def ffloat(name):
        v = request.args.get(name, None)
        if v is None or v == "": 
            return None
        try:
            return float(v)
        except:
            return None

    since = ffloat("since")
    until = ffloat("until")
    src_ip = request.args.get("src_ip", "").strip() or None
    dst_ip = request.args.get("dst_ip", "").strip() or None
    q = request.args.get("q", "")

    return jsonify(STORE.get_flows(
        limit=limit, verdict=verdict,
        since=since, until=until,
        src_ip=src_ip, dst_ip=dst_ip,
        q=q
    ))

# =========================
# NEW: ingest raw flow -> predict -> store
# =========================
IDS = IDSRuntimeV2(
    tau_low=CONFIG.get("tau_low", 0.2),
    tau_high=CONFIG.get("tau_high", 0.95),
)
@app.post("/api/ingest")
def api_ingest():
    payload = request.get_json(force=True, silent=True) or {}
    flow = payload.get("flow", payload)  # cho phép gửi {flow:{...}} hoặc gửi thẳng {...}

    out = IDS.predict_flow(flow)
    # optional: attach ground-truth nếu tool replay gửi kèm
    if "gt_label" in payload: out["gt_label"] = payload["gt_label"]
    if "gt_attack" in payload: out["gt_attack"] = payload["gt_attack"]

    out["ts"] = float(payload.get("ts", time.time()))
    out["meta"] = _extract_meta(flow)
    STORE.add(out)
    return jsonify({"ok": True})

@app.post("/api/ingest_bulk")
def api_ingest_bulk():
    payload = request.get_json(force=True, silent=True) or {}
    items = payload.get("items", [])
    ts0 = float(payload.get("ts0", time.time()))

    n = 0
    for idx, it in enumerate(items):
        flow = it.get("flow", it)
        out = IDS.predict_flow(flow)
        if "gt_label" in it: out["gt_label"] = it["gt_label"]
        if "gt_attack" in it: out["gt_attack"] = it["gt_attack"]

        out["ts"] = float(it.get("ts", ts0 + idx * 0.001))
        out["meta"] = _extract_meta(flow)
        STORE.add(out)
        n += 1

    return jsonify({"ok": True, "ingested": n})


if __name__ == "__main__":
    # http://127.0.0.1:5000
    app.run(host="0.0.0.0", port=5000, debug=False)
