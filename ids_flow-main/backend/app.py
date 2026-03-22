from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from backend.store import STORE
from backend.config import CONFIG
import sys, io, csv, json, gc
from pathlib import Path
# ids_flow-main/backend/app.py -> ids_flow-main -> nckh2025
ROOT = Path(__file__).resolve().parents[2]  # /home/sus/NCKH/nckh2025
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ids.runtime.predictor_v2 import IDSRuntimeV2
import time

import re
def _norm(s: str) -> str:
    # "Flow Duration" == "flow_duration" == "FLOW-DURATION"
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def _to_int(x, d=None):
    try: return int(float(x))
    except: return d

def _to_float(x, d=None):
    try: return float(x)
    except: return d

def _get_first(flow: dict, *keys, default=None):
    # 1) match chính xác trước
    for k in keys:
        if k in flow and flow[k] not in (None, "", "NA", "N/A"):
            return flow[k]

    # 2) match normalize (bỏ space/_/-/case)
    norm_flow = {_norm(k): v for k, v in flow.items()}
    for k in keys:
        nk = _norm(k)
        if nk in norm_flow and norm_flow[nk] not in (None, "", "NA", "N/A"):
            return norm_flow[nk]

    return default

def _extract_meta(flow: dict) -> dict:
    proto = _get_first(flow, "PROTOCOL", "protocol", default=None)

    # IP variations
    src_ip = _get_first(flow, "SRC_IP", "src_ip", "IPV4_SRC_ADDR", "source_ip", default=None)
    dst_ip = _get_first(flow, "DST_IP", "dst_ip", "IPV4_DST_ADDR", "dest_ip", "destination_ip", default=None)

    # Port variations
    src_port = _get_first(flow, "SRC_PORT", "src_port", "L4_SRC_PORT", "source_port", "sport", default=None)
    dst_port = _get_first(flow, "DST_PORT", "dst_port", "L4_DST_PORT", "dest_port", "destination_port", "dport", default=None)

    pkts = _get_first(flow, "IN_PKTS", "src2dst_packets", "total_fwd_packets", "tot_fwd_pkts", "bidirectional_packets", default=0)
    bytes = _get_first(flow, "IN_BYTES", "src2dst_bytes", "total_length_of_fwd_packets", "totlen_fwd_pkts", "bidirectional_bytes", default=0)
    dur_ms = _get_first(flow, "FLOW_DURATION_MILLISECONDS", "flow_duration", "duration_ms", "duration", default=None)

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
# KHÔNG đặt MAX_CONTENT_LENGTH toàn cục vì /api/upload_csv cần nhận file lớn (streaming).
# /api/upload_csv_chunk tự kiểm tra kích thước chunk bên trong handler.



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
    # dede_mode: chọn mode cho DeDe
    #   "original" = DeDe threshold gốc (tốt nhất cho ToN-IoT) ← MẶC ĐỊNH
    #   "bypass"   = bỏ qua DeDe, dùng Standard Stacking (an toàn nhất)
    #   "adaptive" = Adaptive Inverted (thiết kế cho CICIDS, có thể gây FP với ToN-IoT)
    dede_mode    = payload.get("dede_mode", "original")
    use_adaptive = payload.get("use_adaptive_dede", False)  # backward-compat

    if not items:
        return jsonify({"ok": True, "ingested": 0})

    # ── Auto-detect: CIC CSV vs NF real-time ─────────────────────────
    first_flow = items[0].get("flow", items[0])
    _cic_hints = {"bwdiatmax", "flowduration", "totfwdpkts", "fwdpktlenmax", "totbwdpkts"}
    _first_norm = {re.sub(r'[^a-z0-9]+', '', k.lower()) for k in first_flow.keys()}
    is_cic_batch = bool(_first_norm & _cic_hints)

    n = 0

    if is_cic_batch and IDS.model is not None and hasattr(IDS.model, "predict_csv_batch"):
        # ── CIC CSV batch path ───────────────────────────────────────────
        flow_list = [it.get("flow", it) for it in items]
        preds = IDS.model.predict_csv_batch(
            flow_list,
            use_adaptive_dede=use_adaptive,
            dede_mode=dede_mode,
        )

        for idx, (it, res) in enumerate(zip(items, preds)):
            flow = it.get("flow", it)
            is_attack = res.get("prediction", 0) == 1
            ml_verdict = "attack" if is_attack else "benign"
            out = {
                "p_attack": 1.0 if is_attack else 0.0,
                "stage": res.get("stage", "exp9_ml"),
                "verdict": ml_verdict,
                "family": res.get("label", "benign") if is_attack else None,
                "family_conf": 1.0 if is_attack else 0.0,
                "rule_info": None,
                "rule_result": {"hit": False},
                "ml_binary": {"p_attack": 1.0 if is_attack else 0.0, "verdict": ml_verdict},
                "ml_family": {"name": None, "conf": None},
                "final_source": "ml-only" if is_attack else "none",
                "ml_verdict": ml_verdict,
                "dede_error": res.get("error", 0.0),
            }
            if "gt_label" in it: out["gt_label"] = it["gt_label"]
            if "gt_attack" in it: out["gt_attack"] = it["gt_attack"]
            out["ts"] = float(it.get("ts", ts0 + idx * 0.001))
            out["meta"] = _extract_meta(flow)
            STORE.add(out)
            n += 1
    else:
        # ── NF real-time path: predict từng flow (rules + ML) ──
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




# =========================================================
# CSV Upload endpoint – nhận file CSV từ browser (CICIDS style)
# =========================================================

# 76 feature names của exp9 (dùng để map CS v column)
EXP9_FEATURES = [
    "bwd iat max", "bwd iat mean", "tot bwd pkts", "bwd urg flags",
    "active mean", "fwd pkts/b avg", "fwd blk rate avg", "fwd byts/b avg",
    "bwd byts/b avg", "fwd iat min", "init fwd win byts", "flow iat mean",
    "idle max", "fwd pkt len max", "flow duration", "totlen fwd pkts",
    "subflow bwd byts", "flow byts/s", "bwd iat std", "fin flag cnt",
    "urg flag cnt", "bwd pkt len max", "active std", "fwd urg flags",
    "bwd pkts/b avg", "bwd header len", "totlen bwd pkts", "fwd iat tot",
    "cwe flag count", "pkt len min", "bwd blk rate avg", "pkt size avg",
    "bwd iat tot", "active max", "flow iat max", "pkt len mean",
    "subflow bwd pkts", "ack flag cnt", "fwd pkt len std", "fwd seg size avg",
    "idle mean", "pkt len std", "syn flag cnt", "fwd act data pkts",
    "fwd header len", "active min", "fwd pkts/s", "psh flag cnt",
    "idle std", "tot fwd pkts", "pkt len var", "idle min",
    "subflow fwd byts", "bwd pkts/s", "bwd iat min", "fwd pkt len mean",
    "flow pkts/s", "fwd iat std", "flow iat min", "bwd psh flags",
    "bwd pkt len min", "rst flag cnt", "down/up ratio", "init bwd win byts",
    "subflow fwd pkts", "fwd psh flags", "flow iat std", "fwd pkt len min",
    "bwd seg size avg", "fwd iat max", "pkt len max", "fwd iat mean",
    "fwd seg size min", "bwd pkt len std", "ece flag cnt", "bwd pkt len mean",
]

META_REMAP = {
    "Src IP": "src_ip",    "SRC_IP": "src_ip",    "Source IP": "src_ip",
    "Dst IP": "dst_ip",    "DST_IP": "dst_ip",    "Destination IP": "dst_ip",
    "Src Port": "src_port", "SRC_PORT": "src_port", "Source Port": "src_port",
    "Dst Port": "dst_port", "DST_PORT": "dst_port", "Destination Port": "dst_port",
    "Protocol": "protocol", "PROTOCOL": "protocol",
    "Timestamp": "timestamp", "Flow ID": "flow_id",
    "Label": "label",
}

def _norm_key(k):
    return re.sub(r"[^a-z0-9]+", "", (k or "").lower())

def _build_col_map(csv_columns):
    """Map tên feature của exp9 sang tên column trong CSV.
    Dùng Exp9IDS._get_variants() nếu model đã load để tận dụng synonym mapping đầy đủ.
    """
    norm_to_col = {}
    for c in csv_columns:
        norm_to_col.setdefault(_norm_key(c), c)

    # Try dùng Exp9IDS variant matching (99% match rate với CICIDS2017)
    if IDS.model is not None and hasattr(IDS.model, "_get_variants"):
        get_variants = IDS.model._get_variants
    else:
        # Fallback: simple synonym matching
        def get_variants(feat):
            s = feat.lower()
            variants = {s}
            for old, new in [("pkts","packets"),("byts","bytes"),("pkt","packet"),("len","length"),("tot ","total "),(" tot"," total")]:
                if old in s: variants.add(s.replace(old, new))
                if new in s: variants.add(s.replace(new, old))
            for v in list(variants):
                variants.add(v.replace("/", " per "))
            return [_norm_key(v) for v in variants]

    feat_to_col = {}
    for feat in EXP9_FEATURES:
        # 1) Exact match
        nfeat = _norm_key(feat)
        if nfeat in norm_to_col:
            feat_to_col[feat] = norm_to_col[nfeat]
            continue
        # 2) Variant matching
        for nv in get_variants(feat):
            if nv in norm_to_col:
                feat_to_col[feat] = norm_to_col[nv]
                break
    return feat_to_col

def _csv_row_to_flow(row_dict, feat_to_col):
    """Chuyển 1 row dict của CSV thành payload (kèm meta)."""
    flow = {}
    # Meta
    for src_col, dst_key in META_REMAP.items():
        if src_col in row_dict:
            flow[dst_key] = row_dict[src_col]
    # ML features - giữ đúng tên feature của exp9
    for feat in EXP9_FEATURES:
        if feat in feat_to_col:
            col = feat_to_col[feat]
            v = row_dict.get(col, 0.0)
            try:
                fv = float(v)
                import math
                flow[feat] = 0.0 if (not math.isfinite(fv)) else fv
            except (TypeError, ValueError):
                flow[feat] = 0.0
        else:
            flow[feat] = 0.0
    return flow

@app.post("/api/upload_csv")
def api_upload_csv():
    """Nhận file CSV từ browser, chạy exp9 ML inference.
    Xử lý theo batch nhỏ để tránh tràn RAM với file lớn.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files supported"}), 400

    max_rows      = int(request.form.get("max_rows", 2000))
    use_adaptive  = request.form.get("use_adaptive_dede", "false").lower() != "false"
    # dede_mode: "original"=DeDe gốc (ToN-IoT) | "bypass"=an toàn | "adaptive"=CICIDS
    dede_mode     = request.form.get("dede_mode", "original")
    dede_low_pct  = float(request.form.get("dede_low_pct",  "15"))
    dede_high_pct = float(request.form.get("dede_high_pct", "30"))
    batch_size    = int(request.form.get("batch_size", 500))

    feat_to_col   = None
    n_mapped      = 0
    n_ingested    = 0
    total_rows    = 0
    adaptive_info = {}
    ts0           = time.time()

    try:
        # Stream đọc từ file object – không load toàn bộ vào RAM
        text_stream = io.TextIOWrapper(f.stream, encoding="utf-8", errors="replace")
        reader      = csv.DictReader(text_stream)

        batch = []
        for raw_row in reader:
            if total_rows >= max_rows:
                break
            total_rows += 1
            row = dict(raw_row)

            # Build column map từ header của dòng đầu tiên
            if feat_to_col is None:
                feat_to_col = _build_col_map(list(row.keys()))
                n_mapped    = len(feat_to_col)

            batch.append(row)

            # Khi đủ batch_size → predict + store → giải phóng RAM
            if len(batch) >= batch_size:
                _process_csv_batch(
                    batch, feat_to_col, ts0, n_ingested,
                    use_adaptive, dede_low_pct, dede_high_pct,
                    adaptive_info, dede_mode=dede_mode
                )
                n_ingested += len(batch)
                batch = []
                gc.collect()

        # Batch cuối (số dòng còn lại < batch_size)
        if batch:
            _process_csv_batch(
                batch, feat_to_col, ts0, n_ingested,
                use_adaptive, dede_low_pct, dede_high_pct,
                adaptive_info, dede_mode=dede_mode
            )
            n_ingested += len(batch)
            batch = []
            gc.collect()

    except Exception as e:
        return jsonify({"error": f"CSV parse error: {e}"}), 400

    if n_ingested == 0:
        return jsonify({"error": "CSV is empty or no rows matched"}), 400

    return jsonify({
        "ok": True,
        "ingested": n_ingested,
        "total_csv_rows": total_rows,
        "mapped_features": n_mapped,
        "missing_features": len(EXP9_FEATURES) - n_mapped,
        "dede_mode": dede_mode,
        "adaptive_thr": adaptive_info,
    })


def _process_csv_batch(batch_rows, feat_to_col, ts0, offset,
                       use_adaptive, dede_low_pct, dede_high_pct, adaptive_info_out,
                       dede_mode="original"):
    """Predict + store 1 batch nhỏ. Giải phóng list sau khi dùng xong."""
    flow_list = [_csv_row_to_flow(r, feat_to_col) for r in batch_rows]

    if hasattr(IDS, "predict_csv_batch"):
        preds = IDS.predict_csv_batch(
            flow_list,
            use_adaptive_dede=use_adaptive,
            low_pct=dede_low_pct,
            high_pct=dede_high_pct,
            dede_mode=dede_mode,
        )
    else:
        preds = [IDS.predict_csv_row(fl) for fl in flow_list]

    for idx, (flow, out) in enumerate(zip(flow_list, preds)):
        row_dict = batch_rows[idx]
        gt = flow.get("label") or row_dict.get("Label") or row_dict.get("label")
        if gt:
            out["gt_label"] = gt

        out["ts"] = ts0 + (offset + idx) * 0.001
        out["meta"] = {
            "src_ip":  flow.get("src_ip"),
            "dst_ip":  flow.get("dst_ip"),
            "src_port": _to_int(flow.get("src_port"), None),
            "dst_port": _to_int(flow.get("dst_port"), None),
            "proto":    _to_int(flow.get("protocol"), None),
            "pkts":     flow.get("tot fwd pkts", 0.0),
            "bytes":    flow.get("totlen fwd pkts", 0.0),
            "dur_ms":   flow.get("flow duration", 0.0) / 1000.0,
        }
        STORE.add(out)

    # Giữ adaptive_info từ batch đầu tiên
    if not adaptive_info_out and preds and "adaptive_thr" in preds[0]:
        adaptive_info_out.update(preds[0]["adaptive_thr"])

    # Giải phóng list ngay
    del flow_list, preds


# =========================================================
# CHUNKED CSV Upload – nhận từng chunk text/binary riêng lẻ
# Dùng cho file rất lớn (>100MB) – frontend chia chunk 4MB
# =========================================================

# ── In-memory chunk buffer (per upload_id) ─────────────────────────────────
# Lưu state giữa các chunk: header đã parse chưa, số dòng đã xử lý, ...
import threading
_CHUNK_SESSIONS: dict = {}   # upload_id → dict
_CHUNK_LOCK = threading.Lock()


@app.post("/api/upload_csv_chunk")
def api_upload_csv_chunk():
    """Endpoint nhận 1 chunk text CSV (plain text, không phải file upload).
    Frontend gửi nhiều request, mỗi request 1 chunk ~4 MB.
    Lợi thế: RAM backend chỉ giữ 1 chunk tại một thời điểm.

    Form fields:
        upload_id   : string – ID duy nhất cho lần upload này
        chunk_index : int    – 0-based index của chunk
        total_chunks: int    – tổng số chunks
        is_last     : "1" hoặc "0"
        max_rows    : int    – giới hạn tổng số dòng (0 = không giới hạn)
        batch_size  : int    – xử lý bao nhiêu dòng / lần predict (default 500)
        use_adaptive_dede: "true"/"false"
        dede_low_pct / dede_high_pct: float
    File:
        chunk   : blob / part của file CSV
    """
    if "chunk" not in request.files:
        return jsonify({"error": "No chunk data"}), 400

    upload_id   = request.form.get("upload_id", "default")
    chunk_index = int(request.form.get("chunk_index", 0))
    is_last     = request.form.get("is_last", "0") == "1"
    max_rows    = int(request.form.get("max_rows", 0))      # 0 = không giới hạn
    batch_size  = int(request.form.get("batch_size", 500))
    use_adaptive  = request.form.get("use_adaptive_dede", "false").lower() != "false"
    dede_low_pct  = float(request.form.get("dede_low_pct",  "15"))
    dede_high_pct = float(request.form.get("dede_high_pct", "30"))

    chunk_file = request.files["chunk"]
    # Kiểm tra kích thước chunk tối đa 20MB để tránh tràn RAM
    MAX_CHUNK = 20 * 1024 * 1024
    chunk_file.stream.seek(0, 2)
    chunk_bytes_len = chunk_file.stream.tell()
    chunk_file.stream.seek(0)
    if chunk_bytes_len > MAX_CHUNK:
        return jsonify({"error": f"Chunk quá lớn ({chunk_bytes_len/1024/1024:.1f}MB). Tối đa 20MB/chunk."}), 413

    try:
        chunk_text = chunk_file.read().decode("utf-8", errors="replace")
    except Exception as e:
        return jsonify({"error": f"Decode error: {e}"}), 400


    with _CHUNK_LOCK:
        if upload_id not in _CHUNK_SESSIONS:
            _CHUNK_SESSIONS[upload_id] = {
                "leftover": "",      # phần dòng dở dang từ chunk trước
                "header": None,      # list[str] – tên cột CSV
                "feat_to_col": None,
                "n_mapped": 0,
                "n_ingested": 0,
                "total_rows": 0,
                "ts0": time.time(),
                "adaptive_info": {},
                "done": False,
            }
        sess = _CHUNK_SESSIONS[upload_id]

    if sess["done"]:
        return jsonify({"error": "Session already finished"}), 400

    # Nối phần dư từ chunk trước + chunk mới
    raw = sess["leftover"] + chunk_text
    del chunk_text  # giải phóng ngay

    # Tách thành các dòng; line cuối có thể chưa có '\n' → giữ lại
    lines = raw.splitlines(keepends=True)
    del raw

    if lines and not lines[-1].endswith("\n") and not is_last:
        sess["leftover"] = lines.pop()   # dòng dở dang
    else:
        sess["leftover"] = ""

    # Nếu chưa có header → dòng đầu tiên là header
    if sess["header"] is None:
        if not lines:
            return jsonify({"chunk_done": True, "n_ingested_this_chunk": 0,
                            "total_ingested": 0})
        header_line = lines.pop(0).rstrip("\n\r")
        sess["header"] = next(csv.reader([header_line]))
        sess["feat_to_col"] = _build_col_map(sess["header"])
        sess["n_mapped"]    = len(sess["feat_to_col"])

    # Parse các dòng data thành dict, rồi xử lý theo mini-batch
    header     = sess["header"]
    feat_to_col = sess["feat_to_col"]
    ts0        = sess["ts0"]
    n_chunk    = 0

    batch: list = []
    for line in lines:
        line = line.rstrip("\n\r")
        if not line:
            continue
        # Giới hạn tổng dòng nếu có
        if max_rows > 0 and sess["total_rows"] >= max_rows:
            break
        try:
            vals = next(csv.reader([line]))
        except Exception:
            continue
        if len(vals) != len(header):
            continue   # bỏ qua dòng lỗi

        row = dict(zip(header, vals))
        batch.append(row)
        sess["total_rows"] += 1

        if len(batch) >= batch_size:
            _process_csv_batch(
                batch, feat_to_col, ts0, sess["n_ingested"],
                use_adaptive, dede_low_pct, dede_high_pct,
                sess["adaptive_info"]
            )
            sess["n_ingested"] += len(batch)
            n_chunk += len(batch)
            batch = []
            gc.collect()

    # Batch cuối của chunk này
    if batch:
        _process_csv_batch(
            batch, feat_to_col, ts0, sess["n_ingested"],
            use_adaptive, dede_low_pct, dede_high_pct,
            sess["adaptive_info"]
        )
        sess["n_ingested"] += len(batch)
        n_chunk += len(batch)
        del batch
        gc.collect()

    del lines

    resp = {
        "chunk_done": True,
        "chunk_index": chunk_index,
        "n_ingested_this_chunk": n_chunk,
        "total_ingested": sess["n_ingested"],
        "total_rows": sess["total_rows"],
    }

    if is_last:
        sess["done"] = True
        resp["finished"] = True
        resp["mapped_features"]   = sess["n_mapped"]
        resp["missing_features"]  = len(EXP9_FEATURES) - sess["n_mapped"]
        resp["adaptive_thr"]      = sess["adaptive_info"]
        resp["dede_mode"]          = "adaptive" if (use_adaptive and sess["total_rows"] >= 10) else "bypass"
        # Xóa session khỏi bộ nhớ
        with _CHUNK_LOCK:
            _CHUNK_SESSIONS.pop(upload_id, None)
        gc.collect()

    return jsonify(resp)


@app.post("/api/reset")
def api_reset():
    """Xóa toàn bộ dữ liệu trong STORE (dùng để reset dashboard trước khi upload mới)."""
    STORE.clear()
    return jsonify({"ok": True})


if __name__ == "__main__":
    # http://127.0.0.1:5000
    app.run(host="0.0.0.0", port=5000, debug=False)
