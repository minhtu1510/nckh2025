"""
Exp9 Two-Path Routing — Inference class cho Web
================================================
Usage:
    from inference_exp9 import Exp9IDS
    ids = Exp9IDS("models/deploy_exp9")
    ids.predict_single_dict(flow_dict)   # dict với key tùy dạng (CIC, NFFlow, real-time)
    ids.predict_csv_row(row_dict)        # dict từ CSV CICIDS2017/2018 (header chuẩn CIC)
    # → {"label": "benign", "stage": "standard", "prediction": 0, "error": 0.001}
"""

import json, numpy as np, sys, joblib, re
import tensorflow as tf
from pathlib import Path


class Exp9IDS:

    def __init__(self, deploy_dir: str = "models/deploy_exp9"):
        d = Path(deploy_dir)
        if not d.exists():
            # Try search in common locations
            paths = [Path("ids_research") / deploy_dir, Path("../") / deploy_dir, Path(".") / deploy_dir]
            for p in paths:
                if p.exists(): d = p; break

        with open(d / "config.json") as f:
            cfg = json.load(f)

        self.low_thr       = cfg["low_thr"]
        self.high_thr      = cfg["high_thr"]
        self.feature_names = cfg.get("feature_names", None)
        self.scaler_min    = cfg.get("scaler_min", None)
        self.scaler_max    = cfg.get("scaler_max", None)
        input_dim          = cfg["input_dim"]

        # Preprocessing
        pre = d / "preprocessing"
        self.scaler   = joblib.load(pre / "scaler.pkl")
        self.selector = joblib.load(pre / "selector.pkl")

        # DeDe RAW
        dede_keras = d / "dede" / "dede_model.keras"
        if dede_keras.exists():
            self.dede = tf.keras.models.load_model(str(dede_keras))
        else:
            raise FileNotFoundError(f"Dede model not found at {dede_keras}")

        # Dual Encoder
        enc = d / "encoder"
        self.benc = tf.keras.models.load_model(str(enc / "benign_encoder.h5"))
        self.menc = tf.keras.models.load_model(str(enc / "malicious_encoder.h5"))

        # Standard Stacking
        self.std = self._load_stack(d / "standard")

        # GAN-Opt Stacking
        self.gan = self._load_stack(d / "ganopt")

        print(f"[Exp9IDS] ready  low={self.low_thr:.4f}  high={self.high_thr:.4f}")

    def _load_stack(self, cache: Path) -> dict:
        cfg  = joblib.load(cache / "config.pkl")
        meta = joblib.load(cache / "meta_model.pkl")
        bases = {}
        for name in cfg["base_model_names"]:
            p1 = cache / f"{name}_model.pkl"
            p2 = cache / f"{name}_model.keras"
            if p1.exists():
                bases[name] = joblib.load(p1)
            elif p2.exists():
                bases[name] = tf.keras.models.load_model(str(p2))
        return {"meta": meta, "bases": bases, "names": cfg["base_model_names"]}

    @staticmethod
    def _stack_predict(stack: dict, X_latent: np.ndarray) -> int:
        cols = []
        for name in stack["names"]:
            model = stack["bases"][name]
            if hasattr(model, "predict_proba"):
                preds = model.predict_proba(X_latent)[:, 1]
            elif hasattr(model, "decision_function"):
                raw = model.decision_function(X_latent)
                preds = 1.0 / (1.0 + np.exp(-raw))
            else:
                preds = model.predict(X_latent, verbose=0).flatten().astype(float)
            cols.append(preds)
        mf = np.column_stack(cols)
        return int(stack["meta"].predict(mf)[0])

    def _encode(self, X: np.ndarray) -> np.ndarray:
        X_f32 = X.astype(np.float32)
        zb = self.benc.predict(X_f32, verbose=0)
        zm = self.menc.predict(X_f32, verbose=0)
        return np.hstack([zb, zm])

    def _dede_error(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.dede, "get_reconstruction_error"):
            return self.dede.get_reconstruction_error(X)
        X_f32 = X.astype(np.float32)
        X_rec = self.dede(X_f32, training=False).numpy()
        return np.mean((X_f32 - X_rec) ** 2, axis=1)

    @staticmethod
    def _norm(name: str) -> str:
        """Normalize: bỏ ký tự đặc biệt, lowercase. 'Fwd Pkt Len Max' -> 'fwdpktlenmax'"""
        if not name: return ""
        return "".join(re.findall(r"[a-z0-9]+", name.lower()))

    def _get_variants(self, name: str) -> list:
        """Generate normalized forms covering CIC / NFFlow / NFStreamer naming conventions."""
        s = name.lower()
        variants = {s}

        subs = [
            # Abbreviations <-> Full words
            ("pkts",     "packets"),
            ("byts",     "bytes"),
            ("pkt",      "packet"),
            ("len",      "length"),
            ("tot",      "total"),
            ("cnt",      "count"),
            ("std",      "standard deviation"),
            ("var",      "variance"),
            ("avg",      "average"),
            ("min",      "minimum"),
            ("max",      "maximum"),
            ("fwd",      "forward"),
            ("bwd",      "backward"),
            ("seg",      "segment"),
            ("init",     "initial"),
            ("blk",      "bulk"),
            ("iat",      "inter arrival time"),
        ]
        for old, new in subs:
            # try both word-boundary-aware and naive replace
            if old in s:
                variants.add(s.replace(old, new))
            if new in s:
                variants.add(s.replace(new, old))

        # Direction aliases
        for v in list(variants):
            if "fwd"     in v: variants.add(v.replace("fwd", "forward"))
            if "bwd"     in v: variants.add(v.replace("bwd", "backward"))
            if "forward" in v: variants.add(v.replace("forward", "fwd"))
            if "backward"in v: variants.add(v.replace("backward", "bwd"))
        # src2dst / dst2src aliases
        if "src2dst" in s: variants.add(s.replace("src2dst", "fwd"))
        if "dst2src" in s: variants.add(s.replace("dst2src", "bwd"))
        if "fwd" in s:     variants.add(s.replace("fwd", "src2dst"))
        if "bwd" in s:     variants.add(s.replace("bwd", "dst2src"))

        # CIC-specific oddities
        # "totlen fwd pkts" <-> "total length of fwd packets"
        # "pkt size avg"    <-> "average packet size"
        # "pkt len var"     <-> "packet length variance"
        # "init fwd win byts" <-> "init_win_bytes_forward" / "Init_Win_bytes_forward"
        # "fwd act data pkts" <-> "act_data_pkt_fwd"
        # "fwd seg size min"  <-> "min_seg_size_forward"
        extra_cic = {
            "totlenfwdpkts":         ["totallengthoffwdpackets", "totallengthofforwardpackets"],
            "totlenbwdpkts":         ["totallengthofbwdpackets", "totallengthofbackwardpackets"],
            "totfwdpkts":            ["totalfwdpackets", "totalforwardpackets"],
            "totbwdpkts":            ["totalbackwardpackets", "totalbwdpackets"],
            "pktsizeavg":            ["averagepacketsize", "avgpacketsize"],
            "pktlenvar":             ["packetlengthvariance", "pktlenvariance"],
            "pktlenmin":             ["minpacketlength", "minimumpacketlength"],
            "pktlenmax":             ["maxpacketlength", "maximumpacketlength"],
            "pktlenmean":            ["packetlengthmean", "meanpacketlength"],
            "pktlenstd":             ["packetlengthstd", "packetlengthstandarddeviation"],
            "initfwdwinbyts":        ["initwinbytesforward", "initwinbytesforw", "initwinbytsforward"],
            "initbwdwinbyts":        ["initwinbytesbwd", "initwinbytesbwdward", "initwinbytesbackward"],
            "fwdactdatapkts":        ["actdatapktfwd", "actdatapacketfwd"],
            "fwdsegsizemn":          ["minsegsizemn", "minsegszfwd"],
            "fwdsegsizemin":         ["minsegsizemn", "minsegszfwd", "minsegforwardsize"],
            "fwdheaderlen":          ["fwdheaderlength"],
            "bwdheaderlen":          ["bwdheaderlength"],
            "fwdbytsavg":            ["fwdavgbytesbulk", "fwdavgbulkbytes"],
            "fwdpktsavg":            ["fwdavgpacketsbulk"],
            "fwdblkrateavg":         ["fwdavgbulkrate"],
            "bwdbytsavg":            ["bwdavgbytesbulk"],
            "bwdpktsavg":            ["bwdavgpacketsbulk"],
            "bwdblkrateavg":         ["bwdavgbulkrate"],
            "subflowfwdpkts":        ["subflowforwardpackets"],
            "subflowfwdbyts":        ["subflowforwardbytes"],
            "subflowbwdpkts":        ["subflowbackwardpackets"],
            "subflowbwdbyts":        ["subflowbackwardbytes"],
            "fwdpktlenmax":          ["fwdpacketlengthmax", "maxlengthoffwdpackets"],
            "fwdpktlenmin":          ["fwdpacketlengthmin", "minlengthoffwdpackets"],
            "fwdpktlenmean":         ["fwdpacketlengthmean", "meanlengthoffwdpackets", "avgfwdsegmentsize"],
            "fwdpktlenstd":          ["fwdpacketlengthstd", "fwdpacketlengthstandarddeviation"],
            "fwdsegsizeavg":         ["avgfwdsegmentsize", "fwdpacketlengthmean"],
            "bwdpktlenmax":          ["bwdpacketlengthmax"],
            "bwdpktlenmin":          ["bwdpacketlengthmin"],
            "bwdpktlenmean":         ["bwdpacketlengthmean", "avgbwdsegmentsize"],
            "bwdpktlenstd":          ["bwdpacketlengthstd", "bwdpacketlengthstandarddeviation"],
            "bwdsegsizeavg":         ["avgbwdsegmentsize", "bwdpacketlengthmean"],
            # Bulk rate aliases: "fwd pkts/b avg" = "Fwd Avg Packets/Bulk"
            "fwdpktsbavg":           ["fwdavgpacketsbulk", "fwdavgpacketsperbulk"],
            "fwdbytsbavg":           ["fwdavgbytesbulk", "fwdavgbytesperbulk"],
            "fwdblkrateaverage":     ["fwdavgbulkrate"],
            "bwdpktsbavg":           ["bwdavgpacketsbulk", "bwdavgpacketsperbulk"],
            "bwdbytsbavg":           ["bwdavgbytesbulk", "bwdavgbytesperbulk"],
            "bwdblkrateaverage":     ["bwdavgbulkrate"],
            # fwd/bwd pkts/b avg variants
            "fwdpktsperbbavg":       ["fwdavgpacketsbulk"],
            "fwdbytsperbavg":        ["fwdavgbytesbulk"],
            "bwdpktsperbbavg":       ["bwdavgpacketsbulk"],
            "bwdbytsperbavg":        ["bwdavgbytesbulk"],
        }
        s_norm = self._norm(s)
        if s_norm in extra_cic:
            variants.update(extra_cic[s_norm])
        # Also check reverse: if any CIC alias matches the query
        for canonical, aliases in extra_cic.items():
            if s_norm in aliases:
                variants.add(canonical)
                variants.update(aliases)

        # Slash / "per" aliases
        for v in list(variants):
            if "/" in v:
                variants.add(v.replace("/", "per"))
                variants.add(v.replace("/", " per "))
            if " per " in v:
                variants.add(v.replace(" per ", "/"))

        return list(dict.fromkeys(self._norm(v) for v in variants))

    def _align_record(self, record: dict, dataset_type: str = "auto") -> list:
        """Map arbitrary field dict to the 76-feature vector expected by exp9 scaler.

        dataset_type:
          "cic"  - CICIDS2017/2018 CSV (features đã ở đúng thang)
          "nf"   - NFFlow real-time (PROTOCOL, IN_PKTS, ... – thiếu nhiều CIC features)
          "auto" - tự detect dựa vào keys có trong record
        """
        if not self.feature_names:
            return [0.0] * 76

        # ── Detect dataset type ────────────────────────────────────────────
        if dataset_type == "auto":
            keys_lower = {k.lower() for k in record.keys()}
            # CIC dataset thường có key kiểu "Fwd Packet Length Max" hoặc "fwd pkt len max"
            cic_hints  = {"fwd pkt len max", "bwd pkt len max", "flow duration",
                          "tot fwd pkts", "tot bwd pkts", "fwd iat tot"}
            nf_hints   = {"in_pkts", "out_pkts", "in_bytes", "protocol",
                          "flow_duration_milliseconds", "src_to_dst_avg_throughput"}
            n_cic = sum(1 for h in cic_hints if self._norm(h) in {self._norm(k) for k in record})
            n_nf  = sum(1 for h in nf_hints  if self._norm(h) in {self._norm(k) for k in record})
            dataset_type = "cic" if n_cic >= n_nf else "nf"

        is_cic = (dataset_type == "cic")

        # ── Build normalized lookup ────────────────────────────────────────
        raw_norm = {self._norm(k): v for k, v in record.items()}

        # ── Align each feature ────────────────────────────────────────────
        result = []
        for i, fn in enumerate(self.feature_names):
            s_min = self.scaler_min[i] if self.scaler_min is not None else 0.0
            s_max = self.scaler_max[i] if self.scaler_max is not None else 1.0
            s_range = s_max - s_min

            # Try to find value via fuzzy match
            val = None
            for variant in self._get_variants(fn):
                if variant in raw_norm:
                    val = raw_norm[variant]
                    break

            if val is None:
                # Feature not found → use safe zero (or s_min if s_min > 0)
                fval = max(0.0, s_min) if s_range > 0 else 0.0
            else:
                try:
                    fval = float(val)
                except (ValueError, TypeError):
                    fval = max(0.0, s_min)

                if not np.isfinite(fval):
                    fval = max(0.0, s_min)

            # ── Dataset-specific corrections ──────────────────────────────
            if is_cic:
                # CICIDS: idle/active features đo bằng microseconds (us)
                # scaler training (ToN-IoT) idle max/min/mean = timestamp (1.55e15 us)
                # → nếu training range là timestamp, map CIC's duration tương đối sang đó
                if ("idle" in fn or "active" in fn) and s_max > 1e12:
                    # CIC idle features thường rất nhỏ (0 - vài triệu us)
                    # Clamp về 0 thay vì để bị kéo về timestamp range
                    fval = min(fval, s_max)
                    # Nếu fval quá nhỏ so với training min (là timestamp), giữ nguyên giá trị
                    # thực và để clamp xử lý
            else:
                # NFFlow: flow_duration đơn vị milliseconds, CIC training dùng microseconds
                if "duration" in fn and s_max > 1e6:
                    fval *= 1000.0  # ms → us

            # ── Clamp về training range (LUÔN làm, bất kể dataset) ────────
            if s_range > 0:
                fval = max(s_min, min(s_max, fval))
            else:
                # Constant feature trong training → dùng s_min
                fval = s_min

            result.append(fval if np.isfinite(fval) else 0.0)

        return result

    def predict(self, X_raw: np.ndarray) -> list:
        n_in, n_raw = X_raw.shape[1], self.scaler.n_features_in_
        if n_in == n_raw:
            X = self.selector.transform(self.scaler.transform(X_raw)).astype(np.float32)
        else:
            X = X_raw.astype(np.float32)

        errs = self._dede_error(X)
        out  = []
        for i in range(len(X)):
            e, xi = float(errs[i]), X[[i]]
            if e >= self.high_thr:
                out.append({"prediction": 1, "label": "malicious", "stage": "dede_blocked", "error": e})
            elif e >= self.low_thr:
                p = self._stack_predict(self.gan, self._encode(xi))
                out.append({"prediction": p, "label": "malicious" if p else "benign", "stage": "ganopt", "error": e})
            else:
                p = self._stack_predict(self.std, self._encode(xi))
                out.append({"prediction": p, "label": "malicious" if p else "benign", "stage": "standard", "error": e})
        return out

    def _preprocess(self, X_raw: np.ndarray) -> np.ndarray:
        """Scale + select features từ raw 76-dim → 50-dim."""
        n_in, n_raw = X_raw.shape[1], self.scaler.n_features_in_
        if n_in == n_raw:
            return self.selector.transform(self.scaler.transform(X_raw)).astype(np.float32)
        return X_raw.astype(np.float32)

    def predict_with_adaptive_dede(self, X_raw: np.ndarray,
                                   low_pct: float = 15.0,
                                   high_pct: float = 30.0) -> list:
        """Predict dùng DeDe với **adaptive threshold** — recalibrate theo distribution
        của chính batch uploaded.

        Lý do cần adaptive:
        - DeDe calibrated trên ToN-IoT benign (low_thr≈0.001, high_thr≈0.013)
        - Với CICIDS: benign error ~0.109, attack error ~0.078 (đảo chiều!)
        - Vì DeDe được train thuần trên ToN-IoT benign patterns:
            * CICIDS benign (khác distribution mạnh) → reconstruct kém → error CAO
            * CICIDS attack (một số pattern gần ToN-IoT) → error THẤP HƠN
        - Adaptive: rows có error THẤP nhất trong batch → likely attack (dede_blocked)
        
        Args:
            X_raw   : array (N, 76) raw features chưa scale
            low_pct : percentile dùng làm high_thr (rows dưới ngưỡng này → BLOCK)
            high_pct: percentile dùng làm low_thr  (rows dưới ngưỡng này → ganopt)
        
        Returns: list of dicts cùng format với predict()
        """
        X = self._preprocess(X_raw)
        errs = self._dede_error(X)

        if len(errs) >= 10:
            # Khi có đủ data: calibrate theo batch distribution
            # Rows có error THẤP → gần với ToN-IoT attack patterns → malicious
            # high_thr_adaptive: ngưỡng phân tách attack (error thấp) vs benign (error cao)
            high_thr_a = float(np.percentile(errs, high_pct))
            # low_thr_adaptive: dưới low_pct% → rất chắc là attack
            low_thr_a  = float(np.percentile(errs, low_pct))
            # Sắp xếp logic: rows với error < low_thr_a = BLOCK, < high_thr_a = ganopt
            # Inverted: error thấp = malicious (ngược ToN-IoT)
            inverted = True
        else:
            # Quá ít data → dùng threshold gốc
            high_thr_a = self.high_thr
            low_thr_a  = self.low_thr
            inverted = False

        out = []
        for i in range(len(X)):
            e, xi = float(errs[i]), X[[i]]
            if inverted:
                # Với CICIDS: error THẤP = malicious (gần ToN-IoT attack), error CAO = benign
                if e <= low_thr_a:
                    # Rất có khả năng attack (error rất thấp → model quen reconstruction)
                    out.append({"prediction": 1, "label": "malicious",
                                "stage": "dede_blocked_inv", "error": e,
                                "adaptive_thr": {"low": low_thr_a, "high": high_thr_a}})
                elif e <= high_thr_a:
                    # Error thấp vừa → ganopt stacking quyết định
                    p = self._stack_predict(self.gan, self._encode(xi))
                    out.append({"prediction": p, "label": "malicious" if p else "benign",
                                "stage": "ganopt_inv", "error": e,
                                "adaptive_thr": {"low": low_thr_a, "high": high_thr_a}})
                else:
                    # Error cao → benign path (standard stacking)
                    p = self._stack_predict(self.std, self._encode(xi))
                    out.append({"prediction": p, "label": "malicious" if p else "benign",
                                "stage": "standard_inv", "error": e,
                                "adaptive_thr": {"low": low_thr_a, "high": high_thr_a}})
            else:
                # Dùng threshold gốc (ToN-IoT mode)
                if e >= self.high_thr:
                    out.append({"prediction": 1, "label": "malicious", "stage": "dede_blocked", "error": e})
                elif e >= self.low_thr:
                    p = self._stack_predict(self.gan, self._encode(xi))
                    out.append({"prediction": p, "label": "malicious" if p else "benign", "stage": "ganopt", "error": e})
                else:
                    p = self._stack_predict(self.std, self._encode(xi))
                    out.append({"prediction": p, "label": "malicious" if p else "benign", "stage": "standard", "error": e})
        return out

    def predict_bypass_dede(self, X_raw: np.ndarray) -> list:
        """Predict bỏ qua DeDe threshold hoàn toàn, dùng Standard Stacking trực tiếp.
        Dùng khi batch quá nhỏ (<10 rows) hoặc user muốn bypass hoàn toàn.
        """
        X = self._preprocess(X_raw)
        errs = self._dede_error(X)
        out = []
        for i in range(len(X)):
            e, xi = float(errs[i]), X[[i]]
            p = self._stack_predict(self.std, self._encode(xi))
            out.append({
                "prediction": p,
                "label": "malicious" if p else "benign",
                "stage": "standard_csv",
                "error": e,
            })
        return out

    def predict_single_dict(self, record: dict) -> dict:
        """Predict từ 1 flow record (tự detect dataset type)."""
        row = self._align_record(record, dataset_type="auto")
        return self.predict(np.array([row], dtype=np.float64))[0]

    def predict_csv_row(self, record: dict) -> dict:
        """Predict từ 1 row CSV CICIDS2017/2018 (single-row bypass DeDe).
        Nếu có nhiều rows → dùng predict_csv_batch() để tận dụng adaptive DeDe.
        """
        row = self._align_record(record, dataset_type="cic")
        return self.predict_bypass_dede(np.array([row], dtype=np.float64))[0]

    def predict_csv_batch(self, records: list, use_adaptive_dede: bool = False,
                          low_pct: float = 15.0, high_pct: float = 30.0,
                          dede_mode: str = "auto") -> list:
        """Predict toàn bộ batch CSV cùng một lúc.

        dede_mode (ưu tiên hơn use_adaptive_dede nếu được chỉ định):
          "original" → DeDe threshold GỐC (non-inverted):
                        error >= high_thr → blocked (attack)
                        error [low_thr, high_thr) → ganopt
                        error < low_thr  → standard
                        ✅ Tốt nhất cho ToN-IoT (DeDe train trên ToN-IoT benign)
                        FP ~1% với ToN-IoT benign

          "bypass"   → Bỏ qua DeDe hoàn toàn, Standard Stacking trực tiếp.
                        ✅ An toàn nhất, 0 false alarm từ DeDe
                        Dùng khi không chắc dataset type

          "adaptive" → Adaptive Inverted (thiết kế cho CICIDS):
                        error THẤP = attack (inverted logic)
                        ⚠️ Không dùng cho ToN-IoT — gây 15% false positive!

          "auto"     → Dựa vào use_adaptive_dede (backward-compat):
                        True  → "adaptive"
                        False → "bypass"

        Args:
            records    : list of dicts (mỗi dict là 1 row CSV)
            dede_mode  : "original" | "bypass" | "adaptive" | "auto"
            low_pct / high_pct: percentile cho adaptive thresholds (chỉ dùng khi mode=adaptive)
        """
        if not records:
            return []

        # Resolve mode
        if dede_mode == "auto":
            dede_mode = "adaptive" if use_adaptive_dede else "bypass"

        # Align tất cả records sang 76-dim vector
        rows  = [self._align_record(r, dataset_type="cic") for r in records]
        X_raw = np.array(rows, dtype=np.float64)

        if dede_mode == "original":
            # Dùng threshold gốc thẳng (non-inverted) — tốt nhất cho ToN-IoT
            return self.predict(X_raw)
        elif dede_mode == "adaptive" and len(records) >= 10:
            # Inverted adaptive DeDe (dành cho CICIDS)
            return self.predict_with_adaptive_dede(X_raw, low_pct=low_pct, high_pct=high_pct)
        else:
            # bypass: Standard Stacking trực tiếp (không qua DeDe threshold)
            return self.predict_bypass_dede(X_raw)


    def predict_nf_flow(self, record: dict) -> dict:
        """Predict từ 1 NFFlow real-time record (dùng full DeDe pipeline gốc)."""
        row = self._align_record(record, dataset_type="nf")
        return self.predict(np.array([row], dtype=np.float64))[0]

if __name__ == "__main__":
    ids = Exp9IDS()
    # Test với CIC-style record
    test_cic = {
        "Fwd Packet Length Max": 100,
        "Flow Duration": 500000,
        "Total Fwd Packets": 5,
        "Total Backward Packets": 3,
    }
    print("CIC test:", ids.predict_csv_row(test_cic))
    # Test với NFFlow-style record
    test_nf = {"src2dst_packets": 10, "duration": 500}
    print("NF test:", ids.predict_single_dict(test_nf))
