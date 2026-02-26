"""
Exp9 Two-Path Routing — Inference class cho Web
================================================
Usage:
    from inference_exp9 import Exp9IDS
    ids = Exp9IDS("models/deploy_exp9")
    ids.predict_single(raw_features)
    # → {"label": "benign", "stage": "standard", "prediction": 0, "error": 0.001}
"""

import json, numpy as np, sys, joblib
import tensorflow as tf
from pathlib import Path


class Exp9IDS:

    def __init__(self, deploy_dir: str = "models/deploy_exp9"):
        d = Path(deploy_dir)
        with open(d / "config.json") as f:
            cfg = json.load(f)

        self.low_thr       = cfg["low_thr"]
        self.high_thr      = cfg["high_thr"]
        self.feature_names = cfg.get("feature_names", None)
        self._feat_norm    = {self._norm(fn): fn
                              for fn in (self.feature_names or [])}
        input_dim          = cfg["input_dim"]

        # Preprocessing
        pre = d / "preprocessing"
        self.scaler   = joblib.load(pre / "scaler.pkl")
        self.selector = joblib.load(pre / "selector.pkl")

        # DeDe RAW — ưu tiên load SavedModel (.keras), fallback weights
        dede_keras = d / "dede" / "dede_model.keras"
        dede_wts   = d / "dede" / "dede.weights.h5"
        if dede_keras.exists():
            self.dede = tf.keras.models.load_model(str(dede_keras))
        else:
            # fallback: cần build_dede_model từ research code
            from experiments.dede_adapted.dede_model import build_dede_model
            with open(d / "dede" / "config.json") as f:
                dcfg = json.load(f)
            self.dede = build_dede_model(
                input_dim=dcfg["input_dim"], latent_dim=dcfg.get("latent_dim", 64),
                encoder_hidden_dims=[256, 128], decoder_hidden_dims=[128, 256],
                mask_ratio=dcfg.get("mask_ratio", 0.5), dropout=0.2,
            )
            _ = self.dede(tf.zeros((1, dcfg["input_dim"])), training=False)
            self.dede.load_weights(str(dede_wts))

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
        """
        Load stacking tu cache directory.
        Thu tu base models doc tu config.pkl (chinh xac nhu luc train).
        CHI CAN: joblib, tensorflow — khong can research code.
        """
        cfg  = joblib.load(cache / "config.pkl")
        meta = joblib.load(cache / "meta_model.pkl")
        bases = {}
        for name in cfg["base_model_names"]:  # dung thu tu goc
            p1 = cache / f"{name}_model.pkl"
            p2 = cache / f"{name}_model.keras"
            if p1.exists():
                bases[name] = joblib.load(p1)
            elif p2.exists():
                bases[name] = tf.keras.models.load_model(str(p2))
            else:
                print(f"  [WARN] base model '{name}' not found in {cache}")
        return {"meta": meta, "bases": bases, "names": cfg["base_model_names"]}

    @staticmethod
    def _stack_predict(stack: dict, X_latent: np.ndarray) -> int:
        """Get meta-features from base models → predict via meta_model."""
        cols = []
        for name in stack["names"]:  # dung thu tu goc
            model = stack["bases"][name]
            if hasattr(model, "predict_proba"):
                preds = model.predict_proba(X_latent)[:, 1]
            elif hasattr(model, "decision_function"):
                raw = model.decision_function(X_latent)
                preds = 1.0 / (1.0 + np.exp(-raw))  # sigmoid
            else:
                preds = model.predict(X_latent).astype(float)
            cols.append(preds)
        mf = np.column_stack(cols)
        return int(stack["meta"].predict(mf)[0])

    def _encode(self, X: np.ndarray) -> np.ndarray:
        """Dual-encode: 50-dim → 64-dim latent (benign + malicious encoder)."""
        X_f32 = X.astype(np.float32)
        zb = self.benc.predict(X_f32, verbose=0)
        zm = self.menc.predict(X_f32, verbose=0)
        return np.hstack([zb, zm])

    def _dede_error(self, X: np.ndarray) -> np.ndarray:
        """MSE reconstruction error per sample."""
        if hasattr(self.dede, "get_reconstruction_error"):
            return self.dede.get_reconstruction_error(X)
        X_f32 = X.astype(np.float32)
        X_rec = self.dede(X_f32, training=False).numpy()
        return np.mean((X_f32 - X_rec) ** 2, axis=1)

    @staticmethod
    def _norm(name: str) -> str:
        """'Bwd_IAT_Max' / ' Bwd IAT Max' → 'bwd iat max'"""
        return name.strip().lower().replace("_", " ")

    def _align_record(self, record: dict) -> list:
        """Map dict bất kỳ → list 76 giá trị theo đúng thứ tự."""
        if not self.feature_names:
            raise ValueError("feature_names not in config. Re-run export_for_web.py.")
        norm_record = {self._norm(k): v for k, v in record.items()}
        result, missing = [], []
        for fn in self.feature_names:
            key = self._norm(fn)
            if key in norm_record:
                result.append(float(norm_record[key]))
            else:
                result.append(0.0)
                missing.append(fn)
        if missing:
            print(f"  [WARN] {len(missing)} features missing → fill 0: {missing[:3]}")
        return result

    def predict(self, X_raw: np.ndarray) -> list:
        """Batch predict.
        X_raw: (n, 76) raw features  → tự động apply scaler + selector → (n, 50)
               (n, 50) đã preprocessed → dùng thẳng, bỏ qua scaler/selector
               Số features khác → vẫn thử align, padding 0 nếu thiếu.
        """
        n_in   = X_raw.shape[1]
        n_raw  = self.scaler.n_features_in_          # 76
        n_proc = self.selector.transform(
                     np.zeros((1, n_raw), dtype=np.float32)).shape[1]  # 50

        if n_in == n_raw:
            # Input raw (76-dim) → apply scaler + selector → 50-dim
            X = self.selector.transform(
                    self.scaler.transform(X_raw)).astype(np.float32)
        elif n_in == n_proc:
            # Input đã preprocessed (50-dim) → dùng thẳng
            X = X_raw.astype(np.float32)
        else:
            raise ValueError(
                f"X có {n_in} features — không hợp lệ.\n"
                f"  Chấp nhận: {n_raw} (raw, chưa qua scaler/selector) "
                f"hoặc {n_proc} (đã preprocessed).\n"
                f"  Nếu features không có thứ tự cố định, dùng predict_dict() "
                f"để truyền dict với tên cột."
            )

        errs = self._dede_error(X)
        out  = []
        for i in range(len(X)):
            e, xi = float(errs[i]), X[[i]]
            if e >= self.high_thr:
                out.append({"prediction": 1, "label": "malicious",
                             "stage": "dede_blocked", "error": e})
            elif e >= self.low_thr:
                p = self._stack_predict(self.gan, self._encode(xi))
                out.append({"prediction": p,
                             "label": "malicious" if p else "benign",
                             "stage": "ganopt_stacking", "error": e})
            else:
                p = self._stack_predict(self.std, self._encode(xi))
                out.append({"prediction": p,
                             "label": "malicious" if p else "benign",
                             "stage": "standard_stacking", "error": e})
        return out

    def predict_single(self, raw_features) -> dict:
        """1 sample dạng list/array — 76 raw features hoặc 50 preprocessed features."""
        return self.predict(np.array(raw_features, dtype=np.float64).reshape(1, -1))[0]

    def predict_dict(self, records: list) -> list:
        """Batch từ list of dict — tên cột bất kỳ kiểu, trường thừa bỏ qua."""
        rows = [self._align_record(r) for r in records]
        return self.predict(np.array(rows, dtype=np.float64))

    def predict_single_dict(self, record: dict) -> dict:
        """1 sample dạng dict — tên cột bất kỳ kiểu."""
        return self.predict_dict([record])[0]

    def get_feature_names(self) -> list:
        """76 tên features theo đúng thứ tự cần truyền vào."""
        return list(self.feature_names) if self.feature_names else []


if __name__ == "__main__":
    ids = Exp9IDS("models/deploy_exp9")

    # Test 1: array 76 features
    sample = np.random.rand(76).tolist()
    print("predict_single  :", ids.predict_single(sample))

    # Test 2: dict với tên kiểu CICFlowMeter
    if ids.feature_names:
        record = {" " + fn.title(): np.random.rand() for fn in ids.feature_names}
        record["extra_timestamp"] = "2026-02-26"  # trường thừa → bị bỏ
        print("predict_single_dict:", ids.predict_single_dict(record))
