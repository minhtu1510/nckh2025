"""
Exp9 Two-Path Routing — Inference class cho Web
================================================
Usage:
    from inference_exp9 import Exp9IDS
    ids = Exp9IDS("models/deploy_exp9")
    ids.predict_single(raw_features)
    # → {"label": "benign", "stage": "standard", "error": 0.001}
"""

import json, numpy as np, sys, joblib
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from experiments.dede_adapted.dede_model import build_dede_model
from models.ensemble.stacking import (
    create_stacking_ensemble,
    create_stacking_ensemble_gan_optimized,
)


class Exp9IDS:

    def __init__(self, deploy_dir: str = "models/deploy_exp9"):
        d = Path(deploy_dir)
        with open(d / "config.json") as f:
            cfg = json.load(f)

        self.low_thr  = cfg["low_thr"]
        self.high_thr = cfg["high_thr"]
        input_dim     = cfg["input_dim"]

        # Preprocessing
        pre = d / "preprocessing"
        self.scaler   = joblib.load(pre / "scaler.pkl")
        self.selector = joblib.load(pre / "selector.pkl")

        # DeDe RAW
        with open(d / "dede" / "config.json") as f:
            dcfg = json.load(f)
        self.dede = build_dede_model(
            input_dim=dcfg["input_dim"], latent_dim=dcfg.get("latent_dim", 64),
            encoder_hidden_dims=[256, 128], decoder_hidden_dims=[128, 256],
            mask_ratio=dcfg.get("mask_ratio", 0.5), dropout=0.2,
        )
        _ = self.dede(tf.zeros((1, dcfg["input_dim"])), training=False)
        self.dede.load_weights(str(d / "dede" / "dede.weights.h5"))

        # Dual Encoder
        enc = d / "encoder"
        self.benc = tf.keras.models.load_model(str(enc / "benign_encoder.h5"))
        self.menc = tf.keras.models.load_model(str(enc / "malicious_encoder.h5"))

        # Standard Stacking
        self.std = self._load_stack(create_stacking_ensemble, d / "standard", input_dim)

        # GAN-Opt Stacking
        self.gan = self._load_stack(create_stacking_ensemble_gan_optimized, d / "ganopt", input_dim)

        print(f"[Exp9IDS] ready  low={self.low_thr:.4f}  high={self.high_thr:.4f}")

    def _load_stack(self, fn, cache, dim):
        ens = fn(input_dim=dim)
        ens.meta_model = joblib.load(cache / "meta_model.pkl")
        for name in list(ens.base_models.keys()):
            p1 = cache / f"{name}_model.pkl"
            p2 = cache / f"{name}_model.keras"
            if p1.exists():
                ens.base_models[name] = joblib.load(p1)
            elif p2.exists():
                from tensorflow import keras
                ens.base_models[name] = keras.models.load_model(p2)
        ens.is_fitted = True
        return ens

    def _encode(self, X):
        zb = self.benc.predict(X.astype(np.float32), verbose=0)
        zm = self.menc.predict(X.astype(np.float32), verbose=0)
        return np.hstack([zb, zm])

    def predict(self, X_raw: np.ndarray) -> list:
        X = self.selector.transform(self.scaler.transform(X_raw)).astype(np.float32)
        errs = self.dede.get_reconstruction_error(X)
        n, out = len(X), []
        for i in range(n):
            e = float(errs[i])
            xi = X[[i]]
            if e >= self.high_thr:
                out.append({"label": "malicious", "stage": "dede_blocked",
                             "prediction": 1, "error": e})
            elif e >= self.low_thr:
                p = int(self.gan.predict(self._encode(xi))[0])
                out.append({"label": "malicious" if p else "benign",
                             "stage": "ganopt", "prediction": p, "error": e})
            else:
                p = int(self.std.predict(self._encode(xi))[0])
                out.append({"label": "malicious" if p else "benign",
                             "stage": "standard", "prediction": p, "error": e})
        return out

    def predict_single(self, raw_features) -> dict:
        return self.predict(np.array(raw_features).reshape(1, -1))[0]


if __name__ == "__main__":
    ids = Exp9IDS()
    sample = np.random.rand(1, 77)
    print(ids.predict_single(sample.tolist()[0]))
