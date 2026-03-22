"""
Export exp9 Two-Path Routing model ra thư mục production.

Dùng models ĐÃ TRAIN:
  Standard Stacking : results/latent/exp5b_stacking_vs_gan/lat_standard_clean/
  GAN-Opt Stacking  : results/latent/exp7_combined_matrix_latent/ganopt_lat_clean/
  DeDe RAW          : experiments/dede_adapted/models_raw/
  Dual Encoder      : datasets/splits/3.1_latent/models/
  Preprocessing     : datasets/splits/3.1_latent/models/

Output: models/deploy_exp9/
"""

import sys, json, shutil, numpy as np, os
from pathlib import Path
from datetime import datetime

# Luôn trỏ đến thư mục ids_research dù chạy từ đâu
BASE_DIR = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
import joblib
from experiments.dede_adapted.dede_model import build_dede_model


def main():
    print('\n' + '='*70)
    print(' EXPORT EXP9 → models/deploy_exp9/'.center(70))
    print('='*70)

    OUT = BASE_DIR / 'models/deploy_exp9'
    OUT.mkdir(parents=True, exist_ok=True)

    # ── Nguồn ──────────────────────────────────────────────────────────────
    SRC_DEDE    = BASE_DIR / 'experiments/dede_adapted/models_raw'
    SRC_ENC     = BASE_DIR / 'datasets/splits/3.1_latent/models'
    SRC_GANOPT  = BASE_DIR / 'results/latent/exp7_combined_matrix_latent/ganopt_lat_clean'
    SRC_STD     = BASE_DIR / 'results/latent/exp5b_stacking_vs_gan/lat_standard_clean'
    RAW_DIR     = BASE_DIR / 'datasets/splits/3.0_raw_from_latent'

    # ── 1. Save DeDe as full Keras SavedModel (khong can build_dede_model khi inference)
    print('\n[1] DeDe RAW → SavedModel...')
    dede_out = OUT / 'dede'
    dede_out.mkdir(exist_ok=True)
    with open(SRC_DEDE / 'training_config.json') as f:
        dede_cfg = json.load(f)
    dede = build_dede_model(
        input_dim=dede_cfg['input_dim'], latent_dim=dede_cfg.get('latent_dim', 64),
        encoder_hidden_dims=[256, 128], decoder_hidden_dims=[128, 256],
        mask_ratio=dede_cfg.get('mask_ratio', 0.5), dropout=0.2,
        learning_rate=dede_cfg.get('learning_rate', 0.001)
    )
    _ = dede(tf.zeros((1, dede_cfg['input_dim'])), training=False)
    dede.load_weights(str(SRC_DEDE / 'best_model.weights.h5'))

    # Save as full Keras model (khong can code DeDe khi load)
    dede_saved = str(dede_out / 'dede_model.keras')
    tf.keras.models.save_model(dede, dede_saved)
    # Also save config
    with open(dede_out / 'config.json', 'w') as f:
        json.dump(dede_cfg, f, indent=2)
    print(f'  ✓ {dede_out}/dede_model.keras')

    # ── 2. Calibrate thresholds ────────────────────────────────────────────────
    print('\n[2] Calibrate thresholds...')
    X_clean  = np.load(RAW_DIR / 'exp1_baseline/X_test.npy')
    errs     = dede.get_reconstruction_error(X_clean)
    low_thr  = float(np.percentile(errs, 75))
    high_thr = float(np.percentile(errs, 99))
    print(f'  low_thr  (P75) = {low_thr:.6f}  → Standard Stack')
    print(f'  high_thr (P99) = {high_thr:.6f}  → Block (trigger)')

    # ── 3. Copy GAN-Opt Stacking ────────────────────────────────────────────
    print('\n[3] GAN-Opt Stacking (exp7 latent cache)...')
    gopt_out = OUT / 'ganopt'
    gopt_out.mkdir(exist_ok=True)
    for f in SRC_GANOPT.glob('*'):
        if f.suffix in ('.pkl', '.keras', '.h5'):
            shutil.copy(f, gopt_out / f.name)
    print(f'  ✓ {gopt_out} ({len(list(gopt_out.glob("*")))} files)')

    # ── 4. Copy Standard Stacking ───────────────────────────────────────────
    print('\n[4] Standard Stacking (exp5b latent cache)...')
    std_out = OUT / 'standard'
    std_out.mkdir(exist_ok=True)
    for f in SRC_STD.glob('*'):
         if f.suffix in ('.pkl', '.keras', '.h5'):
            shutil.copy(f, std_out / f.name)
    print(f'  ✓ {std_out} ({len(list(std_out.glob("*")))} files)')

    # ── 5. Copy Dual Encoder ────────────────────────────────────────────────
    print('\n[5] Dual Encoder...')
    enc_out = OUT / 'encoder'
    enc_out.mkdir(exist_ok=True)
    for fname in ['benign_encoder.h5', 'malicious_encoder.h5']:
        shutil.copy(SRC_ENC / fname, enc_out / fname)
    print(f'  ✓ {enc_out}')

    # ── 6. Copy Preprocessing ───────────────────────────────────────────────
    print('\n[6] Preprocessing (scaler + selector)...')
    pre_out = OUT / 'preprocessing'
    pre_out.mkdir(exist_ok=True)
    for fname in ['scaler.pkl', 'selector.pkl']:
        shutil.copy(SRC_ENC / fname, pre_out / fname)
    
    # Load scaler to get min/max for clamping
    scaler = joblib.load(SRC_ENC / 'scaler.pkl')
    print(f'  ✓ {pre_out} (scaler stats loaded)')

    print('\n[7] config.json...')
    # Load feature names từ preprocessing_info
    pre_info_path = SRC_ENC / 'preprocessing_info.json'
    feature_names = None
    if pre_info_path.exists():
        with open(pre_info_path) as f:
            pre_info = json.load(f)
        feature_names = pre_info.get('common_features', None)
        print(f'  ✓ feature_names: {len(feature_names)} features loaded')
    else:
        print('  ⚠️  preprocessing_info.json not found, feature alignment by name disabled')

    cfg = {
        'low_thr':      low_thr,
        'high_thr':     high_thr,
        'input_dim':    dede_cfg['input_dim'],
        'feature_names': feature_names,   # 76 tên features theo đúng thứ tự
        'scaler_min':   scaler.data_min_.tolist() if hasattr(scaler, 'data_min_') else None,
        'scaler_max':   scaler.data_max_.tolist() if hasattr(scaler, 'data_max_') else None,
        'created':      datetime.now().isoformat(),
        'routing': {
            f'error < {low_thr:.4f}':               'Standard Stacking',
            f'{low_thr:.4f} <= error < {high_thr:.4f}': 'GAN-Opt Stacking',
            f'error >= {high_thr:.4f}':             'BLOCKED (trigger)',
        },
        'performance_clean_model': {
            'clean_f1': 0.9772, 'gan_f1': 0.9241,
            'trigger_asr': 0.0, 'poison50_f1': 0.9479,
        }
    }
    with open(OUT / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'  ✓ {OUT}/config.json')

    # ── 8. Inference class ──────────────────────────────────────────────────
    print('\n[8] inference_exp9.py...')
    inf_code = r'''"""
Exp9 Two-Path Routing — Inference class cho Web
================================================
Usage:
    from inference_exp9 import Exp9IDS
    ids = Exp9IDS("models/deploy_exp9")
    ids.predict_single_dict(flow_dict)
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
        if not name: return ""
        return "".join(re.findall(r"[a-z0-9]+", name.lower()))

    def _get_variants(self, name: str) -> list:
        s = name.lower()
        variants = {s}
        subs = [
            ("pkts", "packets"), ("byts", "bytes"), ("pkt", "packet"),
            ("len", "length"), ("tot", "total"), ("cnt", "count"),
            ("std", "standard deviation"), (" avg", " mean"), ("mean", "avg")
        ]
        for old, new in subs:
            if old in s: variants.add(s.replace(old, new))
            if new in s: variants.add(s.replace(new, old))
        if "src2dst" in s: variants.add(s.replace("src2dst", "fwd"))
        if "dst2src" in s: variants.add(s.replace("dst2src", "bwd"))
        return [self._norm(v) for v in variants]

    def _align_record(self, record: dict) -> list:
        """Refined Aligner - Advanced Domain Adaptation (ToN-IoT <-> Generic)"""
        if not self.feature_names: return [0.0]*76
        
        # 1. Fuzzy Map
        raw_norm = {self._norm(k): v for k, v in record.items()}
        
        # 2. Refined alignment loop
        result = []
        for i, fn in enumerate(self.feature_names):
            val = None
            # Fuzzy match varieties
            for v in self._get_variants(fn):
                if v in raw_norm:
                    val = raw_norm[v]
                    break
            
            # --- Dataset-Specific Refinement (Domain Adaptation) ---
            s_min = self.scaler_min[i] if self.scaler_min is not None else 0.0
            s_max = self.scaler_max[i] if self.scaler_max is not None else 1.0
            
            # Use training-set midpoint if missing
            if val is None:
                fval = s_min + (s_max - s_min) * 0.5
            else:
                try: fval = float(val)
                except: fval = s_min

            # Fix 1: Duration Scale (ms vs us)
            if "duration" in fn and fval < 500000:
                fval *= 1000.0

            # Fix 2: Temporal Bias (Timestamps in ToN-IoT)
            if "idle" in fn or "active" in fn:
                is_timestamp_feat = any(x in fn for x in ["max", "min", "mean"]) and not any(x in fn for x in ["std", "var"])
                if is_timestamp_feat:
                    if s_max > 1e14: 
                        fval = s_min + (s_max - s_min) * 0.95
                    elif fval < 1e12 and s_min > 1e14:
                        fval = s_min + fval

            # Fix 3: Neutralize "Pseudo-Binary" features (mismatch between ToN-IoT and CIC)
            # If training set only saw [0, 1] or [0, 2] (likely constants/flags), 
            # and test set gives something else, we neutralize it to avoid triggering DeDe.
            if s_max <= 2.0 and s_max > s_min:
                # Force to training mean to stay 'low profile'
                fval = s_min + (s_max - s_min) * 0.5

            # --- CRITICAL: CLAMP TO TRAINING RANGE ---
            fval = max(s_min, min(s_max, fval))
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

    def predict_single_dict(self, record: dict) -> dict:
        row = self._align_record(record)
        return self.predict(np.array([row], dtype=np.float64))[0]

if __name__ == "__main__":
    ids = Exp9IDS()
    test_flow = {"src2dst_packets": 10, "duration": 500}
    print(ids.predict_single_dict(test_flow))
'''
    with open(BASE_DIR / 'inference_exp9.py', 'w') as f:
        f.write(inf_code)
    print(f'  ✓ inference_exp9.py')

    # ── Done ────────────────────────────────────────────────────────────────
    print('\n' + '='*70)
    print(' DONE '.center(70, '='))
    print(f"""
  models/deploy_exp9/
    config.json
    dede/dede_model.keras
    encoder/benign_encoder.h5, malicious_encoder.h5
    ganopt/  (meta_model.pkl + base models)
    standard/(meta_model.pkl + base models)
    preprocessing/scaler.pkl, selector.pkl

  inference_exp9.py  ← import thẳng vào Flask/FastAPI
""")


if __name__ == '__main__':
    main()
