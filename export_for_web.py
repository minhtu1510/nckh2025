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

    # ── 1. Copy DeDe weights ────────────────────────────────────────────────
    print('\n[1] DeDe RAW...')
    dede_out = OUT / 'dede'
    dede_out.mkdir(exist_ok=True)
    shutil.copy(SRC_DEDE / 'best_model.weights.h5',  dede_out / 'dede.weights.h5')
    shutil.copy(SRC_DEDE / 'training_config.json',    dede_out / 'config.json')
    print(f'  ✓ {dede_out}')

    # ── 2. Load DeDe + tính ngưỡng ─────────────────────────────────────────
    print('\n[2] Calibrate thresholds...')
    with open(dede_out / 'config.json') as f:
        dede_cfg = json.load(f)
    dede = build_dede_model(
        input_dim=dede_cfg['input_dim'], latent_dim=dede_cfg.get('latent_dim', 64),
        encoder_hidden_dims=[256, 128], decoder_hidden_dims=[128, 256],
        mask_ratio=dede_cfg.get('mask_ratio', 0.5), dropout=0.2,
        learning_rate=dede_cfg.get('learning_rate', 0.001)
    )
    _ = dede(tf.zeros((1, dede_cfg['input_dim'])), training=False)
    dede.load_weights(str(dede_out / 'dede.weights.h5'))

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
        shutil.copy(f, gopt_out / f.name)
    print(f'  ✓ {gopt_out} ({len(list(SRC_GANOPT.glob("*")))} files)')

    # ── 4. Copy Standard Stacking ───────────────────────────────────────────
    print('\n[4] Standard Stacking (exp5b latent cache)...')
    std_out = OUT / 'standard'
    std_out.mkdir(exist_ok=True)
    for f in SRC_STD.glob('*'):
        shutil.copy(f, std_out / f.name)
    print(f'  ✓ {std_out} ({len(list(SRC_STD.glob("*")))} files)')

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
    print(f'  ✓ {pre_out}')

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
    inf_code = '''"""
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
    dede/dede.weights.h5
    encoder/benign_encoder.h5, malicious_encoder.h5
    ganopt/  (meta_model.pkl + mlp + knn)
    standard/(meta_model.pkl + mlp + svm + rf + knn)
    preprocessing/scaler.pkl, selector.pkl

  inference_exp9.py  ← import thẳng vào Flask/FastAPI

  from inference_exp9 import Exp9IDS
  ids = Exp9IDS("models/deploy_exp9")
  ids.predict_single(features)
""")


if __name__ == '__main__':
    main()
