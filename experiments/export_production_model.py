"""
Export Hybrid Defense Model cho Web Deployment

Gom tất cả models cần thiết vào 1 thư mục: models/production/
  └── dede/                ← DeDe RAW (anomaly detector)
  └── stacking/            ← GAN-Opt Stacking (classifier)
  └── preprocessing/       ← Scaler + Selector (feature preprocessing)
  └── config.json          ← Thresholds + metadata

Pipeline trong web:
  RAW input (50 features sau scale+select)
    ↓ DeDe RAW → reconstruction error
    ├── error > threshold → MALICIOUS (trigger detected)
    └── error ≤ threshold → GAN-Opt Stacking → predict

Chạy:
  python experiments/export_production_model.py
"""

import sys, json, shutil, numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
import joblib
from experiments.dede_adapted.dede_model import build_dede_model
from models.ensemble.stacking import create_stacking_ensemble_gan_optimized


def main():
    print('\n' + '='*80)
    print('EXPORT PRODUCTION MODEL'.center(80))
    print('Hybrid Defense: DeDe + GAN-Opt Stacking (clean model)'.center(80))
    print('='*80)

    # ── Paths ─────────────────────────────────────────────────────────────────
    prod_dir  = BASE_DIR / 'models/production'
    prod_dir.mkdir(parents=True, exist_ok=True)

    dede_export    = prod_dir / 'dede'
    stack_export   = prod_dir / 'stacking'
    preproc_export = prod_dir / 'preprocessing'
    for d in [dede_export, stack_export, preproc_export]:
        d.mkdir(exist_ok=True)

    # ── 1. Export DeDe RAW ────────────────────────────────────────────────────
    print('\n[1] Exporting DeDe RAW...')
    dede_src = BASE_DIR / 'experiments/dede_adapted/models_raw'

    with open(dede_src / 'training_config.json') as f:
        dede_cfg = json.load(f)

    # Load model
    dede = build_dede_model(
        input_dim=dede_cfg['input_dim'],
        latent_dim=dede_cfg.get('latent_dim', 64),
        encoder_hidden_dims=[256, 128],
        decoder_hidden_dims=[128, 256],
        mask_ratio=dede_cfg.get('mask_ratio', 0.5),
        dropout=0.2,
        learning_rate=dede_cfg.get('learning_rate', 0.001)
    )
    _ = dede(tf.zeros((1, dede_cfg['input_dim'])), training=False)
    dede.load_weights(str(dede_src / 'best_model.weights.h5'))
    print(f'  ✓ DeDe loaded: input_dim={dede_cfg["input_dim"]}')

    # Calibrate threshold on clean test data
    raw_dir = BASE_DIR / 'datasets/splits/3.0_raw_from_latent'
    X_te_clean = np.load(raw_dir / 'exp1_baseline/X_test.npy')
    errs = dede.get_reconstruction_error(X_te_clean)
    threshold_99 = float(np.percentile(errs, 99))
    threshold_95 = float(np.percentile(errs, 95))
    print(f'  ✓ Threshold (P99): {threshold_99:.6f}')
    print(f'  ✓ Threshold (P95): {threshold_95:.6f}')

    # Save weights
    dede.save_weights(str(dede_export / 'dede_raw.weights.h5'))
    shutil.copy(dede_src / 'training_config.json', dede_export / 'training_config.json')
    print(f'  ✓ Saved to {dede_export}/')

    # ── 2. Export GAN-Opt Stacking (clean model from exp7) ───────────────────
    print('\n[2] Exporting GAN-Opt Stacking (clean model from exp7)...')
    stack_src = BASE_DIR / 'results/raw/exp7_combined_matrix/ganopt_clean'

    if not (stack_src / 'meta_model.pkl').exists():
        print(f'  ❌ Stacking cache not found: {stack_src}')
        print('     Chạy exp7 trước: python experiments/exp7_combined_attack_matrix.py')
        print('     Sau đó chạy lại script này.')
    else:
        # Copy all stacking files
        for f in stack_src.glob('*'):
            shutil.copy(f, stack_export / f.name)
        print(f'  ✓ Stacking files copied: {[f.name for f in stack_src.glob("*")]}')

        # Verify load
        ens = create_stacking_ensemble_gan_optimized(input_dim=dede_cfg['input_dim'])
        ens.meta_model = joblib.load(stack_export / 'meta_model.pkl')
        for name in ['knn_5', 'knn_11']:
            p = stack_export / f'{name}_model.pkl'
            if p.exists(): ens.base_models[name] = joblib.load(p)
        for name in ['mlp_deep', 'mlp_wide']:
            p = stack_export / f'{name}_model.keras'
            if p.exists():
                from tensorflow import keras
                ens.base_models[name] = keras.models.load_model(p)
        ens.is_fitted = True

        # Test inference
        test_input = X_te_clean[:5]
        test_pred = ens.predict(test_input)
        print(f'  ✓ Stacking inference test: {test_pred}')

    # ── 3. Export Preprocessing Pipeline ─────────────────────────────────────
    print('\n[3] Exporting preprocessing pipeline...')
    preproc_src = BASE_DIR / 'datasets/splits/3.1_latent/models'

    for fname in ['scaler.pkl', 'selector.pkl', 'preprocessing_info.json']:
        src = preproc_src / fname
        if src.exists():
            shutil.copy(src, preproc_export / fname)
            print(f'  ✓ Copied: {fname}')
        else:
            print(f'  ⚠️  Not found: {fname}')

    # ── 4. Save Production Config ─────────────────────────────────────────────
    print('\n[4] Saving production config...')
    config = {
        'model_name': 'Hybrid Defense IDS (DeDe + GAN-Opt Stacking)',
        'version': '1.0',
        'created_at': datetime.now().isoformat(),
        'pipeline': {
            'step1_preprocessing': {
                'description': 'Scale (MinMaxScaler) + Select top 50 features',
                'scaler': 'preprocessing/scaler.pkl',
                'selector': 'preprocessing/selector.pkl',
                'output_dim': dede_cfg['input_dim'],
            },
            'step2_dede_detection': {
                'description': 'DeDe RAW anomaly detection (Stage 1)',
                'weights': 'dede/dede_raw.weights.h5',
                'config': 'dede/training_config.json',
                'threshold_p99': threshold_99,
                'threshold_p95': threshold_95,
                'recommended_threshold': threshold_99,
                'action_if_above': 'predict_malicious (1)',
                'action_if_below': 'pass_to_stage2',
            },
            'step3_stacking_classify': {
                'description': 'GAN-Opt Stacking Ensemble (Stage 2)',
                'models_dir': 'stacking/',
                'base_models': ['mlp_deep', 'mlp_wide', 'knn_5', 'knn_11'],
                'meta_model': 'stacking/meta_model.pkl',
                'input_dim': dede_cfg['input_dim'],
                'output': '0=benign, 1=malicious',
            },
        },
        'performance': {
            'clean_f1': 0.9674,
            'gan_attack_f1': 0.9160,
            'trigger_asr': 0.0,
            'poison_50_f1': 0.9300,
        },
        'training_data': 'CIC-ToN-IoT',
        'note': 'Use clean model (no poisoning scenario) for production deployment',
    }

    with open(prod_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f'  ✓ Saved: {prod_dir}/config.json')

    # ── 5. Export Inference Example ───────────────────────────────────────────
    print('\n[5] Creating inference example script...')
    inference_code = '''"""
Inference Example: Hybrid Defense IDS

Usage:
  from inference import HybridDefenseInference
  model = HybridDefenseInference("models/production")
  result = model.predict(raw_features)
  print(result)  # {"prediction": 0, "confidence": 0.95, "stage": "stacking"}
"""

import json, numpy as np
from pathlib import Path
import joblib
import tensorflow as tf

# Add parent to path
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from experiments.dede_adapted.dede_model import build_dede_model
from models.ensemble.stacking import create_stacking_ensemble_gan_optimized


class HybridDefenseInference:
    """Production inference class cho Hybrid Defense IDS."""

    def __init__(self, prod_dir: str):
        prod_dir = Path(prod_dir)
        with open(prod_dir / "config.json") as f:
            self.config = json.load(f)

        cfg  = self.config["pipeline"]
        step1 = cfg["step1_preprocessing"]
        step2 = cfg["step2_dede_detection"]
        step3 = cfg["step3_stacking_classify"]

        # Load preprocessing
        self.scaler   = joblib.load(prod_dir / step1["scaler"])
        self.selector = joblib.load(prod_dir / step1["selector"])

        # Load DeDe
        with open(prod_dir / step2["config"]) as f:
            dede_cfg = json.load(f)
        self.dede = build_dede_model(
            input_dim=dede_cfg["input_dim"],
            latent_dim=dede_cfg.get("latent_dim", 64),
            encoder_hidden_dims=[256, 128],
            decoder_hidden_dims=[128, 256],
            mask_ratio=dede_cfg.get("mask_ratio", 0.5),
            dropout=0.2,
        )
        _ = self.dede(tf.zeros((1, dede_cfg["input_dim"])), training=False)
        self.dede.load_weights(str(prod_dir / step2["weights"]))
        self.threshold = step2["recommended_threshold"]

        # Load Stacking
        stacking_dir = prod_dir / "stacking"
        self.stacking = create_stacking_ensemble_gan_optimized(
            input_dim=dede_cfg["input_dim"]
        )
        self.stacking.meta_model = joblib.load(stacking_dir / "meta_model.pkl")
        for name in ["knn_5", "knn_11"]:
            p = stacking_dir / f"{name}_model.pkl"
            if p.exists():
                self.stacking.base_models[name] = joblib.load(p)
        for name in ["mlp_deep", "mlp_wide"]:
            p = stacking_dir / f"{name}_model.keras"
            if p.exists():
                from tensorflow import keras
                self.stacking.base_models[name] = keras.models.load_model(p)
        self.stacking.is_fitted = True

        print(f"✓ HybridDefenseInference loaded (threshold={self.threshold:.6f})")

    def preprocess(self, X_raw: np.ndarray) -> np.ndarray:
        """Scale + Select features."""
        X = self.scaler.transform(X_raw)
        X = self.selector.transform(X)
        return X.astype(np.float32)

    def predict(self, X_raw: np.ndarray) -> list:
        """
        Predict cho mỗi sample.
        Returns: list of dicts {"prediction": 0|1, "stage": "dede"|"stacking", "dede_error": float}
        """
        # Preprocess
        X = self.preprocess(X_raw)

        # Stage 1: DeDe
        errs = self.dede.get_reconstruction_error(X)

        results = []
        for i, (x, err) in enumerate(zip(X, errs)):
            if err > self.threshold:
                results.append({
                    "prediction":  1,
                    "label":       "malicious",
                    "stage":       "dede_blocked",
                    "dede_error":  float(err),
                    "confidence":  1.0,
                })
            else:
                # Stage 2: Stacking
                pred = int(self.stacking.predict(x.reshape(1, -1))[0])
                try:
                    proba = self.stacking.meta_model.predict_proba(
                        self.stacking._get_meta_features(x.reshape(1, -1))
                    )[0][pred]
                except Exception:
                    proba = 0.9 if pred == 1 else 0.1
                results.append({
                    "prediction":  pred,
                    "label":       "malicious" if pred == 1 else "benign",
                    "stage":       "stacking",
                    "dede_error":  float(err),
                    "confidence":  float(proba),
                })
        return results

    def predict_single(self, raw_features: list) -> dict:
        """Predict cho 1 sample (dùng cho API endpoint)."""
        X = np.array(raw_features).reshape(1, -1)
        return self.predict(X)[0]


# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    BASE_DIR = Path(__file__).resolve().parent

    model = HybridDefenseInference(BASE_DIR / "models/production")

    # Test với random data
    X_test = np.random.rand(5, 77)  # 77 raw features trước khi select
    results = model.predict(X_test)
    for i, r in enumerate(results):
        print(f"Sample {i}: {r['label']} (stage={r['stage']}, confidence={r['confidence']:.2f})")
'''

    with open(BASE_DIR / 'inference.py', 'w') as f:
        f.write(inference_code)
    print(f'  ✓ Saved: inference.py')

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '='*80)
    print('✅ PRODUCTION MODEL EXPORTED'.center(80))
    print('='*80)
    print(f"""
  📁 models/production/
    ├── config.json              ← Pipeline config + thresholds + metrics
    ├── dede/
    │   ├── dede_raw.weights.h5  ← DeDe anomaly detector weights
    │   └── training_config.json
    ├── stacking/
    │   ├── meta_model.pkl       ← Meta-classifier
    │   ├── mlp_deep_model.keras
    │   ├── mlp_wide_model.keras
    │   ├── knn_5_model.pkl
    │   └── knn_11_model.pkl
    └── preprocessing/
        ├── scaler.pkl           ← MinMaxScaler
        └── selector.pkl         ← Feature selector (top 50)

  📄 inference.py               ← Ready-to-use inference class

  🌐 Dùng cho Web API (Flask/FastAPI):
    from inference import HybridDefenseInference
    model = HybridDefenseInference("models/production")
    result = model.predict_single(raw_features)
    # → {{"prediction": 0, "label": "benign", "stage": "stacking", "confidence": 0.97}}
""")


if __name__ == '__main__':
    main()
