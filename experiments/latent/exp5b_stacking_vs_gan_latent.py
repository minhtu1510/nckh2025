"""
Exp5b Latent: Stacking Alone vs GAN Attack — LATENT Features (Không có DeDe)

Chuỗi so sánh để kết luận:
  exp5b Standard LAT → exp5b GAN-Opt LAT → exp7 LAT (GAN-Opt + DeDe)

Train: 3.1_latent/exp1_baseline_latent (latent clean)
Test GAN: RAW GAN → dual-encoder → latent  (cùng pipeline với exp7 latent)

Results: results/latent/exp5b_stacking_vs_gan/
"""

import sys, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from models.ensemble.stacking import (
    create_stacking_ensemble,
    create_stacking_ensemble_gan_optimized,
    create_max_voting_ensemble_gan_optimized
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


class DualEncoder:
    def __init__(self, models_dir):
        d = Path(models_dir)
        self.benign_enc    = tf.keras.models.load_model(str(d / 'benign_encoder.h5'))
        self.malicious_enc = tf.keras.models.load_model(str(d / 'malicious_encoder.h5'))

    def encode(self, X_raw, batch_size=2048):
        z_b, z_m = [], []
        for i in range(0, len(X_raw), batch_size):
            b = X_raw[i:i+batch_size].astype(np.float32)
            z_b.append(self.benign_enc.predict(b, verbose=0))
            z_m.append(self.malicious_enc.predict(b, verbose=0))
        return np.hstack([np.concatenate(z_b), np.concatenate(z_m)])


def load_or_train(ens_fn, input_dim, X_tr, y_tr, cache_dir, label):
    cache = Path(cache_dir) / label
    if (cache / 'meta_model.pkl').exists():
        ens = ens_fn(input_dim=input_dim)
        ens.meta_model = joblib.load(cache / 'meta_model.pkl')
        for name in ens.base_models:
            p_pkl   = cache / f'{name}_model.pkl'
            p_keras = cache / f'{name}_model.keras'
            if p_pkl.exists():    ens.base_models[name] = joblib.load(p_pkl)
            elif p_keras.exists():
                from tensorflow import keras
                ens.base_models[name] = keras.models.load_model(p_keras)
        ens.is_fitted = True
        print(f'  ✓ Loaded cache: {label}')
        return ens
    print(f'  Training [{label}] ({len(X_tr):,} × {input_dim}) LATENT...')
    ens = ens_fn(input_dim=input_dim)
    ens.fit(X_tr, y_tr, verbose=False)
    ens.save(cache)
    print(f'  ✓ Saved: {cache}')
    return ens


def evaluate(model, X_lat, y, label):
    pred = model.predict(X_lat)
    return {
        'model':     label,
        'accuracy':  round(accuracy_score(y, pred), 6),
        'precision': round(precision_score(y, pred, zero_division=0), 6),
        'recall':    round(recall_score(y, pred, zero_division=0), 6),
        'f1_score':  round(f1_score(y, pred, zero_division=0), 6),
        'note':      'no_dede_latent',
    }


def main():
    lat_dir  = BASE_DIR / 'datasets/splits/3.1_latent'
    raw_dir  = BASE_DIR / 'datasets/splits/3.0_raw_from_latent'
    enc_dir  = lat_dir / 'models'
    out_dir  = BASE_DIR / 'results/latent/exp5b_stacking_vs_gan'
    out_dir.mkdir(parents=True, exist_ok=True)

    print('\n' + '='*80)
    print('EXP5b LATENT: Stacking Alone vs GAN Attack (Không có DeDe)'.center(80))
    print('Train: Latent | Test: RAW GAN → dual-encode → latent'.center(80))
    print('='*80)

    # Load dual encoder
    print('\n  Loading dual-encoder...')
    dual_enc = DualEncoder(enc_dir)

    # Load latent train (clean)
    X_tr_lat = np.load(lat_dir / 'exp1_baseline_latent/X_train.npy')
    y_tr     = np.load(lat_dir / 'exp1_baseline_latent/y_train.npy')
    print(f'  Train (latent clean): {len(X_tr_lat):,} × {X_tr_lat.shape[1]}')

    # Encode RAW GAN test → latent
    print('\n  Encoding RAW GAN test → latent...')
    X_raw_gan = np.load(raw_dir / 'exp3_gan_attack/X_test.npy')
    y_gan     = np.load(raw_dir / 'exp3_gan_attack/y_test.npy')
    X_lat_gan = dual_enc.encode(X_raw_gan)
    print(f'  GAN test (latent): {len(X_lat_gan):,} × {X_lat_gan.shape[1]}')

    # Encode RAW clean test → latent (reference)
    X_raw_clean = np.load(raw_dir / 'exp1_baseline/X_test.npy')
    y_clean     = np.load(raw_dir / 'exp1_baseline/y_test.npy')
    X_lat_clean = dual_enc.encode(X_raw_clean)
    print(f'  Clean test (latent): {len(X_lat_clean):,} × {X_lat_clean.shape[1]}')

    input_dim = X_tr_lat.shape[1]
    results = []

    # ── Standard Stacking ─────────────────────────────────────────────────────
    print('\n[1] Standard Stacking (LogisticRegression + Soft-Threshold 0.35 + MLPs)...')
    std_ens = load_or_train(create_stacking_ensemble, input_dim,
                            X_tr_lat, y_tr, out_dir, 'lat_standard_clean')
    r_std_clean = evaluate(std_ens, X_lat_clean, y_clean, 'Lat_Standard_Clean')
    r_std_gan   = evaluate(std_ens, X_lat_gan,   y_gan,   'Lat_Standard_GAN')
    results += [r_std_clean, r_std_gan]
    print(f'    Clean  → F1={r_std_clean["f1_score"]:.4f}')
    print(f'    GAN    → F1={r_std_gan["f1_score"]:.4f}')

    # ── GAN-Opt Stacking ──────────────────────────────────────────────────────
    print('\n[2] GAN-Opt (MAX-PROB) Stacking (MLP_deep+MLP_wide+NB+SVM) on LATENT...')
    ganopt_ens = load_or_train(create_max_voting_ensemble_gan_optimized, input_dim,
                               X_tr_lat, y_tr, out_dir, 'lat_max_ganopt_clean')
    r_ganopt_clean = evaluate(ganopt_ens, X_lat_clean, y_clean, 'Lat_GAN-Opt_Clean')
    r_ganopt_gan   = evaluate(ganopt_ens, X_lat_gan,   y_gan,   'Lat_GAN-Opt_GAN')
    results += [r_ganopt_clean, r_ganopt_gan]
    print(f'    Clean  → F1={r_ganopt_clean["f1_score"]:.4f}')
    print(f'    GAN    → F1={r_ganopt_gan["f1_score"]:.4f}')

    # Save
    df = pd.DataFrame(results)
    csv_path = out_dir / f'exp5b_latent_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(csv_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '='*80)
    print('✅ EXP5b LATENT SUMMARY — Stacking vs GAN (Không DeDe)'.center(80))
    print('='*80)
    print(f'\n  {"Model":<28} {"Clean F1":>10} {"GAN F1":>10}')
    print('  ' + '-'*50)
    print(f'  {"Lat Standard (LR threshold 0.35)":<32} '
          f'{r_std_clean["f1_score"]:>10.4f} {r_std_gan["f1_score"]:>10.4f}')
    print(f'  {"Lat GAN-Opt (MAX-PROB RULE)":<32} '
          f'{r_ganopt_clean["f1_score"]:>10.4f} {r_ganopt_gan["f1_score"]:>10.4f}')
    print(f'  {"DeDe+GAN-Opt (exp7 LAT)":<32} {"0.9670":>10} {"0.9251":>10}  ← reference')

    print('\n  📌 Kết luận chuỗi so sánh (GAN test):')
    d1 = r_ganopt_gan["f1_score"] - r_std_gan["f1_score"]
    d2 = 0.9251 - r_ganopt_gan["f1_score"]
    print(f'    [1] GAN-Opt vs Standard: Δ={d1:+.4f}  '
          f'→ GAN-Opt {"tốt hơn ✅" if d1 > 0 else "tệ hơn ❌"} với GAN')
    print(f'    [2] DeDe+GAN-Opt vs GAN-Opt alone: Δ={d2:+.4f}  '
          f'→ Thêm DeDe {"giúp ✅" if d2 > 0 else "cản ❌"} với GAN')

    print(f'\n📁 {csv_path}\n')


if __name__ == '__main__':
    main()
