"""
Exp5b: Stacking Alone vs GAN Attack — RAW Features (Không có DeDe)

Mục tiêu: Cơ sở so sánh trước khi thêm DeDe (exp7)
  Standard Stacking (MLP+SVM+RF+KNN) vs GAN  → F1 = ?
  GAN-Opt Stacking (MLP+KNN variants)  vs GAN  → F1 = ?

Chuỗi so sánh để kết luận:
  exp5b Standard → exp5b GAN-Opt → exp7 (GAN-Opt + DeDe)
  → Tăng dần? DeDe giúp hay cản GAN?

Data:
  Train: 3.0_raw_from_latent/exp1_baseline (clean)
  Test:  3.0_raw_from_latent/exp3_gan_attack (GAN)

Results: results/raw/exp5b_stacking_vs_gan/
"""

import sys, json, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from models.ensemble.stacking import (
    create_stacking_ensemble,
    create_stacking_ensemble_gan_optimized
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


def load_or_train(ens_fn, input_dim, X_tr, y_tr, cache_dir, label):
    cache = Path(cache_dir) / label
    if (cache / 'meta_model.pkl').exists():
        ens = ens_fn(input_dim=input_dim)
        ens.meta_model = joblib.load(cache / 'meta_model.pkl')
        for name in ens.base_models:
            p_pkl  = cache / f'{name}_model.pkl'
            p_keras = cache / f'{name}_model.keras'
            if p_pkl.exists():   ens.base_models[name] = joblib.load(p_pkl)
            elif p_keras.exists():
                from tensorflow import keras
                ens.base_models[name] = keras.models.load_model(p_keras)
        ens.is_fitted = True
        print(f'  ✓ Loaded cache: {label}')
        return ens
    print(f'  Training [{label}] ({len(X_tr):,} × {input_dim})...')
    ens = ens_fn(input_dim=input_dim)
    ens.fit(X_tr, y_tr, verbose=False)
    ens.save(cache)
    print(f'  ✓ Saved: {cache}')
    return ens


def evaluate(model, X, y, label):
    pred = model.predict(X)
    return {
        'model':     label,
        'accuracy':  round(accuracy_score(y, pred), 6),
        'precision': round(precision_score(y, pred, zero_division=0), 6),
        'recall':    round(recall_score(y, pred, zero_division=0), 6),
        'f1_score':  round(f1_score(y, pred, zero_division=0), 6),
        'note':      'no_dede',
    }


def main():
    raw_dir  = BASE_DIR / 'datasets/splits/3.0_raw_from_latent'
    out_dir  = BASE_DIR / 'results/raw/exp5b_stacking_vs_gan'
    out_dir.mkdir(parents=True, exist_ok=True)

    print('\n' + '='*80)
    print('EXP5b RAW: Stacking Alone vs GAN Attack (Không có DeDe)'.center(80))
    print('Cơ sở so sánh → exp7 (Stacking + DeDe)'.center(80))
    print('='*80)

    # Load clean train data
    X_tr = np.load(raw_dir / 'exp1_baseline/X_train.npy')
    y_tr = np.load(raw_dir / 'exp1_baseline/y_train.npy')
    print(f'\n  Train (clean): {len(X_tr):,} × {X_tr.shape[1]}')

    # Load GAN test set
    X_te_gan = np.load(raw_dir / 'exp3_gan_attack/X_test.npy')
    y_te_gan = np.load(raw_dir / 'exp3_gan_attack/y_test.npy')
    print(f'  Test (GAN):    {len(X_te_gan):,}')

    # Also load clean test for reference
    X_te_clean = np.load(raw_dir / 'exp1_baseline/X_test.npy')
    y_te_clean = np.load(raw_dir / 'exp1_baseline/y_test.npy')
    print(f'  Test (clean):  {len(X_te_clean):,}')

    input_dim = X_tr.shape[1]
    results = []

    # ── Standard Stacking (MLP+SVM+RF+KNN) ──────────────────────────────────
    print('\n[1] Standard Stacking (MLP+SVM+RF+KNN)...')
    std_ens = load_or_train(
        create_stacking_ensemble, input_dim,
        X_tr, y_tr, out_dir, 'standard_clean'
    )
    r_std_clean = evaluate(std_ens, X_te_clean, y_te_clean, 'Standard_Clean')
    r_std_gan   = evaluate(std_ens, X_te_gan,   y_te_gan,   'Standard_GAN')
    results += [r_std_clean, r_std_gan]
    print(f'    Clean  → F1={r_std_clean["f1_score"]:.4f}')
    print(f'    GAN    → F1={r_std_gan["f1_score"]:.4f}')

    # ── GAN-Opt Stacking (MLP_deep+MLP_wide+KNN_5+KNN_11) ───────────────────
    print('\n[2] GAN-Opt Stacking (MLP_deep+MLP_wide+KNN_5+KNN_11)...')
    ganopt_ens = load_or_train(
        create_stacking_ensemble_gan_optimized, input_dim,
        X_tr, y_tr, out_dir, 'ganopt_clean'
    )
    r_ganopt_clean = evaluate(ganopt_ens, X_te_clean, y_te_clean, 'GAN-Opt_Clean')
    r_ganopt_gan   = evaluate(ganopt_ens, X_te_gan,   y_te_gan,   'GAN-Opt_GAN')
    results += [r_ganopt_clean, r_ganopt_gan]
    print(f'    Clean  → F1={r_ganopt_clean["f1_score"]:.4f}')
    print(f'    GAN    → F1={r_ganopt_gan["f1_score"]:.4f}')

    # Save
    df = pd.DataFrame(results)
    csv_path = out_dir / f'exp5b_raw_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(csv_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '='*80)
    print('✅ EXP5b RAW SUMMARY — Stacking vs GAN (Không DeDe)'.center(80))
    print('='*80)
    print(f'\n  {"Model":<25} {"Clean F1":>10} {"GAN F1":>10}')
    print('  ' + '-'*47)
    print(f'  {"Standard (MLP+SVM+RF+KNN)":<25} '
          f'{r_std_clean["f1_score"]:>10.4f} {r_std_gan["f1_score"]:>10.4f}')
    print(f'  {"GAN-Opt (MLP+KNN vars)":<25} '
          f'{r_ganopt_clean["f1_score"]:>10.4f} {r_ganopt_gan["f1_score"]:>10.4f}')
    print(f'  {"DeDe+GAN-Opt (exp7 RAW)":<25} {"0.9674":>10} {"0.9160":>10}  ← reference')

    print('\n  📌 Kết luận hướng tới exp7:')
    delta_ganopt_std = r_ganopt_gan["f1_score"] - r_std_gan["f1_score"]
    delta_exp7_ganopt = 0.9160 - r_ganopt_gan["f1_score"]
    print(f'    GAN-Opt vs Standard:     Δ={delta_ganopt_std:+.4f}  '
          f'(GAN-Opt {"better" if delta_ganopt_std > 0 else "worse"} for GAN)')
    print(f'    DeDe+GAN-Opt vs GAN-Opt: Δ={delta_exp7_ganopt:+.4f}  '
          f'(DeDe {"helps" if delta_exp7_ganopt > 0 else "hurts"} for GAN)')

    print(f'\n📁 {csv_path}\n')


if __name__ == '__main__':
    main()
