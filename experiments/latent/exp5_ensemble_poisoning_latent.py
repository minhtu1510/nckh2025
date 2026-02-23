#!/usr/bin/env python3
"""
Exp5 (FIXED): Stacking Ensemble + Data Poisoning Attack — LATENT Approach

CORRECT Threat Model:
    - Attacker poisons TRAINING data (malicious → benign label flip)
    - Defender trains stacking ensemble on POISONED latent training data
    - Test on CLEAN latent test data
    - Compare: Hybrid Defense (DeDe+Stacking) vs plain ensemble on poisoned latent

This replaces the OLD exp5 which wrongly loaded a pre-trained clean model.

Features: 64 latent dims
Output: results/latent/exp5_stacking_poisoning_fixed/
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(base_latent, poison_rate_str):
    """Load poisoned train + clean test latent data."""
    baseline_dir = base_latent / 'exp1_baseline_latent'
    poison_dir   = base_latent / f'exp2_poisoning/poison_{poison_rate_str}'

    # Train on POISONED data (defender unknowingly trains on compromised data)
    X_train = np.load(poison_dir / 'X_train.npy')
    y_train = np.load(poison_dir / 'y_train.npy')

    # Test on CLEAN data (measure damage)
    X_test  = np.load(baseline_dir / 'X_test.npy')
    y_test  = np.load(baseline_dir / 'y_test.npy')

    # Count flipped labels
    y_train_clean = np.load(baseline_dir / 'y_train.npy')
    n_flipped = (y_train != y_train_clean).sum()

    return X_train, y_train, X_test, y_test, n_flipped


def train_stacking(X_train, y_train, input_dim, output_dir=None, label=''):
    """Train stacking ensemble (MLP+SVM+RF+KNN) on given data."""
    from models.ensemble.stacking import create_stacking_ensemble
    import joblib

    # Check cache
    if output_dir:
        cache = Path(output_dir) / f'cache_{label}'
        if (cache / 'meta_model.pkl').exists():
            print(f'    ✓ Loaded cached stacking: {label}')
            ens = create_stacking_ensemble(input_dim=input_dim)
            ens.meta_model = joblib.load(cache / 'meta_model.pkl')
            for name in ['svm', 'rf', 'knn']:
                p = cache / f'{name}_model.pkl'
                if p.exists(): ens.base_models[name] = joblib.load(p)
            mlp_p = cache / 'mlp_model.keras'
            if mlp_p.exists():
                from tensorflow import keras
                ens.base_models['mlp'] = keras.models.load_model(mlp_p)
            ens.is_fitted = True
            return ens

    print(f'    Training stacking on {label} data ({len(X_train):,} samples, dim={input_dim})...')
    ens = create_stacking_ensemble(input_dim=input_dim)
    ens.fit(X_train, y_train, verbose=False)

    if output_dir:
        cache = Path(output_dir) / f'cache_{label}'
        ens.save(cache)
        print(f'    ✓ Saved cache: {cache}')

    return ens


def evaluate(model, X_test, y_test, label):
    """Evaluate model and return metrics dict."""
    preds = model.predict(X_test)
    return {
        'Model':     label,
        'Accuracy':  round(accuracy_score(y_test, preds), 4),
        'Precision': round(precision_score(y_test, preds, zero_division=0), 4),
        'Recall':    round(recall_score(y_test, preds, zero_division=0), 4),
        'F1-Score':  round(f1_score(y_test, preds, zero_division=0), 4),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--poison-rates', nargs='+', default=['05','10','15','50'])
    parser.add_argument('--output-dir',
                        default=str(BASE_DIR / 'results/latent/exp5_stacking_poisoning_fixed'))
    args = parser.parse_args()

    print('\n' + '='*80)
    print('EXP5 (FIXED): STACKING ENSEMBLE + POISONING — LATENT APPROACH'.center(80))
    print('Correct Threat Model: RETRAIN on poisoned latent data'.center(80))
    print('='*80)

    base_latent = BASE_DIR / 'datasets/splits/3.1_latent'
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Load clean baseline for comparison
    print('\n[STEP 0] Loading clean latent baseline data...')
    X_tr_clean = np.load(base_latent / 'exp1_baseline_latent/X_train.npy')
    y_tr_clean = np.load(base_latent / 'exp1_baseline_latent/y_train.npy')
    X_te_clean = np.load(base_latent / 'exp1_baseline_latent/X_test.npy')
    y_te_clean = np.load(base_latent / 'exp1_baseline_latent/y_test.npy')
    input_dim  = X_tr_clean.shape[1]
    print(f'    dim={input_dim}, train={len(X_tr_clean):,}, test={len(X_te_clean):,}')

    # Train CLEAN stacking ensemble (baseline reference)
    print('\n[STEP 1] Training CLEAN stacking ensemble (reference)...')
    ens_clean = train_stacking(X_tr_clean, y_tr_clean, input_dim,
                               output_base, label='clean')
    m_clean = evaluate(ens_clean, X_te_clean, y_te_clean, 'STACKING_CLEAN')
    print(f'    Clean: Acc={m_clean["Accuracy"]:.4f}, F1={m_clean["F1-Score"]:.4f}')

    all_summary = []

    # For each poison rate: retrain on poisoned latent data
    for rate in args.poison_rates:
        print(f'\n{"="*80}')
        print(f'[Poison {rate}%] — CORRECT THREAT MODEL'.center(80))
        print(f'{"="*80}')

        try:
            X_tr_p, y_tr_p, X_te, y_te, n_flipped = load_data(base_latent, rate)
            flip_pct = n_flipped / len(y_tr_clean) * 100
            print(f'    Flipped labels: {n_flipped:,} ({flip_pct:.1f}%)')
            print(f'    Train on POISONED latent → Test on CLEAN latent')

            # Train stacking on POISONED latent data
            out_rate = output_base / f'poison_{rate}'
            out_rate.mkdir(exist_ok=True)
            ens_p = train_stacking(X_tr_p, y_tr_p, input_dim,
                                   out_rate, label=f'poison_{rate}')

            # Evaluate on clean test
            m = evaluate(ens_p, X_te, y_te, f'STACKING_POISON_{rate}')
            m['n_flipped']  = int(n_flipped)
            m['flip_pct']   = round(flip_pct, 2)
            m['delta_f1']   = round(m['F1-Score'] - m_clean['F1-Score'], 4)

            print(f'\n    Results (test on CLEAN data):')
            print(f'      Accuracy:  {m["Accuracy"]:.4f}  (clean: {m_clean["Accuracy"]:.4f})')
            print(f'      F1-Score:  {m["F1-Score"]:.4f}  (clean: {m_clean["F1-Score"]:.4f})')
            print(f'      ΔF1:       {m["delta_f1"]:+.4f}')

            # Save per-rate summary
            rate_df = pd.DataFrame([m_clean.copy(), m])
            rate_df.to_csv(out_rate / f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                           index=False)
            all_summary.append(m)

        except Exception as e:
            print(f'    ⚠️ Failed poison_{rate}: {e}')
            import traceback; traceback.print_exc()

    # Final comparison table
    print('\n' + '='*80)
    print('✅ FINAL COMPARISON — STACKING ON POISONED LATENT DATA'.center(80))
    print('='*80)
    print(f'\n  {"Attack":<18} {"Acc":>8} {"F1":>8} {"ΔF1":>8} {"Flipped":>10}')
    print('  ' + '-'*56)

    # Print clean first
    print(f'  {"clean":<18} {m_clean["Accuracy"]:>8.4f} {m_clean["F1-Score"]:>8.4f} {"±0":>8} {"0":>10}')
    for m in all_summary:
        label = m.get('Model','').replace('STACKING_POISON_', 'poison_')
        fp    = f'{m["n_flipped"]:,} ({m["flip_pct"]:.1f}%)'
        print(f'  {label:<18} {m["Accuracy"]:>8.4f} {m["F1-Score"]:>8.4f} {m["delta_f1"]:>+8.4f} {fp:>10}')

    # Save all results
    all_df = pd.DataFrame([m_clean] + all_summary)
    all_df.to_csv(output_base / f'stacking_poison_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                  index=False)
    print(f'\n📁 Results: {output_base}')
    print('='*80 + '\n')


if __name__ == '__main__':
    main()
