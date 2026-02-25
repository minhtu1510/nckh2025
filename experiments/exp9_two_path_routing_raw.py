"""
Experiment 9 RAW: Two-Path Routing Ensemble — RAW Features

Đề xuất 2: Dùng DeDe reconstruction error để ROUTE sang ensemble phù hợp.

Kiến trúc:
  RAW input
    ↓ DeDe RAW error score (liên tục)
    ├── error < low_thr  → "clean-like"  → Standard Stack (MLP+SVM+RF+KNN)
    ├── error < high_thr → "GAN-like"    → GAN-Opt Stack (MLP_deep+MLP_wide+KNN_5+KNN_11)
    └── error ≥ high_thr → "trigger"     → Block (predict=1 malicious)

Thresholds:
  high_thr = percentile 99 của clean test errors (giống exp7)
  low_thr  = percentile [low_pct] của clean test errors (mặc định 75)
  
  → low_pct=75: 75% samples (clean-like) → Standard, 24% (GAN-like) → GAN-Opt, 1% → Block

So sánh với:
  exp7 RAW (DeDe + GAN-Opt only)
  exp8 RAW (DeDe + Standard only)

Results: results/raw/exp9_two_path_routing/
"""

import sys, json, argparse, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from experiments.dede_adapted.dede_model import build_dede_model
from models.ensemble.stacking import (
    create_stacking_ensemble,
    create_stacking_ensemble_gan_optimized
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


# ── TWO-PATH HYBRID DEFENSE ───────────────────────────────────────────────────

class TwoPathHybridDefense:
    """
    3-way routing dựa trên DeDe reconstruction error:
      error < low_thr  → Standard Stacking (tốt cho clean/poison)
      error < high_thr → GAN-Opt Stacking  (tốt cho GAN)
      error ≥ high_thr → Block as malicious (tốt cho trigger)
    """
    def __init__(self, dede_raw, std_stack, ganopt_stack, low_thr, high_thr):
        self.dede_raw    = dede_raw
        self.std_stack   = std_stack
        self.ganopt_stack = ganopt_stack
        self.low_thr     = low_thr
        self.high_thr    = high_thr

    def predict(self, X, return_details=False):
        n    = len(X)
        pred = np.zeros(n, dtype=int)
        errs = self.dede_raw.get_reconstruction_error(X)

        # Route 3 nhóm
        trigger_mask  = errs >= self.high_thr                         # Block
        ganopt_mask   = (errs >= self.low_thr) & (errs < self.high_thr)  # GAN-Opt
        standard_mask = errs < self.low_thr                           # Standard

        pred[trigger_mask] = 1  # Blocked → malicious

        if ganopt_mask.sum() > 0:
            pred[ganopt_mask] = self.ganopt_stack.predict(X[ganopt_mask])

        if standard_mask.sum() > 0:
            pred[standard_mask] = self.std_stack.predict(X[standard_mask])

        if return_details:
            return pred, {
                'trigger_mask':  trigger_mask,
                'ganopt_mask':   ganopt_mask,
                'standard_mask': standard_mask,
                'errors':        errs,
            }
        return pred

    def evaluate(self, X, y):
        pred, det = self.predict(X, return_details=True)
        n  = len(X)
        return {
            'accuracy':       round(accuracy_score(y, pred), 6),
            'precision':      round(precision_score(y, pred, zero_division=0), 6),
            'recall':         round(recall_score(y, pred, zero_division=0), 6),
            'f1_score':       round(f1_score(y, pred, zero_division=0), 6),
            'route_trigger':  round(det['trigger_mask'].sum() / n * 100, 2),
            'route_ganopt':   round(det['ganopt_mask'].sum() / n * 100, 2),
            'route_standard': round(det['standard_mask'].sum() / n * 100, 2),
            'asr':            None,
        }

    def evaluate_trigger(self, trigger_dir):
        tdir  = Path(trigger_dir)
        X_mal = np.load(tdir / 'X_test_malicious_triggered.npy')
        X_ben = np.load(tdir / 'X_test_benign_clean.npy')
        X_mix = np.load(tdir / 'X_test_mixed_realistic.npy')
        y_mix = np.load(tdir / 'y_test_mixed_realistic.npy')

        errs_mal    = self.dede_raw.get_reconstruction_error(X_mal)
        blocked_mal = (errs_mal >= self.high_thr).sum()
        asr         = ((len(X_mal) - blocked_mal) / len(X_mal)) * 100
        fp_rate     = (self.dede_raw.get_reconstruction_error(X_ben) >= self.high_thr).mean() * 100

        pred_mix, det = self.predict(X_mix, return_details=True)
        n = len(X_mix)
        return {
            'accuracy':       round(accuracy_score(y_mix, pred_mix), 6),
            'precision':      round(precision_score(y_mix, pred_mix, zero_division=0), 6),
            'recall':         round(recall_score(y_mix, pred_mix, zero_division=0), 6),
            'f1_score':       round(f1_score(y_mix, pred_mix, zero_division=0), 6),
            'route_trigger':  round(det['trigger_mask'].sum() / n * 100, 2),
            'route_ganopt':   round(det['ganopt_mask'].sum() / n * 100, 2),
            'route_standard': round(det['standard_mask'].sum() / n * 100, 2),
            'asr':            round(asr, 4),
            'false_positive_rate': round(fp_rate, 2),
        }


# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_dede_raw(model_dir):
    with open(Path(model_dir) / 'training_config.json') as f:
        cfg = json.load(f)
    model = build_dede_model(
        input_dim=cfg['input_dim'], latent_dim=cfg.get('latent_dim', 64),
        encoder_hidden_dims=[256, 128], decoder_hidden_dims=[128, 256],
        mask_ratio=cfg.get('mask_ratio', 0.5), dropout=0.2,
        learning_rate=cfg.get('learning_rate', 0.001)
    )
    _ = model(tf.zeros((1, cfg['input_dim'])), training=False)
    model.load_weights(str(Path(model_dir) / 'best_model.weights.h5'))
    print(f'  ✓ DeDe RAW: input_dim={cfg["input_dim"]}')
    return model


def load_or_train(ens_fn, input_dim, X_tr, y_tr, cache_dir, label):
    cache = Path(cache_dir) / label
    if (cache / 'meta_model.pkl').exists():
        ens = ens_fn(input_dim=input_dim)
        ens.meta_model = joblib.load(cache / 'meta_model.pkl')
        for name in list(ens.base_models.keys()):
            p_pkl   = cache / f'{name}_model.pkl'
            p_keras = cache / f'{name}_model.keras'
            if p_pkl.exists():    ens.base_models[name] = joblib.load(p_pkl)
            elif p_keras.exists():
                from tensorflow import keras
                ens.base_models[name] = keras.models.load_model(p_keras)
        ens.is_fitted = True
        print(f'  ✓ Loaded cache: {label}')
        return ens
    print(f'\n  Training [{label}] ({len(X_tr):,} × {input_dim})...')
    ens = ens_fn(input_dim=input_dim)
    ens.fit(X_tr, y_tr, verbose=False)
    ens.save(cache)
    print(f'  ✓ Saved: {cache}')
    return ens


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',      default='datasets/splits/3.0_raw_from_latent')
    parser.add_argument('--dede-raw',      default='experiments/dede_adapted/models_raw')
    parser.add_argument('--output-dir',    default='results/raw/exp9_two_path_routing')
    parser.add_argument('--trigger-rate',  default='10')
    parser.add_argument('--low-pct',  type=int, default=75,
                        help='Percentile cho low threshold (route to Standard Stacking)')
    parser.add_argument('--high-pct', type=int, default=99,
                        help='Percentile cho high threshold (block as trigger)')
    args = parser.parse_args()

    print('\n' + '='*80)
    print('EXPERIMENT 9 RAW: TWO-PATH ROUTING ENSEMBLE'.center(80))
    print(f'Routing: error<P{args.low_pct}→Standard | P{args.low_pct}~P{args.high_pct}→GAN-Opt | >P{args.high_pct}→Block'.center(80))
    print('='*80)

    out_dir  = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir  = Path(args.data_dir)

    # Load DeDe
    print('\n[1] Loading DeDe RAW...')
    dede_raw = load_dede_raw(args.dede_raw)

    # Calibrate thresholds on clean test set
    print('\n[2] Calibrating thresholds on clean test set...')
    X_te_clean = np.load(raw_dir / 'exp1_baseline/X_test.npy')
    y_te_clean = np.load(raw_dir / 'exp1_baseline/y_test.npy')
    errs_clean = dede_raw.get_reconstruction_error(X_te_clean)
    low_thr  = np.percentile(errs_clean, args.low_pct)
    high_thr = np.percentile(errs_clean, args.high_pct)
    print(f'    Low threshold  (P{args.low_pct}): {low_thr:.6f}  → Standard Stacking')
    print(f'    High threshold (P{args.high_pct}): {high_thr:.6f}  → Block (trigger)')
    print(f'    Gap: P{args.low_pct}-P{args.high_pct} → GAN-Opt Stacking')

    # Check routing distribution on test sets
    print('\n[2b] Verifying routing distribution...')
    for name, X_t in [('Clean', X_te_clean),
                      ('GAN', np.load(raw_dir / 'exp3_gan_attack/X_test.npy'))]:
        errs = dede_raw.get_reconstruction_error(X_t)
        n    = len(X_t)
        pct_std = (errs < low_thr).mean() * 100
        pct_gan = ((errs >= low_thr) & (errs < high_thr)).mean() * 100
        pct_blk = (errs >= high_thr).mean() * 100
        print(f'    {name:6s}: Standard={pct_std:.1f}%  GAN-Opt={pct_gan:.1f}%  Block={pct_blk:.1f}%')

    # Load test sets
    X_te_gan = np.load(raw_dir / 'exp3_gan_attack/X_test.npy')
    y_te_gan = np.load(raw_dir / 'exp3_gan_attack/y_test.npy')
    trigger_dir = raw_dir / f'exp5_trigger/trigger_{args.trigger_rate}'

    # Training scenarios
    train_scenarios = [
        ('clean',     raw_dir / 'exp1_baseline'),
        ('poison_05', raw_dir / 'exp2_poisoning/poison_05'),
        ('poison_10', raw_dir / 'exp2_poisoning/poison_10'),
        ('poison_15', raw_dir / 'exp2_poisoning/poison_15'),
        ('poison_50', raw_dir / 'exp2_poisoning/poison_50'),
    ]

    all_results = []

    print('\n' + '='*80)
    print('[3] Running Two-Path Routing Matrix...')
    print('='*80)

    for train_label, train_dir in train_scenarios:
        if not (train_dir / 'X_train.npy').exists():
            print(f'\n  ⚠️  Skip [{train_label}]')
            continue

        X_tr = np.load(train_dir / 'X_train.npy')
        y_tr = np.load(train_dir / 'y_train.npy')
        print(f'\n  ► [{train_label}] Train: {len(X_tr):,} × {X_tr.shape[1]}')

        # Load or train BOTH ensembles (reuse cache từ exp7/exp8 nếu có)
        # Try exp7 cache first, then exp8, then train new
        exp7_cache = BASE_DIR / f'results/raw/exp7_combined_matrix/ganopt_{train_label}'
        exp8_cache = BASE_DIR / f'results/raw/exp8_standard_stacking/standard_{train_label}'

        # GAN-Opt
        if exp7_cache.exists() and (exp7_cache / 'meta_model.pkl').exists():
            ganopt = load_or_train(create_stacking_ensemble_gan_optimized,
                                  X_tr.shape[1], X_tr, y_tr, out_dir, f'ganopt_{train_label}')
            # actually try to load from exp7 cache directly
            ganopt_src = exp7_cache
            ganopt_dir_use = ganopt_src
            ganopt = load_or_train(create_stacking_ensemble_gan_optimized,
                                  X_tr.shape[1], X_tr, y_tr,
                                  BASE_DIR / 'results/raw/exp7_combined_matrix', f'ganopt_{train_label}')
        else:
            ganopt = load_or_train(create_stacking_ensemble_gan_optimized,
                                  X_tr.shape[1], X_tr, y_tr, out_dir, f'ganopt_{train_label}')

        # Standard
        if exp8_cache.exists() and (exp8_cache / 'meta_model.pkl').exists():
            std = load_or_train(create_stacking_ensemble,
                               X_tr.shape[1], X_tr, y_tr,
                               BASE_DIR / 'results/raw/exp8_standard_stacking', f'standard_{train_label}')
        else:
            std = load_or_train(create_stacking_ensemble,
                               X_tr.shape[1], X_tr, y_tr, out_dir, f'standard_{train_label}')

        hds = TwoPathHybridDefense(dede_raw, std, ganopt, low_thr, high_thr)

        # Evaluate
        m_c = hds.evaluate(X_te_clean, y_te_clean)
        all_results.append({'train_scenario': train_label, 'test_type': 'clean', **m_c})
        print(f'    [Clean]   F1={m_c["f1_score"]:.4f}  '
              f'(Std={m_c["route_standard"]:.0f}% GAN={m_c["route_ganopt"]:.0f}% Blk={m_c["route_trigger"]:.0f}%)')

        m_g = hds.evaluate(X_te_gan, y_te_gan)
        all_results.append({'train_scenario': train_label, 'test_type': 'gan_attack', **m_g})
        print(f'    [GAN]     F1={m_g["f1_score"]:.4f}  '
              f'(Std={m_g["route_standard"]:.0f}% GAN={m_g["route_ganopt"]:.0f}% Blk={m_g["route_trigger"]:.0f}%)')

        m_t = hds.evaluate_trigger(trigger_dir)
        all_results.append({'train_scenario': train_label, 'test_type': f'trigger_{args.trigger_rate}', **m_t})
        print(f'    [Trigger] F1={m_t["f1_score"]:.4f}  ASR={m_t["asr"]:.2f}%')

    # Save
    df = pd.DataFrame(all_results)
    csv_path = out_dir / 'exp9_raw_results.csv'
    df.to_csv(csv_path, index=False)

    with open(out_dir / 'exp9_config.json', 'w') as f:
        json.dump({'low_pct': args.low_pct, 'high_pct': args.high_pct,
                   'low_thr': float(low_thr), 'high_thr': float(high_thr)}, f, indent=2)

    # Print matrix + compare with exp7 & exp8
    print('\n' + '='*80)
    print('✅ EXP9 RAW — Two-Path Routing Hybrid'.center(80))
    print('='*80)
    header = 'Train \\ Test'
    print(f'\n  {header:<16} {"Clean F1":>10} {"GAN F1":>10} {"Trigger F1":>12} {"ASR":>8}')
    print('  ' + '-'*60)
    for train_label, _ in train_scenarios:
        rows = {r['test_type']: r for r in all_results if r['train_scenario'] == train_label}
        if not rows: continue
        f1_c = rows.get('clean', {}).get('f1_score', float('nan'))
        f1_g = rows.get('gan_attack', {}).get('f1_score', float('nan'))
        f1_t = rows.get(f'trigger_{args.trigger_rate}', {}).get('f1_score', float('nan'))
        asr  = rows.get(f'trigger_{args.trigger_rate}', {}).get('asr', float('nan'))
        print(f'  {train_label:<16} {f1_c:>10.4f} {f1_g:>10.4f} {f1_t:>12.4f} {asr:>7.2f}%')

    # Compare clean model on 3 tests
    print('\n📊 So sánh Clean Model: exp7(GAN-Opt) vs exp8(Standard) vs exp9(Routing):')
    exp7_vals = {'clean': 0.9674, 'gan_attack': 0.9160, f'trigger_{args.trigger_rate}': 0.9755}
    exp8_vals = {}  # Will be read if exp8 results exist
    print(f'\n  {"Test":<20} {"exp7 GAN-Opt":>14} {"exp9 Routing":>14}')
    print('  ' + '-'*50)
    for te, exp7_f1 in exp7_vals.items():
        row = next((r for r in all_results if r['train_scenario']=='clean' and r['test_type']==te), None)
        if row:
            exp9_f1 = row['f1_score']
            delta   = exp9_f1 - exp7_f1
            tag     = '← routing better ✅' if delta > 0.003 else ('← ganopt better' if delta < -0.003 else '≈ equal')
            print(f'  {te:<20} {exp7_f1:>14.4f} {exp9_f1:>14.4f}  {delta:>+.4f}  {tag}')

    print(f'\n📁 {csv_path}\n')


if __name__ == '__main__':
    main()
