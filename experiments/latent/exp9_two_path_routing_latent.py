"""
Experiment 9 Latent: Two-Path Routing Ensemble — LATENT Features

Kiến trúc (giống exp9 RAW nhưng dùng dual-encoded latent cho Stage 2):
  RAW input
    ↓ DeDe RAW error score
    ├── error < low_thr  → Standard Stack (latent)   ← "clean-like"
    ├── error < high_thr → GAN-Opt Stack (latent)    ← "GAN-like"
    └── error ≥ high_thr → Block                     ← "trigger"

Train: Latent (dual-encoder, 64-dim)
Test:  RAW → (DeDe error routing) → dual-encode → latent → stacking

Results: results/latent/exp9_two_path_routing/
"""

import sys, json, argparse, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from experiments.dede_adapted.dede_model import build_dede_model
from models.ensemble.stacking import (
    create_stacking_ensemble,
    create_stacking_ensemble_gan_optimized
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


# ── DUAL ENCODER ─────────────────────────────────────────────────────────────

class DualEncoder:
    def __init__(self, models_dir):
        d = Path(models_dir)
        self.benign_enc    = tf.keras.models.load_model(str(d / 'benign_encoder.h5'))
        self.malicious_enc = tf.keras.models.load_model(str(d / 'malicious_encoder.h5'))
        print(f'  ✓ Dual-encoder loaded')

    def encode(self, X_raw, batch_size=2048):
        z_b, z_m = [], []
        for i in range(0, len(X_raw), batch_size):
            b = X_raw[i:i+batch_size].astype(np.float32)
            z_b.append(self.benign_enc.predict(b, verbose=0))
            z_m.append(self.malicious_enc.predict(b, verbose=0))
        return np.hstack([np.concatenate(z_b), np.concatenate(z_m)])


# ── TWO-PATH HYBRID DEFENSE LATENT ───────────────────────────────────────────

class TwoPathHybridLatent:
    """
    Routing bởi DeDe error (RAW space), classify bởi stacking (latent space).
    """
    def __init__(self, dede_raw, dual_enc, std_stack, ganopt_stack, low_thr, high_thr):
        self.dede_raw     = dede_raw
        self.dual_enc     = dual_enc
        self.std_stack    = std_stack
        self.ganopt_stack = ganopt_stack
        self.low_thr      = low_thr
        self.high_thr     = high_thr

    def predict(self, X_raw, return_details=False):
        n    = len(X_raw)
        pred = np.zeros(n, dtype=int)
        errs = self.dede_raw.get_reconstruction_error(X_raw)

        trigger_mask  = errs >= self.high_thr
        ganopt_mask   = (errs >= self.low_thr) & (errs < self.high_thr)
        standard_mask = errs < self.low_thr

        pred[trigger_mask] = 1

        if ganopt_mask.sum() > 0:
            X_lat = self.dual_enc.encode(X_raw[ganopt_mask])
            pred[ganopt_mask] = self.ganopt_stack.predict(X_lat)

        if standard_mask.sum() > 0:
            X_lat = self.dual_enc.encode(X_raw[standard_mask])
            pred[standard_mask] = self.std_stack.predict(X_lat)

        if return_details:
            return pred, {
                'trigger_mask':  trigger_mask,
                'ganopt_mask':   ganopt_mask,
                'standard_mask': standard_mask,
            }
        return pred

    def evaluate(self, X_raw, y):
        pred, det = self.predict(X_raw, return_details=True)
        n = len(X_raw)
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
    print(f'  ✓ DeDe RAW loaded')
    return model


def load_or_train_lat(ens_fn, input_dim, X_tr_lat, y_tr, cache_dir, label):
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
    print(f'\n  Training [{label}] ({len(X_tr_lat):,} × {input_dim}) LATENT...')
    ens = ens_fn(input_dim=input_dim)
    ens.fit(X_tr_lat, y_tr, verbose=False)
    ens.save(cache)
    print(f'  ✓ Saved: {cache}')
    return ens


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent-dir',   default='datasets/splits/3.1_latent')
    parser.add_argument('--raw-dir',      default='datasets/splits/3.0_raw_from_latent')
    parser.add_argument('--dede-raw',     default='experiments/dede_adapted/models_raw')
    parser.add_argument('--dual-enc-dir', default='datasets/splits/3.1_latent/models')
    parser.add_argument('--output-dir',   default='results/latent/exp9_two_path_routing')
    parser.add_argument('--trigger-rate', default='10')
    parser.add_argument('--low-pct',  type=int, default=75)
    parser.add_argument('--high-pct', type=int, default=99)
    args = parser.parse_args()

    print('\n' + '='*80)
    print('EXPERIMENT 9 LATENT: TWO-PATH ROUTING — LATENT'.center(80))
    print(f'Route P{args.low_pct}: Standard | P{args.low_pct}-P{args.high_pct}: GAN-Opt | >P{args.high_pct}: Block'.center(80))
    print('='*80)

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    lat_dir = Path(args.latent_dir)
    raw_dir = Path(args.raw_dir)

    print('\n[1] Loading DeDe RAW...')
    dede_raw = load_dede_raw(args.dede_raw)

    print('\n[2] Loading Dual-Encoder...')
    dual_enc = DualEncoder(args.dual_enc_dir)

    print('\n[3] Calibrating thresholds on RAW clean test...')
    X_te_clean_raw = np.load(raw_dir / 'exp1_baseline/X_test.npy')
    y_te_clean     = np.load(raw_dir / 'exp1_baseline/y_test.npy')
    errs_clean     = dede_raw.get_reconstruction_error(X_te_clean_raw)
    low_thr  = np.percentile(errs_clean, args.low_pct)
    high_thr = np.percentile(errs_clean, args.high_pct)
    print(f'    P{args.low_pct}  (low):  {low_thr:.6f}')
    print(f'    P{args.high_pct} (high): {high_thr:.6f}')

    # Verify routing on test sets
    print('\n[3b] Routing distribution on test sets:')
    X_te_gan_raw = np.load(raw_dir / 'exp3_gan_attack/X_test.npy')
    y_te_gan     = np.load(raw_dir / 'exp3_gan_attack/y_test.npy')
    for name, X_t in [('Clean', X_te_clean_raw), ('GAN', X_te_gan_raw)]:
        errs = dede_raw.get_reconstruction_error(X_t)
        n = len(X_t)
        print(f'    {name}: Standard={( errs < low_thr).mean()*100:.1f}%  '
              f'GAN-Opt={(( errs>=low_thr)&( errs<high_thr)).mean()*100:.1f}%  '
              f'Block={(errs>=high_thr).mean()*100:.1f}%')

    trigger_dir = raw_dir / f'exp5_trigger/trigger_{args.trigger_rate}'

    train_scenarios = [
        ('clean',     lat_dir / 'exp1_baseline_latent'),
        ('poison_05', lat_dir / 'exp2_poisoning/poison_05'),
        ('poison_10', lat_dir / 'exp2_poisoning/poison_10'),
        ('poison_15', lat_dir / 'exp2_poisoning/poison_15'),
        ('poison_50', lat_dir / 'exp2_poisoning/poison_50'),
    ]

    all_results = []
    print('\n' + '='*80)
    print('[4] Running Latent Two-Path Routing Matrix...')
    print('='*80)

    for train_label, train_dir in train_scenarios:
        if not (train_dir / 'X_train.npy').exists():
            print(f'\n  ⚠️  Skip [{train_label}]')
            continue
        X_tr_lat = np.load(train_dir / 'X_train.npy')
        y_tr     = np.load(train_dir / 'y_train.npy')
        print(f'\n  ► [{train_label}] Latent train: {len(X_tr_lat):,} × {X_tr_lat.shape[1]}')

        # Reuse cache from exp7/exp8 latent if available
        exp7_cache_dir = BASE_DIR / 'results/latent/exp7_combined_matrix_latent'
        exp8_cache_dir = BASE_DIR / 'results/latent/exp8_standard_stacking'

        ganopt = load_or_train_lat(create_stacking_ensemble_gan_optimized,
                                   X_tr_lat.shape[1], X_tr_lat, y_tr,
                                   exp7_cache_dir, f'ganopt_lat_{train_label}')
        std = load_or_train_lat(create_stacking_ensemble,
                                X_tr_lat.shape[1], X_tr_lat, y_tr,
                                exp8_cache_dir, f'standard_lat_{train_label}')

        hds = TwoPathHybridLatent(dede_raw, dual_enc, std, ganopt, low_thr, high_thr)

        m_c = hds.evaluate(X_te_clean_raw, y_te_clean)
        all_results.append({'train_scenario': train_label, 'test_type': 'clean', **m_c})
        print(f'    [Clean]   F1={m_c["f1_score"]:.4f}  '
              f'(Std={m_c["route_standard"]:.0f}% GAN={m_c["route_ganopt"]:.0f}% Blk={m_c["route_trigger"]:.0f}%)')

        m_g = hds.evaluate(X_te_gan_raw, y_te_gan)
        all_results.append({'train_scenario': train_label, 'test_type': 'gan_attack', **m_g})
        print(f'    [GAN]     F1={m_g["f1_score"]:.4f}  '
              f'(Std={m_g["route_standard"]:.0f}% GAN={m_g["route_ganopt"]:.0f}% Blk={m_g["route_trigger"]:.0f}%)')

        m_t = hds.evaluate_trigger(trigger_dir)
        all_results.append({'train_scenario': train_label, 'test_type': f'trigger_{args.trigger_rate}', **m_t})
        print(f'    [Trigger] F1={m_t["f1_score"]:.4f}  ASR={m_t["asr"]:.2f}%')

    # Save
    df = pd.DataFrame(all_results)
    csv_path = out_dir / 'exp9_latent_results.csv'
    df.to_csv(csv_path, index=False)
    with open(out_dir / 'exp9_latent_config.json', 'w') as f:
        json.dump({'low_pct': args.low_pct, 'high_pct': args.high_pct,
                   'low_thr': float(low_thr), 'high_thr': float(high_thr)}, f, indent=2)

    # Print matrix
    print('\n' + '='*80)
    print('✅ EXP9 LATENT — Two-Path Routing'.center(80))
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

    # Compare exp7 vs exp9
    exp7_lat = {'clean': 0.9670, 'gan_attack': 0.9251, f'trigger_{args.trigger_rate}': 0.9751}
    print('\n📊 exp7 LAT(GAN-Opt only) vs exp9 LAT(Two-Path Routing) — clean model:')
    print(f'\n  {"Test":<20} {"exp7 GAN-Opt":>14} {"exp9 Routing":>14} {"Δ":>8}')
    print('  ' + '-'*58)
    for te, e7_f1 in exp7_lat.items():
        row = next((r for r in all_results if r['train_scenario']=='clean' and r['test_type']==te), None)
        if row:
            e9_f1 = row['f1_score']
            delta = e9_f1 - e7_f1
            tag   = '← routing ✅' if delta > 0.003 else ('← ganopt ✅' if delta < -0.003 else '≈ equal')
            print(f'  {te:<20} {e7_f1:>14.4f} {e9_f1:>14.4f} {delta:>+8.4f}  {tag}')

    print(f'\n📁 {csv_path}\n')


if __name__ == '__main__':
    main()
