"""
Experiment 8 Latent: DeDe + Standard Stacking (MLP+SVM+RF+KNN) — LATENT Features

So sánh với exp7 Latent (DeDe + GAN-Opt):
  exp7: DeDe + GAN-Opt  (MLP_deep + MLP_wide + KNN_5 + KNN_11) — loại SVM, RF
  exp8: DeDe + Standard (MLP + SVM + RF + KNN)                  — giữ SVM, RF

Train: Latent (64-dim dual-encoder)
Test:  RAW → dual-encoder → latent

Results: results/latent/exp8_standard_stacking/
"""

import sys, json, argparse, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from experiments.dede_adapted.dede_model import build_dede_model
from models.ensemble.stacking import create_stacking_ensemble   # ← Standard
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


# ── DUAL ENCODER ─────────────────────────────────────────────────────────────

class DualEncoder:
    def __init__(self, models_dir):
        models_dir = Path(models_dir)
        self.benign_enc    = tf.keras.models.load_model(str(models_dir / 'benign_encoder.h5'))
        self.malicious_enc = tf.keras.models.load_model(str(models_dir / 'malicious_encoder.h5'))
        print(f'  ✓ Dual-encoder loaded from {models_dir.name}')

    def encode(self, X_raw, batch_size=2048):
        n = len(X_raw)
        z_b, z_m = [], []
        for i in range(0, n, batch_size):
            b = X_raw[i:i+batch_size].astype(np.float32)
            z_b.append(self.benign_enc.predict(b, verbose=0))
            z_m.append(self.malicious_enc.predict(b, verbose=0))
        return np.hstack([np.concatenate(z_b), np.concatenate(z_m)])


# ── HYBRID DEFENSE ────────────────────────────────────────────────────────────

class HybridDefenseLatentStandard:
    def __init__(self, dede_raw, dual_enc, stacking, thr_raw):
        self.dede_raw = dede_raw
        self.dual_enc = dual_enc
        self.ensemble = stacking
        self.thr_raw  = thr_raw

    def predict(self, X_raw, return_details=False):
        n    = len(X_raw)
        pred = np.zeros(n, dtype=int)
        errs = self.dede_raw.get_reconstruction_error(X_raw)
        mask = errs > self.thr_raw
        pred[mask] = 1
        if (~mask).sum() > 0:
            X_lat = self.dual_enc.encode(X_raw[~mask])
            pred[~mask] = self.ensemble.predict(X_lat)
        if return_details:
            return pred, {'trigger_mask': mask}
        return pred

    def evaluate(self, X_raw, y):
        pred, det = self.predict(X_raw, return_details=True)
        s1 = det['trigger_mask'].sum()
        return {
            'accuracy':   round(accuracy_score(y, pred), 6),
            'precision':  round(precision_score(y, pred, zero_division=0), 6),
            'recall':     round(recall_score(y, pred, zero_division=0), 6),
            'f1_score':   round(f1_score(y, pred, zero_division=0), 6),
            'stage1_pct': round(s1 / len(X_raw) * 100, 2),
            'asr':        None,
        }

    def evaluate_trigger(self, trigger_dir):
        tdir  = Path(trigger_dir)
        X_mal = np.load(tdir / 'X_test_malicious_triggered.npy')
        X_ben = np.load(tdir / 'X_test_benign_clean.npy')
        X_mix = np.load(tdir / 'X_test_mixed_realistic.npy')
        y_mix = np.load(tdir / 'y_test_mixed_realistic.npy')

        errs_mal    = self.dede_raw.get_reconstruction_error(X_mal)
        blocked_mal = (errs_mal > self.thr_raw).sum()
        asr         = ((len(X_mal) - blocked_mal) / len(X_mal)) * 100
        fp_rate     = (self.dede_raw.get_reconstruction_error(X_ben) > self.thr_raw).mean() * 100
        pred_mix, _ = self.predict(X_mix, return_details=True)
        return {
            'accuracy':            round(accuracy_score(y_mix, pred_mix), 6),
            'precision':           round(precision_score(y_mix, pred_mix, zero_division=0), 6),
            'recall':              round(recall_score(y_mix, pred_mix, zero_division=0), 6),
            'f1_score':            round(f1_score(y_mix, pred_mix, zero_division=0), 6),
            'stage1_pct':          round(blocked_mal / len(X_mal) * 100, 2),
            'asr':                 round(asr, 4),
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
    print(f'  ✓ DeDe RAW: input_dim={cfg["input_dim"]}, val_loss={cfg["best_val_loss"]:.4f}')
    return model


def load_or_train_standard_stacking_lat(X_tr_lat, y_tr, label, save_dir):
    """Standard Stacking trên LATENT features (MLP+SVM+RF+KNN)."""
    cache = Path(save_dir) / f'standard_lat_{label}'
    if (cache / 'meta_model.pkl').exists():
        ens = create_stacking_ensemble(input_dim=X_tr_lat.shape[1])
        ens.meta_model = joblib.load(cache / 'meta_model.pkl')
        for name in ['svm', 'rf', 'knn']:
            p = cache / f'{name}_model.pkl'
            if p.exists(): ens.base_models[name] = joblib.load(p)
        mlp_p = cache / 'mlp_model.keras'
        if mlp_p.exists():
            from tensorflow import keras
            ens.base_models['mlp'] = keras.models.load_model(mlp_p)
        ens.is_fitted = True
        print(f'  ✓ Loaded standard latent cache: {label}')
        return ens

    print(f'\n  Training Standard Stacking [{label}] ({len(X_tr_lat):,} × {X_tr_lat.shape[1]}) [MLP+SVM+RF+KNN] LATENT...')
    ens = create_stacking_ensemble(input_dim=X_tr_lat.shape[1])
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
    parser.add_argument('--output-dir',   default='results/latent/exp8_standard_stacking')
    parser.add_argument('--trigger-rate', default='10')
    parser.add_argument('--threshold-pct', type=int, default=99)
    args = parser.parse_args()

    print('\n' + '='*80)
    print('EXPERIMENT 8 LATENT: DeDe + Standard Stacking (MLP+SVM+RF+KNN)'.center(80))
    print('Train: LATENT | Test: RAW → dual-encode → latent'.center(80))
    print('='*80)

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    lat_dir = Path(args.latent_dir)
    raw_dir = Path(args.raw_dir)

    print('\n[1] Loading DeDe RAW...')
    dede_raw = load_dede_raw(args.dede_raw)

    print('\n[2] Loading Dual-Encoder...')
    dual_enc = DualEncoder(args.dual_enc_dir)

    print('\n[3] Calibrating threshold...')
    X_te_clean_raw = np.load(raw_dir / 'exp1_baseline/X_test.npy')
    y_te_clean     = np.load(raw_dir / 'exp1_baseline/y_test.npy')
    thr = np.percentile(dede_raw.get_reconstruction_error(X_te_clean_raw), args.threshold_pct)
    print(f'    Threshold: {thr:.6f}')

    print('\n[4] Loading FIXED test sets (RAW)...')
    X_te_gan_raw = np.load(raw_dir / 'exp3_gan_attack/X_test.npy')
    y_te_gan     = np.load(raw_dir / 'exp3_gan_attack/y_test.npy')
    trigger_dir  = raw_dir / f'exp5_trigger/trigger_{args.trigger_rate}'
    print(f'    Clean: {len(X_te_clean_raw):,} | GAN: {len(X_te_gan_raw):,}')

    train_scenarios = [
        ('clean',     lat_dir / 'exp1_baseline_latent'),
        ('poison_05', lat_dir / 'exp2_poisoning/poison_05'),
        ('poison_10', lat_dir / 'exp2_poisoning/poison_10'),
        ('poison_15', lat_dir / 'exp2_poisoning/poison_15'),
        ('poison_50', lat_dir / 'exp2_poisoning/poison_50'),
    ]

    all_results = []
    print('\n' + '='*80)
    print('[5] Running Matrix: Latent Standard Stacking (MLP+SVM+RF+KNN) + DeDe')
    print('='*80)

    for train_label, train_dir in train_scenarios:
        if not (train_dir / 'X_train.npy').exists():
            print(f'\n  ⚠️  Skip [{train_label}]')
            continue
        X_tr_lat = np.load(train_dir / 'X_train.npy')
        y_tr     = np.load(train_dir / 'y_train.npy')
        print(f'\n  ► [{train_label}] Latent train: {len(X_tr_lat):,} × {X_tr_lat.shape[1]}')

        ens = load_or_train_standard_stacking_lat(X_tr_lat, y_tr, train_label, out_dir)
        hds = HybridDefenseLatentStandard(dede_raw, dual_enc, ens, thr)

        m_c = hds.evaluate(X_te_clean_raw, y_te_clean)
        all_results.append({'train_scenario': train_label, 'test_type': 'clean', **m_c})
        print(f'    [Clean]   F1={m_c["f1_score"]:.4f}')

        m_g = hds.evaluate(X_te_gan_raw, y_te_gan)
        all_results.append({'train_scenario': train_label, 'test_type': 'gan_attack', **m_g})
        print(f'    [GAN]     F1={m_g["f1_score"]:.4f}')

        m_t = hds.evaluate_trigger(trigger_dir)
        all_results.append({'train_scenario': train_label, 'test_type': f'trigger_{args.trigger_rate}', **m_t})
        print(f'    [Trigger] F1={m_t["f1_score"]:.4f}  ASR={m_t["asr"]:.2f}%')

    # Save
    df = pd.DataFrame(all_results)
    csv_path = out_dir / 'exp8_latent_results.csv'
    df.to_csv(csv_path, index=False)

    # Print matrix
    print('\n' + '='*80)
    print('✅ EXP8 LATENT — Standard Stacking + DeDe'.center(80))
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

    # Compare exp7 latent vs exp8 latent
    exp7_lat_ref = {
        ('clean', 'clean'): 0.9670,
        ('clean', 'gan_attack'): 0.9251,
        ('clean', f'trigger_{args.trigger_rate}'): 0.9751,
        ('poison_50', 'clean'): 0.9338,
        ('poison_50', 'gan_attack'): 0.9215,
    }
    print('\n📊 EXP7 LAT (GAN-Opt) vs EXP8 LAT (Standard):')
    print(f'  {"Scenario":<30} {"exp7 GAN-Opt":>13} {"exp8 Standard":>14} {"Δ":>8}')
    print('  ' + '-'*68)
    for (tr, te), ref_f1 in exp7_lat_ref.items():
        row = next((r for r in all_results if r['train_scenario']==tr and r['test_type']==te), None)
        if row:
            std_f1 = row['f1_score']
            delta  = std_f1 - ref_f1
            tag    = '← std better' if delta > 0.003 else ('← ganopt better' if delta < -0.003 else '≈ equal')
            print(f'  {f"{tr}/{te}":<30} {ref_f1:>13.4f} {std_f1:>14.4f} {delta:>+8.4f}  {tag}')

    print(f'\n📁 {csv_path}\n')


if __name__ == '__main__':
    main()
