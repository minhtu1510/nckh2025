"""
Compute ROC-AUC cho tất cả experiments đã chạy.

Script này load các stacking models đã cache từ exp7/exp8/exp9
và compute thêm ROC-AUC, PR-AUC, MCC cho mỗi (train_scenario, test_type).

Chạy sau khi exp7, exp8, exp9 đã hoàn thành.

Output:
  results/summary/auc_results_raw.csv
  results/summary/auc_results_latent.csv
  results/summary/roc_curves/  (PNG files)
"""

import sys, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
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
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    matthews_corrcoef, f1_score, accuracy_score,
    roc_curve, precision_recall_curve
)
import joblib


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
    return model


def try_load_stacking(ens_fn, input_dim, cache_dir, label):
    """Load stacking từ cache, skip nếu chưa có."""
    cache = Path(cache_dir) / label
    if not (cache / 'meta_model.pkl').exists():
        return None
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
    return ens


def get_proba(model, X):
    """Lấy probability scores từ stacking meta_model."""
    try:
        return model.meta_model.predict_proba(
            model._get_meta_features(X)
        )[:, 1]
    except Exception:
        pred = model.predict(X)
        return pred.astype(float)


def hybrid_predict_proba(dede_raw, stacking, thr_raw, X, dual_enc=None):
    """
    Trả về probability scores cho hybrid defense:
    - DeDe error → normalized score cho triggered samples
    - Stacking proba cho remaining samples
    """
    n     = len(X)
    proba = np.zeros(n)
    errs  = dede_raw.get_reconstruction_error(X)

    max_err = errs.max() if errs.max() > 0 else 1.0
    dede_scores = errs / max_err

    mask = errs > thr_raw
    proba[mask] = np.clip(dede_scores[mask], 0.8, 1.0)

    if (~mask).sum() > 0:
        X_pass = X[~mask]
        if dual_enc is not None:
            X_pass = dual_enc.encode(X_pass)
        try:
            stack_proba = get_proba(stacking, X_pass)
        except Exception:
            stack_proba = stacking.predict(X_pass).astype(float)
        proba[~mask] = stack_proba

    return proba


def routing_predict_proba(dede_raw, std_stack, ganopt_stack,
                          low_thr, high_thr, X, dual_enc=None):
    """Exp9 Two-Path Routing probability scores."""
    n     = len(X)
    proba = np.zeros(n)
    errs  = dede_raw.get_reconstruction_error(X)

    max_err = errs.max() if errs.max() > 0 else 1.0
    dede_scores = errs / max_err

    trigger_mask  = errs >= high_thr
    ganopt_mask   = (errs >= low_thr) & (~trigger_mask)
    standard_mask = errs < low_thr

    proba[trigger_mask] = np.clip(dede_scores[trigger_mask], 0.8, 1.0)

    for mask, stack in [(ganopt_mask, ganopt_stack), (standard_mask, std_stack)]:
        if mask.sum() > 0:
            Xm = X[mask]
            if dual_enc is not None:
                Xm = dual_enc.encode(Xm)
            try:
                proba[mask] = get_proba(stack, Xm)
            except Exception:
                proba[mask] = stack.predict(Xm).astype(float)

    return proba


def compute_metrics(y_true, y_pred, y_proba):
    return {
        'f1_score':     round(f1_score(y_true, y_pred, zero_division=0), 4),
        'accuracy':     round(accuracy_score(y_true, y_pred), 4),
        'roc_auc':      round(roc_auc_score(y_true, y_proba), 4),
        'pr_auc':       round(average_precision_score(y_true, y_proba), 4),
        'mcc':          round(matthews_corrcoef(y_true, y_pred), 4),
    }


def plot_roc(roc_data, title, save_path):
    """Plot ROC curves cho nhiều scenarios."""
    plt.figure(figsize=(8, 6))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
    for i, (label, fpr, tpr, auc) in enumerate(roc_data):
        plt.plot(fpr, tpr, color=colors[i % len(colors)],
                 linewidth=2, label=f'{label} (AUC={auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    plt.xlim([0, 1]); plt.ylim([0, 1.02])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  📊 Saved: {save_path}')


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print('\n' + '='*80)
    print('ROC-AUC EVALUATION — Tất cả experiments'.center(80))
    print('='*80)

    raw_dir  = BASE_DIR / 'datasets/splits/3.0_raw_from_latent'
    lat_dir  = BASE_DIR / 'datasets/splits/3.1_latent'
    out_dir  = BASE_DIR / 'results/summary'
    roc_dir  = out_dir / 'roc_curves'
    out_dir.mkdir(parents=True, exist_ok=True)
    roc_dir.mkdir(parents=True, exist_ok=True)

    # Load DeDe RAW
    print('\n[1] Loading DeDe RAW...')
    dede_raw = load_dede_raw(BASE_DIR / 'experiments/dede_adapted/models_raw')

    # Calibrate threshold
    X_te_clean_raw = np.load(raw_dir / 'exp1_baseline/X_test.npy')
    y_te_clean     = np.load(raw_dir / 'exp1_baseline/y_test.npy')
    thr_raw = np.percentile(dede_raw.get_reconstruction_error(X_te_clean_raw), 99)

    # Load GAN test
    X_te_gan = np.load(raw_dir / 'exp3_gan_attack/X_test.npy')
    y_te_gan = np.load(raw_dir / 'exp3_gan_attack/y_test.npy')

    # Trigger test (malicious triggered)
    trig_dir = raw_dir / 'exp5_trigger/trigger_10'
    X_trig   = np.load(trig_dir / 'X_test_malicious_triggered.npy')
    y_trig   = np.load(trig_dir / 'y_test_malicious_triggered.npy')
    X_ben    = np.load(trig_dir / 'X_test_benign_clean.npy')
    y_ben    = np.load(trig_dir / 'y_test_benign_clean.npy')
    X_mix    = np.load(trig_dir / 'X_test_mixed_realistic.npy')
    y_mix    = np.load(trig_dir / 'y_test_mixed_realistic.npy')

    # Load Dual Encoder for latent
    try:
        benign_enc    = tf.keras.models.load_model(str(lat_dir / 'models/benign_encoder.h5'))
        malicious_enc = tf.keras.models.load_model(str(lat_dir / 'models/malicious_encoder.h5'))
        class DualEnc:
            def encode(self, X, bs=2048):
                zb, zm = [], []
                for i in range(0, len(X), bs):
                    b = X[i:i+bs].astype(np.float32)
                    zb.append(benign_enc.predict(b, verbose=0))
                    zm.append(malicious_enc.predict(b, verbose=0))
                return np.hstack([np.concatenate(zb), np.concatenate(zm)])
        dual_enc = DualEnc()
        print('  ✓ Dual-encoder loaded')
    except Exception as e:
        dual_enc = None
        print(f'  ⚠️  Dual-encoder not available: {e}')

    all_results = []
    train_scenarios = ['clean', 'poison_05', 'poison_10', 'poison_15', 'poison_50']

    # ── RAW experiments (exp7: GAN-Opt + DeDe) ───────────────────────────────
    print('\n[2] RAW experiments (exp7 GAN-Opt stacking)...')
    exp7_cache = BASE_DIR / 'results/raw/exp7_combined_matrix'
    roc_data_clean, roc_data_gan, roc_data_trig = [], [], []

    for tr in train_scenarios:
        ens = try_load_stacking(create_stacking_ensemble_gan_optimized,
                               X_te_clean_raw.shape[1], exp7_cache, f'ganopt_{tr}')
        if ens is None:
            print(f'  ⚠️  Skip {tr} (cache not found)')
            continue

        for test_X, test_y, test_name in [
            (X_te_clean_raw, y_te_clean,  'clean'),
            (X_te_gan,       y_te_gan,    'gan_attack'),
            (X_mix,          y_mix,       'trigger_10'),
        ]:
            proba = hybrid_predict_proba(dede_raw, ens, thr_raw, test_X)
            pred  = (proba >= 0.5).astype(int)
            m = compute_metrics(test_y, pred, proba)
            m.update({'experiment': 'exp7_RAW_GAN-Opt',
                      'feature_space': 'RAW', 'train_scenario': tr, 'test_type': test_name})
            all_results.append(m)
            print(f'  [{tr}/{test_name}] F1={m["f1_score"]:.4f}  AUC={m["roc_auc"]:.4f}  MCC={m["mcc"]:.4f}')

            # Collect ROC for clean model
            if tr == 'clean':
                fpr, tpr, _ = roc_curve(test_y, proba)
                if test_name   == 'clean':      roc_data_clean.append((f'exp7 RAW', fpr, tpr, m['roc_auc']))
                elif test_name == 'gan_attack': roc_data_gan.append((f'exp7 RAW', fpr, tpr, m['roc_auc']))
                elif test_name == 'trigger_10': roc_data_trig.append((f'exp7 RAW', fpr, tpr, m['roc_auc']))

    # ── LATENT experiments (exp7 latent: GAN-Opt + DeDe) ─────────────────────
    if dual_enc is not None:
        print('\n[3] LATENT experiments (exp7 latent GAN-Opt stacking)...')
        exp7_lat_cache = BASE_DIR / 'results/latent/exp7_combined_matrix_latent'

        for tr in train_scenarios:
            lat_dir_tr = lat_dir / ('exp1_baseline_latent' if tr == 'clean'
                                    else f'exp2_poisoning/{tr}')
            if not (lat_dir_tr / 'X_train.npy').exists():
                continue

            ens = try_load_stacking(create_stacking_ensemble_gan_optimized,
                                   64, exp7_lat_cache, f'ganopt_lat_{tr}')
            if ens is None:
                print(f'  ⚠️  Skip {tr} (latent cache not found)')
                continue

            for test_X_raw, test_y, test_name in [
                (X_te_clean_raw, y_te_clean,  'clean'),
                (X_te_gan,       y_te_gan,    'gan_attack'),
                (X_mix,          y_mix,       'trigger_10'),
            ]:
                proba = hybrid_predict_proba(dede_raw, ens, thr_raw, test_X_raw, dual_enc)
                pred  = (proba >= 0.5).astype(int)
                m = compute_metrics(test_y, pred, proba)
                m.update({'experiment': 'exp7_LAT_GAN-Opt',
                          'feature_space': 'Latent', 'train_scenario': tr, 'test_type': test_name})
                all_results.append(m)
                print(f'  [{tr}/{test_name}] F1={m["f1_score"]:.4f}  AUC={m["roc_auc"]:.4f}  MCC={m["mcc"]:.4f}')

                if tr == 'clean':
                    fpr, tpr, _ = roc_curve(test_y, proba)
                    if test_name   == 'clean':      roc_data_clean.append((f'exp7 LAT', fpr, tpr, m['roc_auc']))
                    elif test_name == 'gan_attack': roc_data_gan.append((f'exp7 LAT', fpr, tpr, m['roc_auc']))
                    elif test_name == 'trigger_10': roc_data_trig.append((f'exp7 LAT', fpr, tpr, m['roc_auc']))

    # ── EXP8 RAW (Standard Stacking + DeDe) ────────────────────────────────
    print('\n[4] RAW experiments (exp8 Standard stacking)...')
    exp8_raw_cache = BASE_DIR / 'results/raw/exp8_standard_stacking'
    for tr in train_scenarios:
        ens = try_load_stacking(create_stacking_ensemble,
                               X_te_clean_raw.shape[1], exp8_raw_cache, f'standard_{tr}')
        if ens is None: continue
        for test_X, test_y, test_name in [
            (X_te_clean_raw, y_te_clean, 'clean'),
            (X_te_gan,       y_te_gan,   'gan_attack'),
            (X_mix,          y_mix,      'trigger_10'),
        ]:
            proba = hybrid_predict_proba(dede_raw, ens, thr_raw, test_X)
            pred  = (proba >= 0.5).astype(int)
            m = compute_metrics(test_y, pred, proba)
            m.update({'experiment': 'exp8_RAW_Standard',
                      'feature_space': 'RAW', 'train_scenario': tr, 'test_type': test_name})
            all_results.append(m)
            print(f'  [{tr}/{test_name}] F1={m["f1_score"]:.4f}  AUC={m["roc_auc"]:.4f}  MCC={m["mcc"]:.4f}')
            if tr == 'clean':
                fpr, tpr, _ = roc_curve(test_y, proba)
                if test_name == 'clean':      roc_data_clean.append(('exp8 RAW', fpr, tpr, m['roc_auc']))
                elif test_name == 'gan_attack': roc_data_gan.append(('exp8 RAW', fpr, tpr, m['roc_auc']))
                elif test_name == 'trigger_10': roc_data_trig.append(('exp8 RAW', fpr, tpr, m['roc_auc']))

    # ── EXP8 LATENT (Standard Stacking + DeDe + dual-enc) ────────────────────
    if dual_enc is not None:
        print('\n[5] LATENT experiments (exp8 Standard stacking)...')
        exp8_lat_cache = BASE_DIR / 'results/latent/exp8_standard_stacking'
        for tr in train_scenarios:
            ens = try_load_stacking(create_stacking_ensemble,
                                   64, exp8_lat_cache, f'standard_lat_{tr}')
            if ens is None: continue
            for test_X_raw, test_y, test_name in [
                (X_te_clean_raw, y_te_clean, 'clean'),
                (X_te_gan,       y_te_gan,   'gan_attack'),
                (X_mix,          y_mix,      'trigger_10'),
            ]:
                proba = hybrid_predict_proba(dede_raw, ens, thr_raw, test_X_raw, dual_enc)
                pred  = (proba >= 0.5).astype(int)
                m = compute_metrics(test_y, pred, proba)
                m.update({'experiment': 'exp8_LAT_Standard',
                          'feature_space': 'Latent', 'train_scenario': tr, 'test_type': test_name})
                all_results.append(m)
                print(f'  [{tr}/{test_name}] F1={m["f1_score"]:.4f}  AUC={m["roc_auc"]:.4f}  MCC={m["mcc"]:.4f}')
                if tr == 'clean':
                    fpr, tpr, _ = roc_curve(test_y, proba)
                    if test_name == 'clean':      roc_data_clean.append(('exp8 LAT', fpr, tpr, m['roc_auc']))
                    elif test_name == 'gan_attack': roc_data_gan.append(('exp8 LAT', fpr, tpr, m['roc_auc']))
                    elif test_name == 'trigger_10': roc_data_trig.append(('exp8 LAT', fpr, tpr, m['roc_auc']))

    # ── EXP9 LATENT (Two-Path Routing) ───────────────────────────────────────
    if dual_enc is not None:
        print('\n[6] LATENT experiments (exp9 Two-Path Routing)...')
        errs_clean = dede_raw.get_reconstruction_error(X_te_clean_raw)
        low_thr  = float(np.percentile(errs_clean, 75))
        high_thr = float(np.percentile(errs_clean, 99))
        exp7_lat_cache = BASE_DIR / 'results/latent/exp7_combined_matrix_latent'
        exp8_lat_cache = BASE_DIR / 'results/latent/exp8_standard_stacking'
        # Fallback to exp5b
        exp5b_lat_cache = BASE_DIR / 'results/latent/exp5b_stacking_vs_gan'

        for tr in train_scenarios:
            ganopt_label = f'ganopt_lat_{tr}'
            std_label    = f'standard_lat_{tr}'
            std_fb_label = f'lat_standard_{tr}' if tr == 'clean' else None

            ganopt = try_load_stacking(create_stacking_ensemble_gan_optimized,
                                      64, exp7_lat_cache, ganopt_label)
            std    = try_load_stacking(create_stacking_ensemble,
                                      64, exp8_lat_cache, std_label)
            # Fallback exp5b for std
            if std is None and tr == 'clean':
                std = try_load_stacking(create_stacking_ensemble,
                                        64, exp5b_lat_cache, 'lat_standard_clean')
            if ganopt is None or std is None:
                print(f'  ⚠️  Skip {tr} (exp9 missing models)')
                continue

            for test_X_raw, test_y, test_name in [
                (X_te_clean_raw, y_te_clean, 'clean'),
                (X_te_gan,       y_te_gan,   'gan_attack'),
                (X_mix,          y_mix,      'trigger_10'),
            ]:
                proba = routing_predict_proba(dede_raw, std, ganopt,
                                             low_thr, high_thr, test_X_raw, dual_enc)
                pred  = (proba >= 0.5).astype(int)
                m = compute_metrics(test_y, pred, proba)
                m.update({'experiment': 'exp9_LAT_Routing',
                          'feature_space': 'Latent', 'train_scenario': tr, 'test_type': test_name})
                all_results.append(m)
                print(f'  [{tr}/{test_name}] F1={m["f1_score"]:.4f}  AUC={m["roc_auc"]:.4f}  MCC={m["mcc"]:.4f}')
                if tr == 'clean':
                    fpr, tpr, _ = roc_curve(test_y, proba)
                    if test_name == 'clean':      roc_data_clean.append(('exp9 LAT', fpr, tpr, m['roc_auc']))
                    elif test_name == 'gan_attack': roc_data_gan.append(('exp9 LAT', fpr, tpr, m['roc_auc']))
                    elif test_name == 'trigger_10': roc_data_trig.append(('exp9 LAT', fpr, tpr, m['roc_auc']))

    # ── Save results ──────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    csv_path = out_dir / 'auc_results_all.csv'
    df.to_csv(csv_path, index=False)

    # ── Plot ROC curves ───────────────────────────────────────────────────────
    print('\n[4] Plotting ROC curves...')
    if roc_data_clean:
        plot_roc(roc_data_clean, 'ROC Curve — Clean Test', roc_dir / 'roc_clean.png')
    if roc_data_gan:
        plot_roc(roc_data_gan,   'ROC Curve — GAN Attack Test', roc_dir / 'roc_gan.png')
    if roc_data_trig:
        plot_roc(roc_data_trig,  'ROC Curve — Trigger Test', roc_dir / 'roc_trigger.png')

    # ── Summary table ─────────────────────────────────────────────────────────
    print('\n' + '='*80)
    print('✅ ROC-AUC SUMMARY — Clean model on all test types'.center(80))
    print('='*80)
    clean_rows = df[df['train_scenario'] == 'clean']
    print(f'\n  {"Experiment":<22} {"Test Type":<15} {"F1":>7} {"AUC":>7} {"PR-AUC":>8} {"MCC":>7}')
    print('  ' + '-'*65)
    for _, row in clean_rows.iterrows():
        print(f'  {row["experiment"]:<22} {row["test_type"]:<15} '
              f'{row["f1_score"]:>7.4f} {row["roc_auc"]:>7.4f} '
              f'{row["pr_auc"]:>8.4f} {row["mcc"]:>7.4f}')

    print(f'\n📁 {csv_path}')
    print(f'📊 ROC curves: {roc_dir}/')


if __name__ == '__main__':
    main()
