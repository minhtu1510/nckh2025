"""
Experiment 10 RAW: Poisoned Full-System Retraining
===================================================

Câu hỏi nghiên cứu
-------------------
Khi defender buộc phải train lại TOÀN BỘ hệ thống trên dữ liệu bị nhiễm
(vì không phát hiện ra attack), hiệu năng suy giảm đến đâu so với:
  - Exp2 (chỉ base models bị poison)
  - Exp9 (stacking + DeDe vẫn clean)

Cấu trúc đánh giá (leo thang từng level)
-----------------------------------------
  Level 1 — Base models đơn lẻ (giống Exp2):
    MLP, SVM, RF, KNN — mỗi model retrain riêng trên poisoned data
    → Baseline: poison ảnh hưởng thế nào trước khi có ensemble

  Level 2 — Stacking Ensemble (giống Exp5b):
    Standard Stacking (MLP+SVM+RF+KNN) — retrain trên poisoned
    GAN-Opt Stacking (MLP×2+KNN×2)    — retrain trên poisoned
    → Ensemble có giúp hồi phục so với từng model đơn lẻ không?

  Level 3 — Two-Path Routing + DeDe (giống Exp9):
    DeDe cũng retrain trên poisoned data
    Routing: DeDe(poison) → Standard/GAN-Opt (poison) / Block
    → Khi cả defense layer (DeDe) bị nhiễm, hệ thống còn hoạt động không?

Threat model
------------
  Attacker: Chèn poison_rate% malicious samples với label=benign
  Defender: Không biết bị tấn công → retrain tất cả như bình thường

Usage
-----
    python experiments/exp10_poisoned_retrain_raw.py
    python experiments/exp10_poisoned_retrain_raw.py --poison-rates 5 10 15 50
    python experiments/exp10_poisoned_retrain_raw.py --level 1      # chỉ base models
    python experiments/exp10_poisoned_retrain_raw.py --level 2      # base + stacking
    python experiments/exp10_poisoned_retrain_raw.py --level 3      # tất cả (default)
"""

import sys, json, argparse, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from experiments.dede_adapted.dede_model import build_dede_model
from models.ensemble.stacking import (
    create_stacking_ensemble,
    create_stacking_ensemble_gan_optimized,
)
from models.advanced.mlp import create_mlp_model
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
RAW_SPLITS    = BASE_DIR / "datasets/splits/3.0_raw_from_latent"
DEDE_CLEAN    = BASE_DIR / "experiments/dede_adapted/models_raw"
RESULTS_BASE  = BASE_DIR / "results/raw/exp10_poisoned_retrain"
PYTHON        = "/home/mtu/miniconda3/envs/fl-fedavg/bin/python"
RANDOM_STATE  = 42


# ── METRICS HELPER ────────────────────────────────────────────────────────────

def eval_metrics(y_true, y_pred):
    return {
        'accuracy':    round(float(accuracy_score(y_true,   y_pred)), 6),
        'precision':   round(float(precision_score(y_true,   y_pred, zero_division=0)), 6),
        'recall':      round(float(recall_score(y_true,      y_pred, zero_division=0)), 6),
        # f1_score = binary (giống run_model_evaluation.py của Exp2)
        'f1_score':    round(float(f1_score(y_true,          y_pred,
                               average='binary', zero_division=0)), 6),
        # f1_weighted — bổ sung để so sánh (tốt hơn khi class imbalanced)
        'f1_weighted': round(float(f1_score(y_true,          y_pred,
                               average='weighted', zero_division=0)), 6),
    }


def print_metric_row(label, m_clean, m_gan):
    print(f"  {label:<35}  Clean={m_clean['f1_score']:.4f}  GAN={m_gan['f1_score']:.4f}")


# ── LEVEL 1: BASE MODELS ──────────────────────────────────────────────────────
# Config giống hệt run_model_evaluation.py (Exp2) để so sánh công bằng

def build_base_models(input_dim):
    """
    Config giống run_model_evaluation.py của Exp2:
      MLP: Dense(50)→Dense(1), Simple architecture
      SVM: LinearSVC không Calibrate
      RF:  n_jobs=4 (không -1)
      KNN: n_jobs=4
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    mlp = Sequential([
        Dense(50, input_dim=input_dim, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return {
        'mlp': mlp,
        'svm': LinearSVC(C=1.0, max_iter=3000, random_state=RANDOM_STATE),  # không Calibrate — giống Exp2
        'rf':  RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=4),
        'knn': KNeighborsClassifier(n_neighbors=5, n_jobs=4),
    }


def train_base_models(X_tr, y_tr, input_dim, cache_dir: Path, force=True):
    """
    Train từng base model trên poisoned data.
    Config giống Exp2 (run_model_evaluation.py) để so sánh công bằng.
    force=True: luôn train mới (không cache) vì Exp10 yêu cầu retrain.
    """
    from tensorflow.keras.callbacks import EarlyStopping as ES
    models = {}
    base_models = build_base_models(input_dim)

    for name, model in base_models.items():
        model_path_pkl   = cache_dir / f'{name}_model.pkl'
        model_path_keras = cache_dir / f'{name}_model.keras'

        if not force and (model_path_pkl.exists() or model_path_keras.exists()):
            if model_path_pkl.exists():
                models[name] = joblib.load(model_path_pkl)
            else:
                models[name] = tf.keras.models.load_model(model_path_keras)
            print(f'    ✓ {name}: loaded from cache')
            continue

        print(f'    Training {name} ({len(X_tr):,} samples)...', end=' ', flush=True)
        if name == 'mlp':
            # batch=64, epochs=30 — giống run_model_evaluation.py Exp2
            model.fit(X_tr, y_tr,
                      epochs=30, batch_size=64,
                      validation_split=0.2, verbose=0,
                      callbacks=[ES(monitor='val_loss', patience=8,
                                    restore_best_weights=True, verbose=0)])
            model.save(model_path_keras)
        else:
            model.fit(X_tr, y_tr)
            joblib.dump(model, model_path_pkl)
        print('✓')
        models[name] = model

    return models


def predict_base(model, name, X):
    """Predict cho cả sklearn và keras models."""
    if name == 'mlp':
        proba = model.predict(X, verbose=0)
        return (proba.flatten() >= 0.5).astype(int)
    return model.predict(X)


# ── LEVEL 2: STACKING ─────────────────────────────────────────────────────────

def train_stacking(ens_fn, input_dim, X_tr, y_tr, save_dir: Path, label: str):
    """Luôn train mới."""
    print(f'    Training [{label}] ({len(X_tr):,} × {input_dim}) ...', flush=True)
    ens = ens_fn(input_dim=input_dim)
    ens.fit(X_tr, y_tr, verbose=False)
    ens.save(save_dir / label)
    print(f'    ✓ Saved → {save_dir / label}')
    return ens


# ── LEVEL 3: DeDe + TWO-PATH ROUTING ─────────────────────────────────────────

def load_dede(model_dir: Path):
    with open(model_dir / 'training_config.json') as f:
        cfg = json.load(f)
    model = build_dede_model(
        input_dim=cfg['input_dim'], latent_dim=cfg.get('latent_dim', 64),
        encoder_hidden_dims=[256, 128], decoder_hidden_dims=[128, 256],
        mask_ratio=cfg.get('mask_ratio', 0.5), dropout=0.2,
        learning_rate=cfg.get('learning_rate', 0.001),
    )
    _ = model(tf.zeros((1, cfg['input_dim'])), training=False)
    model.load_weights(str(model_dir / 'best_model.weights.h5'))
    print(f'  ✓ DeDe loaded from {model_dir.name}')
    return model


def train_dede(X_train, X_val, output_dir: Path,
               epochs=80, batch_size=128) -> object:
    """Train DeDe từ đầu trên poisoned data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dim = X_train.shape[1]
    model = build_dede_model(
        input_dim=input_dim, latent_dim=64,
        encoder_hidden_dims=[256, 128], decoder_hidden_dims=[128, 256],
        mask_ratio=0.5, dropout=0.2, learning_rate=0.001,
    )
    _ = model(tf.zeros((1, input_dim)), training=False)
    ckpt = str(output_dir / 'best_model.weights.h5')
    cbs  = [
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=0),
        ModelCheckpoint(ckpt, monitor='val_loss', save_best_only=True,
                        save_weights_only=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                          min_lr=1e-6, verbose=0),
    ]
    print(f'    Training DeDe ({len(X_train):,}×{input_dim}, epochs≤{epochs}) ...', flush=True)
    hist = model.fit(X_train, X_train,
                     validation_data=(X_val, X_val),
                     epochs=epochs, batch_size=batch_size,
                     callbacks=cbs, verbose=0)
    best = min(hist.history['val_loss'])
    print(f'    ✓ DeDe done  best_val_loss={best:.6f}')
    with open(output_dir / 'training_config.json', 'w') as f:
        json.dump({'input_dim': int(input_dim), 'latent_dim': 64, 'mask_ratio': 0.5,
                   'learning_rate': 0.001, 'best_val_loss': float(best),
                   'trained_on': 'poisoned'}, f, indent=2)
    return model


def two_path_predict(dede, std_stack, ganopt_stack, X, low_thr, high_thr):
    errs         = dede.get_reconstruction_error(X)
    n            = len(X)
    pred         = np.zeros(n, dtype=int)
    trigger_mask = errs >= high_thr
    ganopt_mask  = (errs >= low_thr) & (errs < high_thr)
    std_mask     = errs < low_thr
    pred[trigger_mask] = 1
    if ganopt_mask.sum() > 0:
        pred[ganopt_mask]  = ganopt_stack.predict(X[ganopt_mask])
    if std_mask.sum() > 0:
        pred[std_mask]     = std_stack.predict(X[std_mask])
    return pred, errs, {
        'route_trigger':  round(trigger_mask.sum() / n * 100, 2),
        'route_ganopt':   round(ganopt_mask.sum()  / n * 100, 2),
        'route_standard': round(std_mask.sum()     / n * 100, 2),
        'low_thr':        round(float(low_thr), 6),
        'high_thr':       round(float(high_thr), 6),
    }


def eval_trigger(dede, std_stack, ganopt_stack, trigger_dir, low_thr, high_thr):
    tdir = Path(trigger_dir)
    required = ['X_test_malicious_triggered.npy', 'X_test_benign_clean.npy',
                'X_test_mixed_realistic.npy',     'y_test_mixed_realistic.npy']
    if not all((tdir / f).exists() for f in required):
        return {'f1_score': float('nan'), 'asr': float('nan'),
                'false_positive_rate': float('nan')}

    X_mal = np.load(tdir / 'X_test_malicious_triggered.npy')
    X_ben = np.load(tdir / 'X_test_benign_clean.npy')
    X_mix = np.load(tdir / 'X_test_mixed_realistic.npy')
    y_mix = np.load(tdir / 'y_test_mixed_realistic.npy')

    errs_mal = dede.get_reconstruction_error(X_mal)
    blocked  = (errs_mal >= high_thr).sum()
    asr      = (len(X_mal) - blocked) / len(X_mal) * 100
    fp_rate  = (dede.get_reconstruction_error(X_ben) >= high_thr).mean() * 100

    pred, _, route = two_path_predict(dede, std_stack, ganopt_stack, X_mix, low_thr, high_thr)
    return {
        **eval_metrics(y_mix, pred),
        'asr': round(asr, 4),
        'false_positive_rate': round(fp_rate, 2),
        **route,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Exp10 RAW: Poisoned Full-System Retraining'
    )
    parser.add_argument('--poison-rates', nargs='+', type=int, default=[5, 10, 15, 50])
    parser.add_argument('--trigger-rate', default='10')
    parser.add_argument('--level', type=int, default=3, choices=[1, 2, 3],
                        help='Evaluation level: 1=base models, 2=+stacking, 3=+DeDe (default:3)')
    parser.add_argument('--low-pct',  type=int, default=75)
    parser.add_argument('--high-pct', type=int, default=99)
    parser.add_argument('--dede-epochs', type=int, default=80)
    parser.add_argument('--subsample', type=int, default=None,
                        help='Giới hạn số train samples (vd: 50000) để test nhanh. None=dùng tất cả')
    parser.add_argument('--raw-dir',   default=str(RAW_SPLITS))
    parser.add_argument('--output-dir', default=str(RESULTS_BASE))
    args = parser.parse_args()

    out_dir  = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache    = out_dir / 'cache'
    raw_dir  = Path(args.raw_dir)
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')

    print('\n' + '='*80)
    print(' EXPERIMENT 10 RAW: POISONED FULL-SYSTEM RETRAINING '.center(80, '='))
    print('='*80)
    print(f"""
  Level {args.level} evaluation:
    Level 1  Base models đơn lẻ (MLP, SVM, RF, KNN) — retrain trên poisoned
    Level 2  Stacking ensemble (Standard + GAN-Opt)  — retrain trên poisoned
    Level 3  Two-Path Routing + DeDe                 — DeDe cũng retrain trên poisoned
""")

    # ── Load clean test sets (để đánh giá khách quan) ─────────────────
    print('[INIT] Loading clean test sets...')
    X_te_clean = np.load(raw_dir / 'exp1_baseline/X_test.npy')
    y_te_clean = np.load(raw_dir / 'exp1_baseline/y_test.npy')
    X_te_gan   = np.load(raw_dir / 'exp3_gan_attack/X_test.npy')
    y_te_gan   = np.load(raw_dir / 'exp3_gan_attack/y_test.npy')
    trigger_dir = raw_dir / f'exp5_trigger/trigger_{args.trigger_rate}'
    X_tr_clean  = np.load(raw_dir / 'exp1_baseline/X_train.npy')
    y_tr_clean  = np.load(raw_dir / 'exp1_baseline/y_train.npy')

    print(f'  Clean test : {len(X_te_clean):,}')
    print(f'  GAN test   : {len(X_te_gan):,}')

    # ── Load clean DeDe (dùng làm baseline cho comparison) ─────────────
    dede_clean  = None
    lt_c = ht_c = None
    if args.level >= 3 and DEDE_CLEAN.exists():
        print('\n[INIT] Loading clean DeDe ...')
        dede_clean = load_dede(DEDE_CLEAN)
        errs_calib = dede_clean.get_reconstruction_error(X_te_clean)
        lt_c = float(np.percentile(errs_calib, args.low_pct))
        ht_c = float(np.percentile(errs_calib, args.high_pct))
        print(f'  DeDe(clean) thresh: low={lt_c:.6f}  high={ht_c:.6f}')

    all_results = []

    # ═══════════════════════════════════════════════════════════════════
    for rate in args.poison_rates:
        rate_str   = f'{rate:02d}'
        poison_dir = raw_dir / f'exp2_poisoning/poison_{rate_str}'

        if not (poison_dir / 'X_train.npy').exists():
            print(f'\n  ⚠️  Skipping poison_{rate}% — data not found')
            continue

        print('\n' + '='*80)
        print(f'  POISON RATE: {rate}%'.center(80))
        print('='*80)

        X_tr_p = np.load(poison_dir / 'X_train.npy')
        y_tr_p = np.load(poison_dir / 'y_train.npy')
        flips  = (y_tr_p != y_tr_clean).sum()
        print(f'  Train: {len(X_tr_p):,} × {X_tr_p.shape[1]}'
              f'  ({flips:,} labels flipped = {flips/len(y_tr_clean)*100:.1f}%)')

        # Subsample để test nhanh (giữ nguyên tỷ lệ benign/malicious)
        if args.subsample and args.subsample < len(X_tr_p):
            rng_sub = np.random.RandomState(RANDOM_STATE)
            idx_sub = rng_sub.choice(len(X_tr_p), args.subsample, replace=False)
            X_tr_p  = X_tr_p[idx_sub]
            y_tr_p  = y_tr_p[idx_sub]
            print(f'  ⚡ Subsampled to {len(X_tr_p):,} samples (--subsample={args.subsample})')

        input_dim  = X_tr_p.shape[1]
        rate_cache = cache / f'poison_{rate_str}'
        rate_cache.mkdir(parents=True, exist_ok=True)

        # ── LEVEL 1: Base models đơn lẻ ─────────────────────────────────
        print(f'\n  ── LEVEL 1: Base models đơn lẻ (retrain trên poisoned) ──')
        base_dir = rate_cache / 'base_models_poison'
        base_dir.mkdir(exist_ok=True)
        base_models = train_base_models(X_tr_p, y_tr_p, input_dim, base_dir, force=True)

        for name, model in base_models.items():
            pred_clean = predict_base(model, name, X_te_clean)
            pred_gan   = predict_base(model, name, X_te_gan)
            m_c = eval_metrics(y_te_clean, pred_clean)
            m_g = eval_metrics(y_te_gan,   pred_gan)
            print_metric_row(f'  {name.upper()}(poison)', m_c, m_g)
            for m, ttype in [(m_c, 'clean'), (m_g, 'gan_attack')]:
                all_results.append({
                    'poison_rate': rate, 'level': 1,
                    'model': name.upper(),
                    'trained_on': f'poisoned_{rate}pct',
                    'test_type': ttype,
                    **m,
                })

        if args.level < 2:
            continue

        # ── LEVEL 2: Stacking Ensemble ───────────────────────────────────
        print(f'\n  ── LEVEL 2: Stacking Ensemble (retrain trên poisoned) ──')
        stack_dir = rate_cache / 'stacking_poison'

        std_p = train_stacking(create_stacking_ensemble, input_dim,
                               X_tr_p, y_tr_p, stack_dir, 'standard')
        gan_p = train_stacking(create_stacking_ensemble_gan_optimized, input_dim,
                               X_tr_p, y_tr_p, stack_dir, 'gan_opt')

        for ens, ens_name in [(std_p, 'Standard_Stack'), (gan_p, 'GAN-Opt_Stack')]:
            pred_c = ens.predict(X_te_clean)
            pred_g = ens.predict(X_te_gan)
            m_c = eval_metrics(y_te_clean, pred_c)
            m_g = eval_metrics(y_te_gan,   pred_g)
            print_metric_row(f'  {ens_name}(poison)', m_c, m_g)
            for m, ttype in [(m_c, 'clean'), (m_g, 'gan_attack')]:
                all_results.append({
                    'poison_rate': rate, 'level': 2,
                    'model': ens_name,
                    'trained_on': f'poisoned_{rate}pct',
                    'test_type': ttype,
                    **m,
                })

        if args.level < 3:
            continue

        # ── LEVEL 3a: Two-Path Routing với DeDe CLEAN (baseline — như Exp9) ──
        if dede_clean is not None:
            print(f'\n  ── LEVEL 3a: Two-Path Routing + DeDe(CLEAN) ── [ref: Exp9]')
            pred_c, _, route_c = two_path_predict(dede_clean, std_p, gan_p,
                                                   X_te_clean, lt_c, ht_c)
            pred_g, _, route_g = two_path_predict(dede_clean, std_p, gan_p,
                                                   X_te_gan, lt_c, ht_c)
            m_c = {**eval_metrics(y_te_clean, pred_c), **route_c}
            m_g = {**eval_metrics(y_te_gan,   pred_g), **route_g}
            trig = eval_trigger(dede_clean, std_p, gan_p, trigger_dir, lt_c, ht_c)
            print_metric_row('  TwoPath+DeDe(clean)/Stack(poison)', m_c, m_g)
            print(f'    Route: Std={m_c["route_standard"]:.0f}%'
                  f'  GAN={m_c["route_ganopt"]:.0f}%'
                  f'  Blk={m_c["route_trigger"]:.0f}%')
            print(f'    Trigger: F1={trig.get("f1_score","nan"):.4f}'
                  f'  ASR={trig.get("asr", float("nan")):.2f}%')
            for m, ttype in [(m_c, 'clean'), (m_g, 'gan_attack'), (trig, f'trigger_{args.trigger_rate}')]:
                all_results.append({
                    'poison_rate': rate, 'level': '3a_dede_clean',
                    'model': 'TwoPath_DeDe-clean_Stack-poison',
                    'trained_on': f'stack:poisoned_{rate}pct  dede:clean',
                    'test_type': ttype,
                    **m,
                })

        # ── LEVEL 3b: Two-Path Routing với DeDe POISONED (Exp10 core) ──
        print(f'\n  ── LEVEL 3b: Two-Path Routing + DeDe(POISONED) ── [EXP10 CORE]')
        X_val_p  = np.load(poison_dir / 'X_test.npy')
        dede_p_dir = rate_cache / f'dede_poison_{rate_str}'
        dede_p = train_dede(X_tr_p, X_val_p, dede_p_dir, epochs=args.dede_epochs)

        # Calibrate threshold trên poisoned val (defender không biết bị nhiễm)
        errs_p  = dede_p.get_reconstruction_error(X_val_p)
        lt_p    = float(np.percentile(errs_p, args.low_pct))
        ht_p    = float(np.percentile(errs_p, args.high_pct))
        print(f'    DeDe(poison) thresh: low={lt_p:.6f}  high={ht_p:.6f}')

        # So sánh shift của threshold
        if lt_c is not None:
            print(f'    Threshold shift vs clean: Δlow={lt_p - lt_c:+.6f}'
                  f'  Δhigh={ht_p - ht_c:+.6f}')

        pred_c, _, route_c = two_path_predict(dede_p, std_p, gan_p,
                                               X_te_clean, lt_p, ht_p)
        pred_g, _, route_g = two_path_predict(dede_p, std_p, gan_p,
                                               X_te_gan, lt_p, ht_p)
        m_c = {**eval_metrics(y_te_clean, pred_c), **route_c}
        m_g = {**eval_metrics(y_te_gan,   pred_g), **route_g}
        trig = eval_trigger(dede_p, std_p, gan_p, trigger_dir, lt_p, ht_p)

        print_metric_row('  TwoPath+DeDe(poison)/Stack(poison)', m_c, m_g)
        print(f'    Route: Std={m_c["route_standard"]:.0f}%'
              f'  GAN={m_c["route_ganopt"]:.0f}%'
              f'  Blk={m_c["route_trigger"]:.0f}%')
        print(f'    Trigger: F1={trig.get("f1_score","nan"):.4f}'
              f'  ASR={trig.get("asr", float("nan")):.2f}%')

        for m, ttype in [(m_c, 'clean'), (m_g, 'gan_attack'), (trig, f'trigger_{args.trigger_rate}')]:
            all_results.append({
                'poison_rate': rate, 'level': '3b_dede_poison',
                'model': 'TwoPath_DeDe-poison_Stack-poison',
                'trained_on': f'poisoned_{rate}pct',
                'test_type': ttype,
                'dede_low_thr': lt_p,
                'dede_high_thr': ht_p,
                **m,
            })

        print(f'\n  ✅ Completed poison rate {rate}%')

    # ── Save all results ───────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    csv_path = out_dir / f'exp10_raw_results_{ts}.csv'
    df.to_csv(csv_path, index=False)
    df.to_csv(out_dir / 'exp10_raw_results_latest.csv', index=False)

    # ── Print summary table ────────────────────────────────────────────
    print('\n' + '='*80)
    print('✅ EXP10 RAW — SUMMARY'.center(80))
    print('='*80)

    # Pivot: model × poison_rate → F1 (clean test)
    for test_t in ['clean', 'gan_attack']:
        print(f'\n📊 F1 on [{test_t}] test:\n')
        rate_cols = sorted(df['poison_rate'].dropna().unique().astype(int))
        header = f"  {'Model':<45}" + ''.join(f' P{r:02d}' for r in rate_cols)
        print(header)
        print('  ' + '-' * (45 + 4 * len(rate_cols)))

        models_ordered = []
        # Level 1: base models
        for nm in ['MLP', 'SVM', 'RF', 'KNN']:
            models_ordered.append((nm, '1', f'{nm}(retrain)'))
        # Level 2: stacking
        models_ordered.append(('Standard_Stack', '2', 'Standard Stack(retrain)'))
        models_ordered.append(('GAN-Opt_Stack',  '2', 'GAN-Opt Stack(retrain)'))
        # Level 3
        models_ordered.append(('TwoPath_DeDe-clean_Stack-poison', '3a_dede_clean',
                                'TwoPath+DeDe(clean)+Stack(poison)'))
        models_ordered.append(('TwoPath_DeDe-poison_Stack-poison', '3b_dede_poison',
                                'TwoPath+DeDe(poison)+Stack(poison)'))

        for model_key, level_key, display in models_ordered:
            row = f'  {display:<45}'
            for rate in rate_cols:
                sub = df[(df['model'] == model_key) &
                         (df['test_type'] == test_t) &
                         (df['poison_rate'] == rate)]
                val = sub['f1_score'].values[0] if len(sub) else float('nan')
                row += f' {val:.4f}' if not np.isnan(val) else '    --'
            print(row)

    print(f'\n📁 Results: {csv_path}')
    print(f'📁 Latest:  {out_dir}/exp10_raw_results_latest.csv\n')


if __name__ == '__main__':
    main()
