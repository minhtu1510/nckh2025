"""
Experiment 10 LATENT: Poisoned Full-System Retraining
======================================================

Câu hỏi nghiên cứu
------------------
Khi TOÀN BỘ hệ thống (DeDe + DualEncoder + Stacking) bị retrain trên
poisoned data, hiệu năng suy giảm đến đâu?

So sánh 3 mức độ (giống Exp10 RAW nhưng trong latent space):
  Level 1 — Base models đơn lẻ trên LATENT bị nhiễm (MLP,SVM,RF,KNN)
  Level 2 — Stacking Ensemble:
      Scenario A: DeDe(clean) + DualEnc(clean)  + Stack(clean)   ← baseline Exp9 Latent
      Scenario B: DeDe(clean) + DualEnc(clean)  + Stack(poison)  ← chỉ classifier bị nhiễm
      Scenario C: DeDe(poison)+ DualEnc(poison) + Stack(poison)  ← FULL system (Exp10 core)
  Level 3 — Two-Path Routing (như Level 2 nhưng với routing metrics + trigger eval)

Điểm then chốt:
  Stacking [C] train trên latent từ POISONED encoder
  Khi predict: X_raw → POISONED DualEncoder → latent bị nhiễm → stacking(poison)
  → Đây là scenario đúng khi defender deploy hệ thống đã bị nhiễm

Output format: giống Exp8/Exp9 để dễ so sánh
  train_scenario, test_type, accuracy, precision, recall, f1_score,
  route_trigger, route_ganopt, route_standard, asr, false_positive_rate

Usage
-----
    python experiments/latent/exp10_poisoned_retrain_latent.py
    python experiments/latent/exp10_poisoned_retrain_latent.py --level 1  # base models only
    python experiments/latent/exp10_poisoned_retrain_latent.py --level 2  # + stacking
    python experiments/latent/exp10_poisoned_retrain_latent.py --level 3  # + routing (default)
    python experiments/latent/exp10_poisoned_retrain_latent.py --poison-rates 10 --level 1
"""

import sys, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from experiments.dede_adapted.dede_model import build_dede_model
from models.ensemble.stacking import (
    create_stacking_ensemble,
    create_stacking_ensemble_gan_optimized,
)
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# ── Constants ─────────────────────────────────────────────────────────────────
LATENT_EXP10   = BASE_DIR / "datasets/splits/exp10_latent"   # output của prepare_exp10_data.py
LATENT_CLEAN   = BASE_DIR / "datasets/splits/3.1_latent"     # baseline clean latent
RAW_SPLITS     = BASE_DIR / "datasets/splits/3.0_raw_from_latent"
DEDE_CLEAN_DIR = BASE_DIR / "experiments/dede_adapted/models_raw"
DUAL_ENC_CLEAN = BASE_DIR / "datasets/splits/3.1_latent/models"
RESULTS_BASE   = BASE_DIR / "results/latent/exp10_poisoned_retrain"
RANDOM_STATE   = 42


# ── DualEncoder ───────────────────────────────────────────────────────────────

class DualEncoder:
    """Wrapper cho cặp benign + malicious encoder."""
    def __init__(self, models_dir: Path):
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


# ── TwoPathHybridLatent — giống Exp9 ─────────────────────────────────────────

class TwoPathHybridLatent:
    """
    Routing bởi DeDe error (RAW space), classify bởi stacking (latent space).
    Giống class trong Exp9 để đảm bảo nhất quán.
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
            'accuracy':       round(float(accuracy_score(y, pred)), 6),
            'precision':      round(float(precision_score(y, pred, zero_division=0)), 6),
            'recall':         round(float(recall_score(y, pred, zero_division=0)), 6),
            'f1_score':       round(float(f1_score(y, pred, zero_division=0)), 6),
            'route_trigger':  round(det['trigger_mask'].sum() / n * 100, 2),
            'route_ganopt':   round(det['ganopt_mask'].sum()  / n * 100, 2),
            'route_standard': round(det['standard_mask'].sum()/ n * 100, 2),
            'asr':            None,
            'false_positive_rate': None,
        }

    def evaluate_trigger(self, trigger_dir):
        tdir  = Path(trigger_dir)
        files = ['X_test_malicious_triggered.npy', 'X_test_benign_clean.npy',
                 'X_test_mixed_realistic.npy',     'y_test_mixed_realistic.npy']
        if not all((tdir / f).exists() for f in files):
            print(f'  ⚠️  Trigger data not found: {tdir}')
            return None

        X_mal = np.load(tdir / 'X_test_malicious_triggered.npy')
        X_ben = np.load(tdir / 'X_test_benign_clean.npy')
        X_mix = np.load(tdir / 'X_test_mixed_realistic.npy')
        y_mix = np.load(tdir / 'y_test_mixed_realistic.npy')

        errs_mal = self.dede_raw.get_reconstruction_error(X_mal)
        blocked  = (errs_mal >= self.high_thr).sum()
        asr      = (len(X_mal) - blocked) / len(X_mal) * 100
        fp_rate  = (self.dede_raw.get_reconstruction_error(X_ben) >= self.high_thr).mean() * 100

        pred_mix, det = self.predict(X_mix, return_details=True)
        n = len(X_mix)
        return {
            'accuracy':       round(float(accuracy_score(y_mix,  pred_mix)), 6),
            'precision':      round(float(precision_score(y_mix, pred_mix, zero_division=0)), 6),
            'recall':         round(float(recall_score(y_mix,    pred_mix, zero_division=0)), 6),
            'f1_score':       round(float(f1_score(y_mix,         pred_mix, zero_division=0)), 6),
            'route_trigger':  round(det['trigger_mask'].sum() / n * 100, 2),
            'route_ganopt':   round(det['ganopt_mask'].sum()  / n * 100, 2),
            'route_standard': round(det['standard_mask'].sum()/ n * 100, 2),
            'asr':            round(asr, 4),
            'false_positive_rate': round(fp_rate, 2),
        }


# ── DeDe helpers ──────────────────────────────────────────────────────────────

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


def train_dede(X_train, X_val, output_dir: Path, epochs=80, batch_size=128):
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
    print(f'    Training DeDe ({len(X_train):,}×{input_dim}, ≤{epochs} epochs)...', flush=True)
    hist = model.fit(X_train, X_train,
                     validation_data=(X_val, X_val),
                     epochs=epochs, batch_size=batch_size,
                     callbacks=cbs, verbose=0)
    best = min(hist.history['val_loss'])
    print(f'    ✓ DeDe done — best_val_loss={best:.6f}')
    with open(output_dir / 'training_config.json', 'w') as f:
        json.dump({'input_dim': int(input_dim), 'latent_dim': 64, 'mask_ratio': 0.5,
                   'learning_rate': 0.001, 'best_val_loss': float(best),
                   'trained_on': 'poisoned'}, f, indent=2)
    return model


# ── Stacking helpers ──────────────────────────────────────────────────────────

def load_or_train_stack(ens_fn, input_dim, X_lat, y, cache_dir: Path, label: str,
                        force=False):
    """Load từ cache nếu có, ngược lại train mới."""
    cache = cache_dir / label
    if not force and (cache / 'meta_model.pkl').exists():
        ens = ens_fn(input_dim=input_dim)
        ens.meta_model = joblib.load(cache / 'meta_model.pkl')
        for nm in list(ens.base_models.keys()):
            pp = cache / f'{nm}_model.pkl'
            pk = cache / f'{nm}_model.keras'
            if pp.exists():   ens.base_models[nm] = joblib.load(pp)
            elif pk.exists(): ens.base_models[nm] = tf.keras.models.load_model(pk)
        ens.is_fitted = True
        print(f'  ✓ Stack cache: {label}')
        return ens
    print(f'  Training stack [{label}] ({len(X_lat):,} × {input_dim})...', flush=True)
    ens = ens_fn(input_dim=input_dim)
    ens.fit(X_lat, y, verbose=False)
    ens.save(cache)
    print(f'  ✓ Saved → {cache}')
    return ens


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Exp10 LATENT: Poisoned Full-System Retraining'
    )
    parser.add_argument('--poison-rates', nargs='+', type=int, default=[5, 10, 15, 50])
    parser.add_argument('--trigger-rate', default='10')
    parser.add_argument('--low-pct',  type=int, default=75)
    parser.add_argument('--high-pct', type=int, default=99)
    parser.add_argument('--dede-epochs', type=int, default=80)
    parser.add_argument('--level', type=int, default=3, choices=[1, 2, 3],
                        help='1=base models, 2=+stacking, 3=+routing (default:3)')
    parser.add_argument('--skip-scenario-a', action='store_true',
                        help='Bỏ qua Scenario A (tiết kiệm thời gian)')
    parser.add_argument('--skip-scenario-b', action='store_true',
                        help='Bỏ qua Scenario B')
    parser.add_argument('--subsample', type=int, default=None,
                        help='Giới hạn train samples để test nhanh')
    parser.add_argument('--latent-dir',    default=str(LATENT_EXP10))
    parser.add_argument('--clean-lat-dir', default=str(LATENT_CLEAN))
    parser.add_argument('--raw-dir',       default=str(RAW_SPLITS))
    parser.add_argument('--output-dir',    default=str(RESULTS_BASE))
    args = parser.parse_args()

    out_dir   = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / 'cache'
    lat_dir   = Path(args.latent_dir)
    cln_lat   = Path(args.clean_lat_dir)
    raw_dir   = Path(args.raw_dir)

    if not lat_dir.exists():
        print(f'\n❌ Exp10 latent data not found: {lat_dir}')
        print('   Hãy chạy trước: python pipelines/preprocessing/prepare_exp10_data.py')
        sys.exit(1)

    print('\n' + '='*80)
    print(' EXPERIMENT 10 LATENT: POISONED FULL-SYSTEM RETRAINING '.center(80, '='))
    print('='*80)
    print(f"""
  Level {args.level} — poison rates: {args.poison_rates}
  Output format: giống Exp8/Exp9 (train_scenario × test_type)

  Level 1: Base models đơn lẻ trên latent bị nhiễm
  Level 2: Stacking (Scenarios A/B/C)
  Level 3: Two-Path Routing (Scenarios A/B/C) + trigger eval
""")

    # ── Load clean test sets (RAW) ─────────────────────────────────── 
    print('[INIT] Loading RAW test sets...')
    X_te_raw_clean = np.load(raw_dir / 'exp1_baseline/X_test.npy')
    y_te_clean     = np.load(raw_dir / 'exp1_baseline/y_test.npy')
    X_te_raw_gan   = np.load(raw_dir / 'exp3_gan_attack/X_test.npy')
    y_te_gan       = np.load(raw_dir / 'exp3_gan_attack/y_test.npy')
    trigger_dir    = raw_dir / f'exp5_trigger/trigger_{args.trigger_rate}'
    X_tr_clean_raw = np.load(raw_dir / 'exp1_baseline/X_train.npy')
    y_tr_clean     = np.load(raw_dir / 'exp1_baseline/y_train.npy')
    print(f'  Clean test : {len(X_te_raw_clean):,}  |  GAN test: {len(X_te_raw_gan):,}')

    # ── Load clean DeDe + DualEncoder ─────────────────────────────────
    dede_clean = dual_enc_clean = None
    lt_c = ht_c = None
    if DEDE_CLEAN_DIR.exists():
        print('\n[INIT] Loading clean DeDe...')
        dede_clean = load_dede(DEDE_CLEAN_DIR)
        errs_c = dede_clean.get_reconstruction_error(X_te_raw_clean)
        lt_c   = float(np.percentile(errs_c, args.low_pct))
        ht_c   = float(np.percentile(errs_c, args.high_pct))
        print(f'  Clean DeDe threshold: low={lt_c:.6f}  high={ht_c:.6f}')

    if DUAL_ENC_CLEAN.exists():
        print('\n[INIT] Loading clean DualEncoder...')
        dual_enc_clean = DualEncoder(DUAL_ENC_CLEAN)
        print('  ✓ DualEncoder(clean) loaded')

    # ── Clean latent test set (encode bởi clean encoder) ──────────────
    # Dùng cho Level 1 (test trên poisoned latent dùng poisoned enc)
    # và làm reference
    X_te_lat_clean_ref = X_te_lat_gan_ref = None

    all_results = []
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ═══════════════════════════════════════════════════════════════════
    for rate in args.poison_rates:
        rate_str   = f'{rate:02d}'
        poison_raw = raw_dir / f'exp2_poisoning/poison_{rate_str}'
        poison_lat = lat_dir / f'poison_{rate_str}'   # output của prepare_exp10_data.py

        if not (poison_raw / 'X_train.npy').exists():
            print(f'\n  ⚠️  Skipping poison_{rate}% — RAW data not found')
            continue

        print('\n' + '='*80)
        print(f'  POISON RATE: {rate}%'.center(80))
        print('='*80)

        X_tr_p_raw = np.load(poison_raw / 'X_train.npy')
        y_tr_p     = np.load(poison_raw / 'y_train.npy')
        X_val_p    = np.load(poison_raw / 'X_test.npy')

        flips = (y_tr_p != y_tr_clean).sum()
        print(f'  Poisoned train: {len(X_tr_p_raw):,} × {X_tr_p_raw.shape[1]}'
              f'  ({flips:,} flips = {flips/len(y_tr_clean)*100:.1f}%)')

        if args.subsample and args.subsample < len(X_tr_p_raw):
            rng_s = np.random.RandomState(RANDOM_STATE)
            idx_s = rng_s.choice(len(X_tr_p_raw), args.subsample, replace=False)
            X_tr_p_raw = X_tr_p_raw[idx_s]
            y_tr_p     = y_tr_p[idx_s]
            print(f'  ⚡ Subsampled to {len(X_tr_p_raw):,}')

        rate_cache = cache_dir / f'poison_{rate_str}'
        rate_cache.mkdir(parents=True, exist_ok=True)

        # ── Load poisoned latent (train + test) ────────────────────────
        # prepare_exp10_data.py đã encode cả train lẫn test bởi poisoned enc
        if (poison_lat / 'X_train.npy').exists():
            X_tr_p_lat = np.load(poison_lat / 'X_train.npy')
            # X_test cũng encode bởi poisoned enc (realistic)
            X_te_p_lat_clean = np.load(poison_lat / 'X_test.npy')
            # Reference: test encode bởi clean enc (nếu có)
            if (poison_lat / 'X_test_cleanenc.npy').exists():
                X_te_p_lat_cleanref = np.load(poison_lat / 'X_test_cleanenc.npy')
            else:
                X_te_p_lat_cleanref = None
            print(f'  Poisoned latent train: {X_tr_p_lat.shape}'
                  f'  |  test(poison enc): {X_te_p_lat_clean.shape}')
        else:
            X_tr_p_lat = X_te_p_lat_clean = X_te_p_lat_cleanref = None
            print(f'  ⚠️  Poisoned latent not found at {poison_lat}')
            print(f'     Run: python pipelines/preprocessing/prepare_exp10_data.py')

        # ── Load poisoned DualEncoder ──────────────────────────────────
        enc_p_dir = lat_dir / 'encoders' / f'poison_{rate_str}'
        if enc_p_dir.exists():
            dual_enc_p = DualEncoder(enc_p_dir)
            print(f'  ✓ DualEncoder(poison) loaded from {enc_p_dir.name}')
        else:
            dual_enc_p = None
            print(f'  ⚠️  Poisoned encoder not found at {enc_p_dir}')

        # ──────────────────────────────────────────────────────────────
        # LEVEL 1: Base models đơn lẻ trên latent bị nhiễm
        # ──────────────────────────────────────────────────────────────
        if X_tr_p_lat is not None:
            print(f'\n  ── LEVEL 1: Base models đơn lẻ (latent bị nhiễm) ──')
            print(f'     Config giống Exp2 Latent — train on poisoned latent (poisoned enc)')
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense
            from tensorflow.keras.callbacks import EarlyStopping as ES

            lat_dim = X_tr_p_lat.shape[1]
            mlp_l1 = Sequential([
                Dense(50, input_dim=lat_dim, activation='relu'),
                Dense(1,  activation='sigmoid'),
            ])
            mlp_l1.compile(loss='binary_crossentropy', optimizer='adam',
                           metrics=['accuracy'])
            base_cfgs = {
                'mlp': ('keras',   mlp_l1),
                'svm': ('sklearn', LinearSVC(C=1.0, max_iter=3000,
                                             random_state=RANDOM_STATE)),
                'rf':  ('sklearn', RandomForestClassifier(n_estimators=100,
                                    random_state=RANDOM_STATE, n_jobs=4)),
                'knn': ('sklearn', KNeighborsClassifier(n_neighbors=5, n_jobs=4)),
            }

            # Test latent: dùng poisoned enc (realistic)
            X_te_lat_for_l1 = X_te_p_lat_clean  # encoded by poisoned enc

            for name, (mtype, model) in base_cfgs.items():
                print(f'    {name} ({len(X_tr_p_lat):,} × {lat_dim})...', end=' ', flush=True)
                if mtype == 'keras':
                    model.fit(X_tr_p_lat, y_tr_p, epochs=30, batch_size=64,
                              validation_split=0.2, verbose=0,
                              callbacks=[ES(monitor='val_loss', patience=8,
                                           restore_best_weights=True, verbose=0)])
                    pred = (model.predict(X_te_lat_for_l1, verbose=0).flatten() >= 0.5).astype(int)
                else:
                    model.fit(X_tr_p_lat, y_tr_p)
                    pred = model.predict(X_te_lat_for_l1)
                print('✓')

                m = {
                    'accuracy':   round(float(accuracy_score(y_te_clean,  pred)), 6),
                    'precision':  round(float(precision_score(y_te_clean,  pred, zero_division=0)), 6),
                    'recall':     round(float(recall_score(y_te_clean,     pred, zero_division=0)), 6),
                    'f1_score':   round(float(f1_score(y_te_clean,          pred, zero_division=0)), 6),
                    'route_trigger': None, 'route_ganopt': None, 'route_standard': None,
                    'asr': None, 'false_positive_rate': None,
                }
                # GAN test
                if mtype == 'keras' and dual_enc_p:
                    X_te_gan_lat = dual_enc_p.encode(X_te_raw_gan)
                    pred_g = (model.predict(X_te_gan_lat, verbose=0).flatten() >= 0.5).astype(int)
                elif mtype == 'sklearn' and dual_enc_p:
                    X_te_gan_lat = dual_enc_p.encode(X_te_raw_gan)
                    pred_g = model.predict(X_te_gan_lat)
                else:
                    pred_g = None

                f1_g = float(f1_score(y_te_gan, pred_g, zero_division=0)) if pred_g is not None else float('nan')
                print(f'    {name.upper()}(poison latent)  Clean={m["f1_score"]:.4f}  GAN={f1_g:.4f}')

                all_results.append({
                    'train_scenario': f'L1_{name}_poison{rate_str}',
                    'test_type': 'clean', **m
                })
                if pred_g is not None:
                    m_g = {**m, 'f1_score': round(f1_g, 6),
                           'accuracy':  round(float(accuracy_score(y_te_gan, pred_g)), 6),
                           'precision': round(float(precision_score(y_te_gan, pred_g, zero_division=0)), 6),
                           'recall':    round(float(recall_score(y_te_gan, pred_g, zero_division=0)), 6)}
                    all_results.append({
                        'train_scenario': f'L1_{name}_poison{rate_str}',
                        'test_type': 'gan_attack', **m_g
                    })

        if args.level < 2:
            continue

        # ──────────────────────────────────────────────────────────────
        # LEVEL 2+3: Stacking Ensemble + Two-Path Routing
        # ──────────────────────────────────────────────────────────────

        # ── Scenario A: DeDe(clean) + DualEnc(clean) + Stack(clean) ──
        # Giống Exp9 Latent hoàn toàn — dùng làm baseline
        if not args.skip_scenario_a and dede_clean and dual_enc_clean:
            scen = f'A_clean_poison{rate_str}'
            print(f'\n  ── Scenario A: DeDe(clean)+DualEnc(clean)+Stack(clean) ── [Exp9 ref]')

            # Load latent train clean
            clean_lat_tr_dir = cln_lat / 'exp1_baseline_latent'
            if (clean_lat_tr_dir / 'X_train.npy').exists():
                X_tr_c_lat = np.load(clean_lat_tr_dir / 'X_train.npy')
                y_tr_c     = np.load(clean_lat_tr_dir / 'y_train.npy')
            elif (lat_dir / 'baseline/X_train.npy').exists():
                X_tr_c_lat = np.load(lat_dir / 'baseline/X_train.npy')
                y_tr_c     = np.load(lat_dir / 'baseline/y_train.npy')
            else:
                X_tr_c_lat = dual_enc_clean.encode(X_tr_clean_raw)
                y_tr_c     = y_tr_clean

            lat_dim_c = X_tr_c_lat.shape[1]
            # Try reuse Exp9 cache
            exp9_std = BASE_DIR / 'results/latent/exp8_standard_stacking' / 'standard_lat_clean'
            exp9_gan = BASE_DIR / 'results/latent/exp7_combined_matrix_latent' / 'ganopt_lat_clean'

            std_a = load_or_train_stack(create_stacking_ensemble, lat_dim_c,
                                        X_tr_c_lat, y_tr_c,
                                        exp9_std.parent if (exp9_std / 'meta_model.pkl').exists()
                                        else rate_cache / 'scen_a',
                                        exp9_std.name if (exp9_std / 'meta_model.pkl').exists()
                                        else 'std_clean')
            gan_a = load_or_train_stack(create_stacking_ensemble_gan_optimized, lat_dim_c,
                                        X_tr_c_lat, y_tr_c,
                                        exp9_gan.parent if (exp9_gan / 'meta_model.pkl').exists()
                                        else rate_cache / 'scen_a',
                                        exp9_gan.name if (exp9_gan / 'meta_model.pkl').exists()
                                        else 'gan_clean')

            hds_a = TwoPathHybridLatent(dede_clean, dual_enc_clean, std_a, gan_a, lt_c, ht_c)

            m_c = hds_a.evaluate(X_te_raw_clean, y_te_clean)
            m_g = hds_a.evaluate(X_te_raw_gan, y_te_gan)
            m_t = hds_a.evaluate_trigger(trigger_dir) if args.level >= 3 else None

            for m, ttype in [(m_c, 'clean'), (m_g, 'gan_attack')]:
                all_results.append({'train_scenario': scen, 'test_type': ttype, **m})
                print(f'    [{ttype:15s}] F1={m["f1_score"]:.4f}'
                      f'  (Std={m["route_standard"]:.0f}%'
                      f' GAN={m["route_ganopt"]:.0f}%'
                      f' Blk={m["route_trigger"]:.0f}%)')
            if m_t:
                all_results.append({'train_scenario': scen,
                                    'test_type': f'trigger_{args.trigger_rate}', **m_t})
                print(f'    [trigger_{args.trigger_rate:10s}] F1={m_t["f1_score"]:.4f}'
                      f'  ASR={m_t["asr"]:.2f}%')

        # ── Scenario B: DeDe(clean) + DualEnc(clean) + Stack(poison) ──
        # Encode poisoned X_train bằng CLEAN encoder → stack train trên latent có label nhiễm
        if not args.skip_scenario_b and dede_clean and dual_enc_clean:
            scen = f'B_stack_poison{rate_str}'
            print(f'\n  ── Scenario B: DeDe(clean)+DualEnc(clean)+Stack(poison) ──')
            print(f'     Stack train: encode X_train_RAW_poisoned bằng CLEAN encoder → poisoned label')

            X_tr_b_lat = dual_enc_clean.encode(X_tr_p_raw)
            lat_dim_b  = X_tr_b_lat.shape[1]

            std_b = load_or_train_stack(create_stacking_ensemble, lat_dim_b,
                                        X_tr_b_lat, y_tr_p,
                                        rate_cache / 'scen_b', 'std_clean_enc_poison_label',
                                        force=True)
            gan_b = load_or_train_stack(create_stacking_ensemble_gan_optimized, lat_dim_b,
                                        X_tr_b_lat, y_tr_p,
                                        rate_cache / 'scen_b', 'gan_clean_enc_poison_label',
                                        force=True)

            hds_b = TwoPathHybridLatent(dede_clean, dual_enc_clean, std_b, gan_b, lt_c, ht_c)

            m_c = hds_b.evaluate(X_te_raw_clean, y_te_clean)
            m_g = hds_b.evaluate(X_te_raw_gan, y_te_gan)
            m_t = hds_b.evaluate_trigger(trigger_dir) if args.level >= 3 else None

            for m, ttype in [(m_c, 'clean'), (m_g, 'gan_attack')]:
                all_results.append({'train_scenario': scen, 'test_type': ttype, **m})
                print(f'    [{ttype:15s}] F1={m["f1_score"]:.4f}'
                      f'  (Std={m["route_standard"]:.0f}%'
                      f' GAN={m["route_ganopt"]:.0f}%'
                      f' Blk={m["route_trigger"]:.0f}%)')
            if m_t:
                all_results.append({'train_scenario': scen,
                                    'test_type': f'trigger_{args.trigger_rate}', **m_t})
                print(f'    [trigger_{args.trigger_rate:10s}] F1={m_t["f1_score"]:.4f}'
                      f'  ASR={m_t["asr"]:.2f}%')

        # ── Scenario C: DeDe(poison) + DualEnc(poison) + Stack(poison) ──
        # FULL SYSTEM POISONED — Exp10 core
        print(f'\n  ── Scenario C: DeDe(poison)+DualEnc(poison)+Stack(poison) ── [EXP10 CORE]')

        # C1: Retrain DeDe trên RAW poisoned
        dede_p_dir = rate_cache / f'dede_poison_{rate_str}'
        dede_p = train_dede(X_tr_p_raw, X_val_p, dede_p_dir, epochs=args.dede_epochs)
        errs_p = dede_p.get_reconstruction_error(X_val_p)
        lt_p   = float(np.percentile(errs_p, args.low_pct))
        ht_p   = float(np.percentile(errs_p, args.high_pct))
        print(f'    DeDe(poison) thresh: low={lt_p:.6f}  high={ht_p:.6f}')
        if lt_c is not None:
            print(f'    Shift vs clean: Δlow={lt_p-lt_c:+.6f}  Δhigh={ht_p-ht_c:+.6f}')

        # C2: Load poisoned DualEncoder + poisoned latent
        # Stack train trên X_tr_p_lat đã encode bởi poisoned encoder
        # Khi predict: X_raw → poisoned enc → latent → stack
        if X_tr_p_lat is not None and dual_enc_p is not None:
            lat_dim_c = X_tr_p_lat.shape[1]
            std_c = load_or_train_stack(create_stacking_ensemble, lat_dim_c,
                                        X_tr_p_lat, y_tr_p,
                                        rate_cache / 'scen_c', 'std_poison_enc_poison_label',
                                        force=True)
            gan_c = load_or_train_stack(create_stacking_ensemble_gan_optimized, lat_dim_c,
                                        X_tr_p_lat, y_tr_p,
                                        rate_cache / 'scen_c', 'gan_poison_enc_poison_label',
                                        force=True)

            # Predict: X_raw → DeDe(poison) routing → DualEnc(poison) → stack(poison)
            hds_c = TwoPathHybridLatent(dede_p, dual_enc_p, std_c, gan_c, lt_p, ht_p)

            scen = f'C_full_poison{rate_str}'
            m_c = hds_c.evaluate(X_te_raw_clean, y_te_clean)
            m_g = hds_c.evaluate(X_te_raw_gan, y_te_gan)
            m_t = hds_c.evaluate_trigger(trigger_dir) if args.level >= 3 else None

            for m, ttype in [(m_c, 'clean'), (m_g, 'gan_attack')]:
                all_results.append({'train_scenario': scen, 'test_type': ttype,
                                    'dede_low_thr': lt_p, 'dede_high_thr': ht_p, **m})
                print(f'    [{ttype:15s}] F1={m["f1_score"]:.4f}'
                      f'  (Std={m["route_standard"]:.0f}%'
                      f' GAN={m["route_ganopt"]:.0f}%'
                      f' Blk={m["route_trigger"]:.0f}%)')
            if m_t:
                all_results.append({'train_scenario': scen,
                                    'test_type': f'trigger_{args.trigger_rate}',
                                    'dede_low_thr': lt_p, 'dede_high_thr': ht_p, **m_t})
                print(f'    [trigger_{args.trigger_rate:10s}] F1={m_t["f1_score"]:.4f}'
                      f'  ASR={m_t["asr"]:.2f}%')
        else:
            print(f'  ⚠️  Skipping Scenario C — no poisoned latent/encoder available')
            print(f'     Run: python pipelines/preprocessing/prepare_exp10_data.py')

        print(f'\n  ✅ Completed poison rate {rate}%')

    # ── Save results — format giống Exp9 ─────────────────────────────────
    df = pd.DataFrame(all_results)
    csv_latest = out_dir / 'exp10_latent_results.csv'       # giống exp9_latent_results.csv
    csv_ts     = out_dir / f'exp10_latent_results_{ts}.csv'
    df.to_csv(csv_latest, index=False)
    df.to_csv(csv_ts,     index=False)
    with open(out_dir / 'exp10_latent_config.json', 'w') as f:
        json.dump({'poison_rates': args.poison_rates, 'level': args.level,
                   'low_pct': args.low_pct, 'high_pct': args.high_pct,
                   'dede_epochs': args.dede_epochs, 'created_at': ts}, f, indent=2)

    # ── Summary table — format giống Exp8/Exp9 ────────────────────────
    print('\n' + '='*80)
    print('✅ EXP10 LATENT — SUMMARY'.center(80))
    print('='*80)

    # Lấy các scenarios để in
    scenarios = df['train_scenario'].unique().tolist()
    for test_t in ['clean', 'gan_attack', f'trigger_{args.trigger_rate}']:
        sub = df[df['test_type'] == test_t]
        if sub.empty:
            continue
        print(f'\n📊 [{test_t}]  F1-score per scenario:\n')
        print(f"  {'Scenario':<40}  {'F1':>8}  {'RouteStd':>9}  {'RouteGAN':>9}  {'Blk':>5}  {'ASR':>7}")
        print('  ' + '-'*82)
        for scen in scenarios:
            row = sub[sub['train_scenario'] == scen]
            if row.empty:
                continue
            r = row.iloc[0]
            f1  = r.get('f1_score',  float('nan'))
            std = r.get('route_standard', None)
            gan = r.get('route_ganopt',   None)
            blk = r.get('route_trigger',  None)
            asr = r.get('asr',            None)
            print(f"  {scen:<40}  {f1:>8.4f}"
                  f"  {f'{std:.0f}%' if std is not None else '':>9}"
                  f"  {f'{gan:.0f}%' if gan is not None else '':>9}"
                  f"  {f'{blk:.0f}%' if blk is not None else '':>5}"
                  f"  {f'{asr:.2f}%' if asr is not None else '':>7}")

    # So sánh A vs B vs C (cùng poison rate)
    print(f'\n📊 Impact của poisoning — F1 degradation (clean test):\n')
    print(f"  {'Poison':>8}  {'A(baseline)':>13}  {'B(stack↓)':>13}  {'C(full↓)':>13}  "
          f"{'Δ(A→B)':>8}  {'Δ(B→C)':>8}")
    print('  ' + '-'*75)
    for rate in args.poison_rates:
        rs = f'{rate:02d}'
        rowA = df[(df['train_scenario'] == f'A_clean_poison{rs}') & (df['test_type'] == 'clean')]
        rowB = df[(df['train_scenario'] == f'B_stack_poison{rs}') & (df['test_type'] == 'clean')]
        rowC = df[(df['train_scenario'] == f'C_full_poison{rs}')  & (df['test_type'] == 'clean')]
        fA = rowA.iloc[0]['f1_score'] if not rowA.empty else float('nan')
        fB = rowB.iloc[0]['f1_score'] if not rowB.empty else float('nan')
        fC = rowC.iloc[0]['f1_score'] if not rowC.empty else float('nan')
        dAB = fB - fA if not (np.isnan(fA) or np.isnan(fB)) else float('nan')
        dBC = fC - fB if not (np.isnan(fB) or np.isnan(fC)) else float('nan')
        print(f'  P{rs}      {fA:>13.4f}  {fB:>13.4f}  {fC:>13.4f}  '
              f'{dAB:>+8.4f}  {dBC:>+8.4f}')

    print(f'\n📁 Results: {csv_latest}')
    print(f'📁 Timestamped: {csv_ts}\n')


if __name__ == '__main__':
    main()
