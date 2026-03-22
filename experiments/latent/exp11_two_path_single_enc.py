"""
Experiment 11-5 LATENT: Two-Path Routing — DeDe + SingleEncoder bị nhiễm
=========================================================================

Mirror của Exp10b, nhưng thay DualEncoder bằng SingleEncoder:
  Exp10b : DeDe(POISON) + DualEnc(POISON)  + Stack(poison latent)
  Exp11-5: DeDe(POISON) + SingleEnc(POISON) + Stack(poison latent)

Mục đích: So sánh công bằng DualEncoder vs SingleEncoder trong scenario
  TOÀN BỘ hệ thống bị retrain trên poisoned data (defender bị lừa).

Output format GIỐNG EXP10b:
  train_scenario, test_type, accuracy, precision, recall,
  f1_score, route_trigger, route_ganopt, route_standard, asr, false_positive_rate

train_scenarios:
  clean       ← Exp9 baseline (reuse) — clean DeDe + clean enc
  poison_05   ← FULL SYSTEM POISONED at 5%  (Single AE)
  poison_10   ← FULL SYSTEM POISONED at 10%
  poison_15   ← FULL SYSTEM POISONED at 15%
  poison_50   ← FULL SYSTEM POISONED at 50%

Data:
  Poisoned SingleEnc latent: datasets/splits/3.2_latent_single_enc/exp11_poisoning_penc/
  Poisoned SingleEncoder   : datasets/splits/3.2_latent_single_enc/models/poisoned_enc/
  (Tạo bởi: python pipelines/preprocessing/prepare_exp11_data.py)

Usage:
    python experiments/latent/exp11_two_path_single_enc.py
    python experiments/latent/exp11_two_path_single_enc.py --poison-rates 10 50
"""

import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from experiments.dede_adapted.dede_model import build_dede_model
from models.ensemble.stacking import (
    create_stacking_ensemble,
    create_stacking_ensemble_gan_optimized,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
SINGLE_ENC_DATA = BASE_DIR / "datasets/splits/3.2_latent_single_enc"
RAW_SPLITS      = BASE_DIR / "datasets/splits/3.0_raw_from_latent"
DEDE_CLEAN_DIR  = BASE_DIR / "experiments/dede_adapted/models_raw"
RESULTS_BASE    = BASE_DIR / "results/latent/exp11_two_path_single_enc"
RANDOM_STATE    = 42


# ── SingleEncoder wrapper (mirror của DualEncoder trong Exp10b) ───────────────

class SingleEncoderWrapper:
    """Wrap SingleEncoder để có interface encode() giống DualEncoder."""
    def __init__(self, model_path: Path):
        self.encoder = tf.keras.models.load_model(str(model_path))

    def encode(self, X_raw, batch_size=2048):
        results = []
        for i in range(0, len(X_raw), batch_size):
            b = X_raw[i:i+batch_size].astype(np.float32)
            results.append(self.encoder.predict(b, verbose=0))
        return np.concatenate(results)


# ── TwoPathHybridSingleEnc (giống TwoPathHybridLatent trong Exp10b) ───────────

class TwoPathHybridSingleEnc:
    """
    Routing bởi DeDe (RAW space), classify bởi stacking (SingleEnc latent).
    Giống Exp10b — chỉ khác SingleEncoder thay vì DualEncoder.
    """
    def __init__(self, dede_raw, single_enc, std_stack, ganopt_stack, low_thr, high_thr):
        self.dede_raw     = dede_raw
        self.single_enc   = single_enc
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
            X_lat = self.single_enc.encode(X_raw[ganopt_mask])
            pred[ganopt_mask] = self.ganopt_stack.predict(X_lat)
        if standard_mask.sum() > 0:
            X_lat = self.single_enc.encode(X_raw[standard_mask])
            pred[standard_mask] = self.std_stack.predict(X_lat)

        if return_details:
            return pred, {'trigger_mask': trigger_mask,
                          'ganopt_mask':  ganopt_mask,
                          'standard_mask': standard_mask}
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

    def evaluate_trigger(self, trigger_dir: Path):
        files = ['X_test_malicious_triggered.npy', 'X_test_benign_clean.npy',
                 'X_test_mixed_realistic.npy',     'y_test_mixed_realistic.npy']
        if not all((trigger_dir / f).exists() for f in files):
            return None
        X_mal = np.load(trigger_dir / 'X_test_malicious_triggered.npy')
        X_ben = np.load(trigger_dir / 'X_test_benign_clean.npy')
        X_mix = np.load(trigger_dir / 'X_test_mixed_realistic.npy')
        y_mix = np.load(trigger_dir / 'y_test_mixed_realistic.npy')

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


# ── DeDe helpers (copy từ Exp10b) ─────────────────────────────────────────────

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
    """Reuse DeDe đã train từ Exp10b nếu có, không train lại."""
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


def load_or_train_stacking(ens_fn, input_dim, X_lat, y, cache: Path, label: str):
    if (cache / 'meta_model.pkl').exists():
        ens = ens_fn(input_dim=input_dim)
        ens.meta_model = joblib.load(cache / 'meta_model.pkl')
        for nm in list(ens.base_models.keys()):
            pp = cache / f'{nm}_model.pkl'
            pk = cache / f'{nm}_model.keras'
            if pp.exists():   ens.base_models[nm] = joblib.load(pp)
            elif pk.exists(): ens.base_models[nm] = tf.keras.models.load_model(pk)
        ens.is_fitted = True
        print(f'  ✓ [{label}] loaded from cache')
        return ens
    print(f'  Training [{label}] ({len(X_lat):,} × {input_dim}) ...', flush=True)
    ens = ens_fn(input_dim=input_dim)
    ens.fit(X_lat, y, verbose=False)
    ens.save(cache)
    print(f'  ✓ Saved → {cache}')
    return ens


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Exp11-5 LATENT: Two-Path Routing — Poisoned DeDe + Poisoned SingleEnc'
    )
    parser.add_argument('--poison-rates', nargs='+', type=int, default=[5, 10, 15, 50])
    parser.add_argument('--trigger-rate', default='10')
    parser.add_argument('--low-pct',  type=int, default=75)
    parser.add_argument('--high-pct', type=int, default=99)
    parser.add_argument('--dede-epochs', type=int, default=80)
    parser.add_argument('--reuse-dede', action='store_true', default=True,
                        help='Reuse DeDe từ Exp10b nếu có (tránh train lại)')
    parser.add_argument('--no-reuse-dede', dest='reuse_dede', action='store_false')
    parser.add_argument('--data-dir',   default=str(SINGLE_ENC_DATA))
    parser.add_argument('--raw-dir',    default=str(RAW_SPLITS))
    parser.add_argument('--output-dir', default=str(RESULTS_BASE))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir  = Path(args.raw_dir)
    out_dir  = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / 'cache'

    if not data_dir.exists():
        print(f'\n❌ Exp11 data not found: {data_dir}')
        print('   Run: python pipelines/preprocessing/prepare_exp11_data.py')
        import sys; sys.exit(1)

    print('\n' + '='*80)
    print(' EXP11-5 LATENT: TWO-PATH ROUTING — POISONED DeDe + SINGLE ENCODER '.center(80, '='))
    print('='*80)
    print(f"""
  Mirror của Exp10b, thay DualEncoder bằng SingleEncoder:
    Exp10b : DeDe(POISON) + DualEnc(POISON)  + Stack(poison latent)
    Exp11-5: DeDe(POISON) + SingleEnc(POISON) + Stack(poison latent)

  Mục đích: So sánh công bằng — cùng DeDe, cùng stacking, chỉ khác encoder.

  Route P{args.low_pct}: Standard | P{args.low_pct}-P{args.high_pct}: GAN-Opt | >P{args.high_pct}: Block
""")

    # ── Load test sets ──────────────────────────────────────────────────
    print('[INIT] Loading test sets...')
    X_te_raw_clean = np.load(raw_dir / 'exp1_baseline/X_test.npy')
    y_te_clean     = np.load(raw_dir / 'exp1_baseline/y_test.npy')
    X_te_raw_gan   = np.load(raw_dir / 'exp3_gan_attack/X_test.npy')
    y_te_gan       = np.load(raw_dir / 'exp3_gan_attack/y_test.npy')
    trigger_dir    = raw_dir / f'exp5_trigger/trigger_{args.trigger_rate}'

    all_results = []
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ── Scenario clean: reuse Exp9 clean results ──────────────────────
    print('\n[STEP 0] Loading Exp9 clean baseline (reuse — same as Exp10b)...')
    exp9_csv = BASE_DIR / 'results/latent/exp9_two_path_routing/exp9_latent_results.csv'
    if exp9_csv.exists():
        df9 = pd.read_csv(exp9_csv)
        clean_rows = df9[df9['train_scenario'] == 'clean'].copy()
        all_results.extend(clean_rows.to_dict('records'))
        print(f'  ✓ Loaded {len(clean_rows)} rows from Exp9 (clean baseline)')
    else:
        print(f'  ⚠️  Exp9 results not found')

    # ── Load clean DeDe threshold reference ───────────────────────────
    dede_clean = lt_c = ht_c = None
    if DEDE_CLEAN_DIR.exists():
        print('\n[INIT] Loading clean DeDe for threshold reference...')
        dede_clean = load_dede(DEDE_CLEAN_DIR)
        errs_c = dede_clean.get_reconstruction_error(X_te_raw_clean)
        lt_c   = float(np.percentile(errs_c, args.low_pct))
        ht_c   = float(np.percentile(errs_c, args.high_pct))
        print(f'  Clean DeDe threshold: low={lt_c:.6f}  high={ht_c:.6f}')

    # ── Exp10b DeDe cache dir (để reuse DeDe đã train) ────────────────
    exp10b_cache = BASE_DIR / 'results/latent/exp10b_two_path/cache'

    # ── Per poison rate ────────────────────────────────────────────────
    for rate in args.poison_rates:
        rs = f'{rate:02d}'

        # Poisoned SingleEnc data (train latent)
        pp_dir = data_dir / f'exp11_poisoning_penc/poison_{rs}'
        if not (pp_dir / 'X_train.npy').exists():
            print(f'\n  ⚠️  Skipping poison_{rs} — exp11_poisoning_penc not found')
            print(f'       Run: python pipelines/preprocessing/prepare_exp11_data.py')
            continue

        # Poisoned SingleEncoder model
        p_enc_path = data_dir / f'models/poisoned_enc/poison_{rs}/single_encoder_poisoned.h5'
        if not p_enc_path.exists():
            print(f'\n  ⚠️  Skipping poison_{rs} — poisoned SingleEncoder not found: {p_enc_path}')
            continue

        # RAW poisoned data (để train DeDe)
        poison_raw = raw_dir / f'exp2_poisoning/poison_{rs}'
        if not (poison_raw / 'X_train.npy').exists():
            print(f'\n  ⚠️  Skipping poison_{rs} — RAW data not found')
            continue

        print('\n' + '='*80)
        print(f'  FULL SYSTEM POISONED — {rate}% (SingleEncoder)'.center(80))
        print('='*80)

        X_tr_p_lat = np.load(pp_dir / 'X_train.npy')
        y_tr_p_lat = np.load(pp_dir / 'y_train.npy')
        print(f'  Poisoned SingleEnc latent train: {X_tr_p_lat.shape}')

        # Load poisoned SingleEncoder
        single_enc_p = SingleEncoderWrapper(p_enc_path)
        print(f'  ✓ SingleEncoder(poison_{rs}) loaded')

        rate_cache = cache_dir / f'poison_{rs}'
        rate_cache.mkdir(parents=True, exist_ok=True)

        # ── [1] DeDe: Reuse từ Exp10b (cùng DeDe = so sánh công bằng) ──
        # Lý do reuse: DeDe train trên RAW data — không liên quan đến encoder
        # → reuse đảm bảo chênh lệch kết quả chỉ do encoder khác nhau
        dede_p = None
        exp10b_dede_dir = exp10b_cache / f'poison_{rs}/dede_poison'

        if args.reuse_dede and exp10b_dede_dir.exists() and \
                (exp10b_dede_dir / 'best_model.weights.h5').exists():
            print(f'\n  [1/3] Reusing DeDe from Exp10b (poison_{rs})...')
            dede_p = load_dede(exp10b_dede_dir)
            # Recalibrate threshold (giống Exp10b)
            X_val_p = np.load(poison_raw / 'X_test.npy')
            errs_p  = dede_p.get_reconstruction_error(X_val_p)
            lt_p = float(np.percentile(errs_p, args.low_pct))
            ht_p = float(np.percentile(errs_p, args.high_pct))
            print(f'  ✓ Reused DeDe(Exp10b)  threshold: low={lt_p:.6f}  high={ht_p:.6f}')
        else:
            print(f'\n  [1/3] Training DeDe on poisoned RAW (poison_{rs})...')
            X_tr_p_raw = np.load(poison_raw / 'X_train.npy')
            X_val_p    = np.load(poison_raw / 'X_test.npy')
            dede_local  = rate_cache / 'dede_poison'
            dede_p = train_dede(X_tr_p_raw, X_val_p, dede_local, epochs=args.dede_epochs)
            errs_p = dede_p.get_reconstruction_error(X_val_p)
            lt_p   = float(np.percentile(errs_p, args.low_pct))
            ht_p   = float(np.percentile(errs_p, args.high_pct))

        print(f'  DeDe(poison) threshold: low={lt_p:.6f}  high={ht_p:.6f}')
        if lt_c is not None:
            print(f'  Shift vs clean: Δlow={lt_p-lt_c:+.6f}  Δhigh={ht_p-ht_c:+.6f}')

        # Routing distribution
        for tname, X_t in [('Clean', X_te_raw_clean), ('GAN', X_te_raw_gan)]:
            errs = dede_p.get_reconstruction_error(X_t)
            print(f'  Route [{tname}]: Std={( errs<lt_p).mean()*100:.1f}%  '
                  f'GAN={(( errs>=lt_p)&( errs<ht_p)).mean()*100:.1f}%  '
                  f'Blk={(errs>=ht_p).mean()*100:.1f}%')

        # ── [2] Train Stacking trên poisoned SingleEnc latent ──────────
        print(f'\n  [2/3] Training Stacking on poisoned SingleEnc latent...')
        lat_dim = X_tr_p_lat.shape[1]

        std_p = load_or_train_stacking(
            create_stacking_ensemble, lat_dim, X_tr_p_lat, y_tr_p_lat,
            rate_cache / 'std_poison', 'std_poison'
        )
        gan_p = load_or_train_stacking(
            create_stacking_ensemble_gan_optimized, lat_dim, X_tr_p_lat, y_tr_p_lat,
            rate_cache / 'gan_poison', 'gan_poison'
        )

        # ── [3] Two-Path: DeDe(poison) + SingleEnc(poison) + Stack ────
        print(f'\n  [3/3] Evaluating Two-Path (DeDe+SingleEnc+Stack, all poisoned)...')
        hds = TwoPathHybridSingleEnc(dede_p, single_enc_p, std_p, gan_p, lt_p, ht_p)

        for X_te, y_te, ttype in [
            (X_te_raw_clean, y_te_clean, 'clean'),
            (X_te_raw_gan,   y_te_gan,   'gan_attack'),
        ]:
            m = hds.evaluate(X_te, y_te)
            all_results.append({
                'train_scenario': f'poison_{rs}', 'test_type': ttype,
                'dede_low_thr': lt_p, 'dede_high_thr': ht_p, **m
            })
            print(f'    [{ttype:15s}] F1={m["f1_score"]:.4f}'
                  f'  (Std={m["route_standard"]:.0f}%'
                  f' GAN={m["route_ganopt"]:.0f}%'
                  f' Blk={m["route_trigger"]:.0f}%)')

        m_t = hds.evaluate_trigger(trigger_dir)
        if m_t:
            all_results.append({
                'train_scenario': f'poison_{rs}',
                'test_type': f'trigger_{args.trigger_rate}',
                'dede_low_thr': lt_p, 'dede_high_thr': ht_p, **m_t
            })
            print(f'    [trigger_{args.trigger_rate:10s}] F1={m_t["f1_score"]:.4f}'
                  f'  ASR={m_t["asr"]:.2f}%  FP={m_t["false_positive_rate"]:.2f}%')

        print(f'\n  ✅ poison_{rs}% done')

    # ── Save ──────────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    csv_path = out_dir / 'exp11_two_path_results.csv'
    df.to_csv(csv_path, index=False)
    with open(out_dir / 'exp11_two_path_config.json', 'w') as f:
        json.dump({'poison_rates': args.poison_rates, 'low_pct': args.low_pct,
                   'high_pct': args.high_pct, 'dede_reused_from_exp10b': args.reuse_dede,
                   'created_at': ts}, f, indent=2)

    # ── Summary + Delta vs Exp10b ──────────────────────────────────────────────
    print('\n' + '='*80)
    print('✅ EXP11-5 — Two-Path (Poisoned DeDe + SingleEnc)'.center(80))
    print('='*80)

    # Load Exp10b để so sánh trực tiếp
    exp10b_csv = BASE_DIR / 'results/latent/exp10b_two_path/exp10b_latent_results.csv'
    df10b = pd.read_csv(exp10b_csv) if exp10b_csv.exists() else None
    if df10b is None:
        print('\n  ⚠️  Exp10b results not found — Delta vs DualEnc sẽ hiển thị N/A')

    header = (f'\n  {"Scenario":<16} {"S11 Clean":>10} {"D10b Clean":>11} {"Δ(S-D)":>8}'
              f'  {"S11 GAN":>9} {"D10b GAN":>9} {"Δ(S-D)":>8}')
    print(header)
    print('  ' + '-'*75)

    all_scenarios = ['clean'] + [f'poison_{rate:02d}' for rate in args.poison_rates]
    for scen in all_scenarios:
        sub = df[df['train_scenario'] == scen] if not df.empty else pd.DataFrame()
        f1_c_s11 = sub[sub['test_type'] == 'clean']['f1_score'].values
        f1_g_s11 = sub[sub['test_type'] == 'gan_attack']['f1_score'].values
        f1_c = f1_c_s11[0] if len(f1_c_s11) else float('nan')
        f1_g = f1_g_s11[0] if len(f1_g_s11) else float('nan')

        f1_c_d10b = f1_g_d10b = float('nan')
        delta_c = delta_g = float('nan')
        if df10b is not None:
            sub10 = df10b[df10b['train_scenario'] == scen]
            rc = sub10[sub10['test_type'] == 'clean']['f1_score'].values
            rg = sub10[sub10['test_type'] == 'gan_attack']['f1_score'].values
            if len(rc): f1_c_d10b = rc[0]; delta_c = f1_c - f1_c_d10b
            if len(rg): f1_g_d10b = rg[0]; delta_g = f1_g - f1_g_d10b

        def fmt(v): return f'{v:.4f}' if not str(v) == 'nan' else '  N/A'
        def fmtd(v): return f'{v:+.4f}' if not str(v) == 'nan' else '   N/A'
        print(f'  {scen:<16} {fmt(f1_c):>10} {fmt(f1_c_d10b):>11} {fmtd(delta_c):>8}'
              f'  {fmt(f1_g):>9} {fmt(f1_g_d10b):>9} {fmtd(delta_g):>8}')

    print()
    print('  Chú thích:')
    print('  . S11   = SingleEncoder + Stacking (Exp11-5) — kết quả mới')
    print('  . D10b  = DualEncoder   + Stacking (Exp10b)  — reference')
    print('  . Δ(S-D) > 0 → SingleEncoder tốt hơn trong scenario này')
    print('  . Δ(S-D) < 0 → DualEncoder   tốt hơn trong scenario này')
    print(f'\n  📁 Results: {csv_path}')
    print(f'  📁 Compare: {exp10b_csv}\n')


if __name__ == '__main__':
    main()
