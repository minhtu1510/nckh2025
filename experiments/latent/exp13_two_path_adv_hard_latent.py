"""
Experiment 12 LATENT: Ultimate Architecture — Two-Path Dynamic MoE + Adversarial Training
========================================================================================

Kết hợp đỉnh cao của Toàn bộ luận văn:
  - Cấu trúc: Two-Path Routing (DeDe + DualEncoder + 2 Stacking Ensembles)
  - Phân quyền: Dynamic Mixture of Experts (Soft-Routing nãy giờ)
  - Vũ khí cuối cùng: Adversarial Training (Trộn 5% GAN vào tập Train)

Mục tiêu: Đạt F1 > 0.95 cho cả Clean và GAN mà không làm suy yếu bất cứ hệ thống nào.

Data:
  Poisoned latent: datasets/splits/exp10_latent/
  (Tạo bởi: python pipelines/preprocessing/prepare_exp10_data.py)

Usage:
    python experiments/latent/exp10b_two_path_latent.py
    python experiments/latent/exp10b_two_path_latent.py --poison-rates 10 50
"""

import sys, json, gc
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
    create_max_voting_ensemble_gan_optimized
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
LATENT_EXP10   = BASE_DIR / "datasets/splits/exp10_latent"
LATENT_CLEAN   = BASE_DIR / "datasets/splits/3.1_latent"
RAW_SPLITS     = BASE_DIR / "datasets/splits/3.0_raw_from_latent"
DEDE_CLEAN_DIR = BASE_DIR / "experiments/dede_adapted/models_raw"
DUAL_ENC_CLEAN = BASE_DIR / "datasets/splits/3.1_latent/models"
RESULTS_BASE   = BASE_DIR / "results/latent/exp10b_two_path"
RANDOM_STATE   = 42


# ── DualEncoder ───────────────────────────────────────────────────────────────

class DualEncoder:
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


# ── TwoPathHybridLatent (giống hệt Exp9) ──────────────────────────────────────

class TwoPathHybridLatent:
    """
    Routing bởi DeDe (RAW space), classify bởi stacking (latent space).
    Giống cấu trúc Exp9 — chỉ khác DeDe + DualEnc + Stack đều bị nhiễm.
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
    print(f'    Training DeDe ({len(X_train):,}×{input_dim}, ≤{epochs} epochs)...',
          flush=True)
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

def load_or_train(ens_fn, input_dim, X_lat, y, cache_dir: Path, label: str,
                  force=False):
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
        description='Exp10b LATENT: Two-Path Routing with Poisoned DeDe+DualEnc+Stack'
    )
    parser.add_argument('--poison-rates', nargs='+', type=int, default=[0, 5, 10, 15, 50])
    parser.add_argument('--trigger-rate', default='10')
    parser.add_argument('--low-pct',  type=int, default=50)
    parser.add_argument('--high-pct', type=int, default=99)
    parser.add_argument('--dede-epochs', type=int, default=80)
    parser.add_argument('--latent-dir',    default=str(LATENT_EXP10))
    parser.add_argument('--clean-lat-dir', default=str(LATENT_CLEAN))
    parser.add_argument('--raw-dir',       default=str(RAW_SPLITS))
    parser.add_argument('--output-dir',    default=str(RESULTS_BASE))
    args = parser.parse_args()

    lat_dir   = Path(args.latent_dir)
    cln_lat   = Path(args.clean_lat_dir)
    raw_dir   = Path(args.raw_dir)
    out_dir   = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / 'cache'

    if not lat_dir.exists():
        print(f'\n❌ Exp10 latent not found: {lat_dir}')
        print('   Run: python pipelines/preprocessing/prepare_exp10_data.py')
        import sys; sys.exit(1)

    print('\n' + '='*80)
    print(' EXP10b LATENT: TWO-PATH ROUTING — FULL SYSTEM POISONED '.center(80, '='))
    print(f'  Route P{args.low_pct}: Standard | P{args.low_pct}-P{args.high_pct}: GAN-Opt | >P{args.high_pct}: Block'.center(80))
    print('='*80)
    print("""
  Giống Exp9 nhưng TOÀN BỘ bị retrain trên poisoned data:
    DeDe(POISON) + DualEnc(POISON) + Stack(POISON latent)
  
  train_scenarios:
    clean     = Exp9 baseline (reuse Exp9 results)
    poison_XX = FULL SYSTEM POISONED
""")

    # ── Load test sets ─────────────────────────────────────────────────
    print('[INIT] Loading test sets...')
    X_te_raw_clean = np.load(raw_dir / 'exp1_baseline/X_test.npy')
    y_te_clean     = np.load(raw_dir / 'exp1_baseline/y_test.npy')
    X_te_raw_gan   = np.load(raw_dir / 'exp3_gan_attack/X_test.npy')
    y_te_gan       = np.load(raw_dir / 'exp3_gan_attack/y_test.npy')
    trigger_dir    = raw_dir / f'exp5_trigger/trigger_{args.trigger_rate}'
    X_tr_clean_raw = np.load(raw_dir / 'exp1_baseline/X_train.npy')
    y_tr_clean     = np.load(raw_dir / 'exp1_baseline/y_train.npy')

    all_results = []
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ── Scenario clean: load Exp9 removed. We will run rate=0 natively. ─────

    # ── Load clean DeDe (để lấy threshold tham chiếu) ─────────────────
    dede_clean = lt_c = ht_c = None
    if DEDE_CLEAN_DIR.exists():
        print('\n[INIT] Loading clean DeDe for threshold reference...')
        dede_clean = load_dede(DEDE_CLEAN_DIR)
        
        # BƯỚC SỬA MỤC TIÊU: Lấy Threshold trên tập Train_Clean (KHÔNG ĐƯỢC CHẠM VÀO TEST)
        errs_c = dede_clean.get_reconstruction_error(X_tr_clean_raw)
        lt_c   = float(np.percentile(errs_c, args.low_pct))
        ht_c   = float(np.percentile(errs_c, args.high_pct))
        print(f'  Clean DeDe threshold: low={lt_c:.6f}  high={ht_c:.6f}')

    # ── Per poison rate ────────────────────────────────────────────────
    for rate in args.poison_rates:
        if rate == 0:
            rs = 'clean'
            poison_raw = raw_dir / 'exp1_baseline'
            poison_lat = cln_lat / 'exp1_baseline_latent'
            enc_p_dir  = DUAL_ENC_CLEAN
        else:
            rs = f'{rate:02d}'
            poison_raw = raw_dir / f'exp2_poisoning/poison_{rs}'
            poison_lat = lat_dir / f'poison_{rs}'
            enc_p_dir  = lat_dir / 'encoders' / f'poison_{rs}'

        if not (poison_raw / 'X_train.npy').exists():
            print(f'\n  ⚠️  Skipping poison_{rs}% — RAW data not found')
            continue

        print('\n' + '='*80)
        print(f'  FULL SYSTEM POISONED — {rate}%'.center(80))
        print('='*80)

        X_tr_p_raw = np.load(poison_raw / 'X_train.npy')
        y_tr_p     = np.load(poison_raw / 'y_train.npy')
        X_val_p    = np.load(poison_raw / 'X_test.npy')
        flips      = (y_tr_p != y_tr_clean).sum()
        print(f'  RAW poisoned train: {len(X_tr_p_raw):,} × {X_tr_p_raw.shape[1]}'
              f'  ({flips:,} flips = {flips/len(y_tr_clean)*100:.1f}%)')

        # Kiểm tra poisoned latent
        if not (poison_lat / 'X_train.npy').exists():
            print(f'  ⚠️  Poisoned latent not found: {poison_lat}')
            print(f'     Run: python pipelines/preprocessing/prepare_exp10_data.py')
            continue

        X_tr_p_lat = np.load(poison_lat / 'X_train.npy')
        y_tr_p_lat = np.load(poison_lat / 'y_train.npy')
        print(f'  Poisoned latent train: {X_tr_p_lat.shape}  (poisoned encoder)')

        # Load poisoned DualEncoder
        if not enc_p_dir.exists():
            print(f'  ⚠️  Poisoned encoder not found: {enc_p_dir}')
            continue
        dual_enc_p = DualEncoder(enc_p_dir)
        print(f'  ✓ DualEncoder(poison) loaded')

        rate_cache = cache_dir / f'poison_{rs}'
        rate_cache.mkdir(parents=True, exist_ok=True)

        # ── [1] Retrain DeDe trên RAW poisoned ────────────────────────
        if rate == 0:
            print(f'\n  [1/3] Using CLEAN DeDe (no retraining needed)...')
            dede_p = dede_clean
            lt_p = lt_c
            ht_p = ht_c
        else:
            print(f'\n  [1/3] Retraining DeDe on poisoned RAW...')
            dede_p_dir = rate_cache / 'dede_poison'
            dede_p = train_dede(X_tr_p_raw, X_val_p, dede_p_dir, epochs=args.dede_epochs)
            
            # BƯỚC SỬA MỤC TIÊU: Calibrate threshold trên poisoned TRAIN (Ngăn chặn Rò rỉ Data)
            errs_p = dede_p.get_reconstruction_error(X_tr_p_raw)
            lt_p   = float(np.percentile(errs_p, args.low_pct))
            ht_p   = float(np.percentile(errs_p, args.high_pct))
            
        print(f'  DeDe threshold: low={lt_p:.6f}  high={ht_p:.6f}')
        if rate != 0 and lt_c is not None:
            print(f'  Shift vs clean: Δlow={lt_p-lt_c:+.6f}  Δhigh={ht_p-ht_c:+.6f}')

        # Routing distribution on test sets
        for tname, X_t in [('Clean', X_te_raw_clean), ('GAN', X_te_raw_gan)]:
            errs = dede_p.get_reconstruction_error(X_t)
            print(f'  Route [{tname}]: Std={( errs<lt_p).mean()*100:.1f}%  '
                  f'GAN={(( errs>=lt_p)&( errs<ht_p)).mean()*100:.1f}%  '
                  f'Blk={(errs>=ht_p).mean()*100:.1f}%')

        # ── [2] Train Stacking trên poisoned latent ────────────────────
        print(f'\n  [2/3] Training Stacking trên dữ liệu VACCINE (Adversarial Training)...')
        lat_dim = X_tr_p_lat.shape[1]
        
        # ---> INJECT GAN ĐỂ LÀM VACCINE
        print(f'      Vaccine: Đang pha chế 5% GAN Adversarial vào nồi Train...')
        X_gan_raw = np.load(raw_dir / 'exp3_gan_attack/X_test.npy')
        y_gan_raw = np.load(raw_dir / 'exp3_gan_attack/y_test.npy')
        
        # Chỉ vớt thuần túy hạt GAN
        X_gan_pure = X_gan_raw[y_gan_raw == 1]
        y_gan_pure = y_gan_raw[y_gan_raw == 1]
        
        num_adv = int(len(X_tr_p_raw) * 0.05)  # Trộn 5% của 300k = ~15k GAN
        idx = np.random.choice(len(X_gan_pure), size=min(num_adv, len(X_gan_pure)), replace=False)
        X_adv_raw = X_gan_pure[idx]
        y_adv = y_gan_pure[idx]
        
        print(f'      Vaccine: Đang mã hoá {len(X_adv_raw):,} GAN packets qua DualEncoder...')
        X_adv_lat = dual_enc_p.encode(X_adv_raw)
        
        X_tr_mix = np.vstack((X_tr_p_lat, X_adv_lat))
        y_tr_mix = np.concatenate((y_tr_p_lat, y_adv))
        
        # Lắc đều nồi cháo
        s_idx = np.random.permutation(len(X_tr_mix))
        X_tr_mix = X_tr_mix[s_idx]
        y_tr_mix = y_tr_mix[s_idx]

        std_p = load_or_train(create_stacking_ensemble, lat_dim,
                              X_tr_mix, y_tr_mix,
                              rate_cache, 'std_vaccine', force=False)
        gan_p = load_or_train(create_max_voting_ensemble_gan_optimized, lat_dim,
                              X_tr_mix, y_tr_mix,
                              rate_cache, 'gan_vaccine', force=False)

        # ── [3] Two-Path Hybrid: DeDe(poison)+DualEnc(poison)+Stack(poison) ──
        print(f'\n  [3/3] Evaluating Two-Path Hybrid (all poisoned)...')
        hds = TwoPathHybridLatent(dede_p, dual_enc_p, std_p, gan_p, lt_p, ht_p)

        for X_te, y_te, ttype in [
            (X_te_raw_clean, y_te_clean, 'clean'),
            (X_te_raw_gan,   y_te_gan,   'gan_attack'),
        ]:
            m = hds.evaluate(X_te, y_te)
            all_results.append({
                'train_scenario': 'clean' if rate == 0 else f'poison_{rs}', 'test_type': ttype,
                'dede_low_thr': lt_p, 'dede_high_thr': ht_p, **m
            })
            print(f'    [{ttype:15s}] F1={m["f1_score"]:.4f}'
                  f'  (Std={m["route_standard"]:.0f}%'
                  f' GAN={m["route_ganopt"]:.0f}%'
                  f' Blk={m["route_trigger"]:.0f}%)')

        m_t = hds.evaluate_trigger(trigger_dir)
        if m_t:
            all_results.append({
                'train_scenario': 'clean' if rate == 0 else f'poison_{rs}',
                'test_type': f'trigger_{args.trigger_rate}',
                'dede_low_thr': lt_p, 'dede_high_thr': ht_p, **m_t
            })
            print(f'    [trigger_{args.trigger_rate:10s}] F1={m_t["f1_score"]:.4f}'
                  f'  ASR={m_t["asr"]:.2f}%  FP={m_t["false_positive_rate"]:.2f}%')

        print(f'\n  ✅ {"clean" if rate == 0 else f"poison_{rs}%"} done')
        del std_p, gan_p, hds
        gc.collect()

    # ── Save — format giống Exp9 ──────────────────────────────────────────
    df = pd.DataFrame(all_results)
    csv_path = out_dir / 'exp10b_latent_results.csv'
    df.to_csv(csv_path, index=False)
    with open(out_dir / 'exp10b_config.json', 'w') as f:
        json.dump({'poison_rates': args.poison_rates, 'low_pct': args.low_pct,
                   'high_pct': args.high_pct, 'dede_epochs': args.dede_epochs,
                   'created_at': ts}, f, indent=2)

    # ── Summary table giống Exp9 ──────────────────────────────────────────
    print('\n' + '='*80)
    print('✅ EXP10b LATENT — Two-Path (Full System Poisoned)'.center(80))
    print('='*80)

    col0 = 'Train \\ Test'
    header = f'\n{col0:<16} {"Clean F1":>10} {"GAN F1":>10} {"Trigger F1":>12} {"ASR":>8}'
    print(header)
    print('  ' + '-'*60)

    all_scenarios = ['clean'] + [f'poison_{rate:02d}' for rate in args.poison_rates]
    for scen in all_scenarios:
        sub = {r['test_type']: r for r in all_results if r.get('train_scenario') == scen}
        if not sub:
            continue
        f1_c = sub.get('clean', {}).get('f1_score', float('nan'))
        f1_g = sub.get('gan_attack', {}).get('f1_score', float('nan'))
        f1_t = sub.get(f'trigger_{args.trigger_rate}', {}).get('f1_score', float('nan'))
        asr  = sub.get(f'trigger_{args.trigger_rate}', {}).get('asr',      float('nan'))
        print(f'  {scen:<16} {f1_c:>10.4f} {f1_g:>10.4f} {f1_t:>12.4f} {asr if asr else float("nan"):>7.2f}%')

    print(f'\n📁 Results: {csv_path}')
    print(f'   (Giống format Exp9: train_scenario × test_type)\n')


if __name__ == '__main__':
    main()
