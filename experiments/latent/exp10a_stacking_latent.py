"""
Experiment 10a LATENT: Stacking trên Poisoned Latent (Poisoned Encoder)
=======================================================================

Tương tự Exp8 (standard stacking) nhưng thay latent clean → latent từ POISONED encoder.
Chứng minh: Khi defender retrain DualEncoder trên poisoned data,
stacking ensemble có còn phòng thủ tốt không?

So sánh trực tiếp:
  Exp8  : train_scenario × train latent (CLEAN encoder)   → stacking
  Exp10a: train_scenario × train latent (POISONED encoder) → stacking

Output format GIỐNG EXP8:
  train_scenario, test_type, accuracy, precision, recall,
  f1_score, stage1_pct, asr, false_positive_rate

train_scenarios:
  clean       ← latent từ CLEAN encoder (= Exp8 baseline, reuse cache)
  poison_05   ← latent từ POISONED encoder 5%
  poison_10   ← latent từ POISONED encoder 10%
  poison_15   ← latent từ POISONED encoder 15%
  poison_50   ← latent từ POISONED encoder 50%

Data:
  Poisoned latent: datasets/splits/exp10_latent/poison_XX/
  (Tạo bởi: python pipelines/preprocessing/prepare_exp10_data.py)

Usage:
    python experiments/latent/exp10a_stacking_latent.py
    python experiments/latent/exp10a_stacking_latent.py --poison-rates 10 50
"""

import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from models.ensemble.stacking import (
    create_stacking_ensemble,
    create_stacking_ensemble_gan_optimized,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
LATENT_EXP10 = BASE_DIR / "datasets/splits/exp10_latent"
LATENT_CLEAN = BASE_DIR / "datasets/splits/3.1_latent"
RAW_SPLITS   = BASE_DIR / "datasets/splits/3.0_raw_from_latent"
RESULTS_BASE = BASE_DIR / "results/latent/exp10a_stacking"
RANDOM_STATE = 42


# ── Stacking helpers (giống Exp8) ─────────────────────────────────────────────

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


def eval_metrics(y_true, y_pred):
    return {
        'accuracy':  round(float(accuracy_score(y_true,  y_pred)), 6),
        'precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        'recall':    round(float(recall_score(y_true,    y_pred, zero_division=0)), 6),
        'f1_score':  round(float(f1_score(y_true,         y_pred, zero_division=0)), 6),
    }


def eval_trigger(ens, trigger_dir: Path):
    """Evaluate trigger attack (giống Exp8)."""
    files = ['X_test_malicious_triggered.npy', 'X_test_benign_clean.npy',
             'X_test_mixed_realistic.npy', 'y_test_mixed_realistic.npy']
    if not all((trigger_dir / f).exists() for f in files):
        return None
    # NOTE: Exp8 sử dụng latent để classify, không raw
    # → Trigger eval cần encode X qua encoder
    # → Return None (trigger eval chỉ có nghĩa khi có DeDe routing)
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Exp10a LATENT: Stacking on Poisoned Latent (Poisoned Encoder)'
    )
    parser.add_argument('--poison-rates', nargs='+', type=int, default=[5, 10, 15, 50])
    parser.add_argument('--trigger-rate', default='10')
    parser.add_argument('--latent-dir',    default=str(LATENT_EXP10))
    parser.add_argument('--clean-lat-dir', default=str(LATENT_CLEAN))
    parser.add_argument('--raw-dir',       default=str(RAW_SPLITS))
    parser.add_argument('--output-dir',    default=str(RESULTS_BASE))
    args = parser.parse_args()

    lat_dir  = Path(args.latent_dir)
    cln_lat  = Path(args.clean_lat_dir)
    raw_dir  = Path(args.raw_dir)
    out_dir  = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / 'cache'

    if not lat_dir.exists():
        print(f'\n❌ Exp10 latent not found: {lat_dir}')
        print('   Run: python pipelines/preprocessing/prepare_exp10_data.py')
        import sys; sys.exit(1)

    print('\n' + '='*80)
    print(' EXP10a LATENT: STACKING ON POISONED LATENT '.center(80, '='))
    print('='*80)
    print(f"""
  Giống Exp8 nhưng dùng POISONED encoder thay vì clean encoder.
  Câu hỏi: Stacking có còn hiệu quả khi latent space bị nhiễm?

  train_scenarios: clean (Exp8 ref) + poison_05/10/15/50 (poisoned enc)
  test: luôn encode bởi CÙNG encoder với train (realistic)
""")

    # Load clean test RAW (để encode khi cần)
    print('[INIT] Loading test sets...')
    X_te_raw_clean = np.load(raw_dir / 'exp1_baseline/X_test.npy')
    y_te_clean     = np.load(raw_dir / 'exp1_baseline/y_test.npy')
    X_te_raw_gan   = np.load(raw_dir / 'exp3_gan_attack/X_test.npy')
    y_te_gan       = np.load(raw_dir / 'exp3_gan_attack/y_test.npy')
    trigger_dir    = raw_dir / f'exp5_trigger/trigger_{args.trigger_rate}'

    all_results = []
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ── Định nghĩa train scenarios giống Exp8 ──────────────────────────────
    # clean = dùng latent từ CLEAN encoder (reuse Exp8 cache)
    # poison_XX = dùng latent từ POISONED encoder (mới trong Exp10a)

    scenarios = [
        # (label, lat_train_dir, lat_test_clean, lat_test_gan, is_clean_enc)
        ('clean', cln_lat / 'exp1_baseline_latent', None, None, True),
    ]
    for rate in args.poison_rates:
        rs = f'{rate:02d}'
        p_dir = lat_dir / f'poison_{rs}'
        if (p_dir / 'X_train.npy').exists():
            scenarios.append((f'poison_{rs}', p_dir, None, None, False))
        else:
            print(f'  ⚠️  poison_{rs} not found, skipping')

    print(f'\n[TRAIN] {len(scenarios)} scenarios: {[s[0] for s in scenarios]}\n')

    for label, train_dir, _, __, is_clean in scenarios:
        if not (train_dir / 'X_train.npy').exists():
            print(f'\n  ⚠️  Skip [{label}] — data not found at {train_dir}')
            continue

        print('\n' + '-'*60)
        print(f'  ► [{label}]  ({"clean" if is_clean else "poisoned"} encoder)')

        X_tr = np.load(train_dir / 'X_train.npy')
        y_tr = np.load(train_dir / 'y_train.npy')
        # Test: dùng X_test.npy từ cùng thư mục (encode bởi cùng encoder)
        X_te_c = np.load(train_dir / 'X_test.npy')    # test_clean đã encode
        y_te_c_local = np.load(train_dir / 'y_test.npy')

        # Với clean scenario: test GAN cần encode bởi clean encoder
        # Với poison scenario: test GAN cần encode bởi poisoned encoder
        # (dùng X_test.npy của poison dir — files được prepare_exp10_data.py tạo)
        if is_clean:
            # Tìm clean test gan latent
            gan_lat_path = cln_lat / 'exp3_gan_attack_latent/X_test.npy'
            if gan_lat_path.exists():
                X_te_g = np.load(gan_lat_path)
            else:
                X_te_g = None
        else:
            # poisoned: dùng poisoned encoder cho test GAN cũng như test clean
            # prepare_exp10_data chỉ tạo X_test (clean) → encode GAN cần encoder
            enc_dir = lat_dir / 'encoders' / f'poison_{label.split("_")[1]}'
            if enc_dir.exists():
                benc = tf.keras.models.load_model(str(enc_dir / 'benign_encoder.h5'))
                menc = tf.keras.models.load_model(str(enc_dir / 'malicious_encoder.h5'))
                print(f'    Encoding GAN test with poisoned encoder...')
                X_te_g = np.hstack([
                    benc.predict(X_te_raw_gan.astype(np.float32), verbose=0),
                    menc.predict(X_te_raw_gan.astype(np.float32), verbose=0),
                ])
            else:
                X_te_g = None
                print(f'    ⚠️  Poisoned encoder not found, skipping GAN test')

        print(f'    Train: {X_tr.shape}  Test-clean: {X_te_c.shape}')
        input_dim = X_tr.shape[1]

        # Cache: poison scenarios always retrain (force=True)
        force = not is_clean

        # Try reuse Exp8 cache for clean scenario
        exp8_cache = BASE_DIR / 'results/latent/exp8_standard_stacking'

        std = load_or_train(
            create_stacking_ensemble, input_dim, X_tr, y_tr,
            exp8_cache if is_clean else cache_dir,
            f'standard_lat_{label}' if is_clean else f'std_{label}',
            force=force,
        )
        gan = load_or_train(
            create_stacking_ensemble_gan_optimized, input_dim, X_tr, y_tr,
            BASE_DIR / 'results/latent/exp7_combined_matrix_latent' if is_clean else cache_dir,
            f'ganopt_lat_{label}' if is_clean else f'gan_{label}',
            force=force,
        )

        # ── Evaluate: Standard Stack ──
        pred_c_std = std.predict(X_te_c)
        m_c = eval_metrics(y_te_clean, pred_c_std)
        print(f'    Standard: F1(clean)={m_c["f1_score"]:.4f}', end='')
        all_results.append({'train_scenario': label, 'test_type': 'clean',
                            'stack_type': 'standard', 'stage1_pct': 1.0,
                            'asr': None, 'false_positive_rate': None, **m_c})

        if X_te_g is not None:
            pred_g_std = std.predict(X_te_g)
            m_g = eval_metrics(y_te_gan, pred_g_std)
            print(f'  F1(GAN)={m_g["f1_score"]:.4f}')
            all_results.append({'train_scenario': label, 'test_type': 'gan_attack',
                                'stack_type': 'standard', 'stage1_pct': 0.95,
                                'asr': None, 'false_positive_rate': None, **m_g})
        else:
            print()

        # ── Evaluate: GAN-Opt Stack ──
        pred_c_gan = gan.predict(X_te_c)
        m_c = eval_metrics(y_te_clean, pred_c_gan)
        print(f'    GAN-Opt: F1(clean)={m_c["f1_score"]:.4f}', end='')
        all_results.append({'train_scenario': label, 'test_type': 'clean',
                            'stack_type': 'ganopt', 'stage1_pct': 1.0,
                            'asr': None, 'false_positive_rate': None, **m_c})

        if X_te_g is not None:
            pred_g_gan = gan.predict(X_te_g)
            m_g = eval_metrics(y_te_gan, pred_g_gan)
            print(f'  F1(GAN)={m_g["f1_score"]:.4f}')
            all_results.append({'train_scenario': label, 'test_type': 'gan_attack',
                                'stack_type': 'ganopt', 'stage1_pct': 0.95,
                                'asr': None, 'false_positive_rate': None, **m_g})
        else:
            print()

    # ── Save — format giống Exp8 ────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    csv_path = out_dir / 'exp10a_latent_results.csv'
    df.to_csv(csv_path, index=False)

    # ── Summary table giống Exp8 ────────────────────────────────────────────
    print('\n' + '='*80)
    print('✅ EXP10a — Stacking trên Poisoned Latent'.center(80))
    print('='*80)
    print(f'\n  So sánh với Exp8 (clean encoder):')
    print(f'\n  {"Scenario":<15}  {"Std F1 Clean":>13}  {"GAN F1 Clean":>13}  '
          f'{"Std F1 GAN":>11}  {"GAN F1 GAN":>11}')
    print('  ' + '-'*70)

    for label, *_ in [(s[0],) for s in scenarios]:
        sub = df[df['train_scenario'] == label]
        f1_std_c = sub[(sub['test_type']=='clean')   & (sub['stack_type']=='standard')]['f1_score']
        f1_gan_c = sub[(sub['test_type']=='clean')   & (sub['stack_type']=='ganopt')  ]['f1_score']
        f1_std_g = sub[(sub['test_type']=='gan_attack') & (sub['stack_type']=='standard')]['f1_score']
        f1_gan_g = sub[(sub['test_type']=='gan_attack') & (sub['stack_type']=='ganopt')  ]['f1_score']
        get = lambda s: s.values[0] if len(s) else float('nan')
        print(f'  {label:<15}  {get(f1_std_c):>13.4f}  {get(f1_gan_c):>13.4f}  '
              f'{get(f1_std_g):>11.4f}  {get(f1_gan_g):>11.4f}')

    # load Exp8 để so sánh trực tiếp
    exp8_csv = BASE_DIR / 'results/latent/exp8_standard_stacking/exp8_latent_results.csv'
    if exp8_csv.exists():
        df8 = pd.read_csv(exp8_csv)
        print(f'\n  📊 Exp8 reference (clean encoder):')
        for sc in ['clean', 'poison_05', 'poison_10', 'poison_15', 'poison_50']:
            row = df8[(df8['train_scenario']==sc) & (df8['test_type']=='clean')]
            if not row.empty:
                print(f'    Exp8 [{sc}] clean F1 = {row.iloc[0]["f1_score"]:.4f}')

    print(f'\n📁 Results: {csv_path}\n')


if __name__ == '__main__':
    main()
