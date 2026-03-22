"""
Experiment 11 LATENT: Single Encoder — Baseline, Poisoning, GAN, Stacking
==========================================================================

So sánh SingleEncoder vs DualEncoder:
  DualEncoder (Exp1-3 Latent): 2 AEs riêng biệt → z=64 (32+32 concat)
  SingleEncoder (Exp11)       : 1 AE chung all data → z=64

Sub-experiments:
  Exp11-1: Baseline (clean)                ↔ Exp1 Latent
  Exp11-2: Poisoning, clean AE             ↔ Exp2 Latent
  Exp11-2b:Poisoning, poisoned AE          ↔ Exp2b Latent
  Exp11-3: GAN attack test                 ↔ Exp3 Latent
  Exp11-4: Stacking on poisoned SingleAE   ↔ Exp10a (Stacking on poisoned DualAE) ← MỚI

Data: datasets/splits/3.2_latent_single_enc/
Usage:
    python experiments/latent/exp11_single_enc_latent.py
    python experiments/latent/exp11_single_enc_latent.py --sub exp11-4
    python experiments/latent/exp11_single_enc_latent.py --sub exp11-4 --poison-rates 10 50
"""

import sys, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from models.ensemble.stacking import (
    create_stacking_ensemble,
    create_stacking_ensemble_gan_optimized,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
SINGLE_ENC_DATA = BASE_DIR / "datasets/splits/3.2_latent_single_enc"
RESULTS_BASE    = BASE_DIR / "results/latent/exp11_single_encoder"
RANDOM_STATE    = 42


# ── Metrics (giống Exp2 Latent) ───────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    return {
        'Accuracy':  round(float(accuracy_score(y_true,  y_pred)), 4),
        'Precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'Recall':    round(float(recall_score(y_true,    y_pred, zero_division=0)), 4),
        'F1-Score':  round(float(f1_score(y_true,         y_pred, zero_division=0)), 4),
    }


# ── Model builders — giống exp2b (config Exp2) ───────────────────────────────

def build_mlp(input_dim):
    model = Sequential([
        Dense(50, input_dim=input_dim, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_eval_models(X_train, y_train, X_test, y_test, input_dim):
    rows = []

    # MLP
    print(f'  [1/5] MLP ...', end=' ', flush=True)
    t0  = time.time()
    mlp = build_mlp(input_dim)
    mlp.fit(X_train, y_train, epochs=30, batch_size=64,
            validation_split=0.2, verbose=0,
            callbacks=[EarlyStopping(monitor='val_loss', patience=8,
                                     restore_best_weights=True, verbose=0)])
    t   = time.time() - t0
    pred = (mlp.predict(X_test, verbose=0).flatten() >= 0.5).astype(int)
    m   = compute_metrics(y_test, pred)
    print(f'F1={m["F1-Score"]:.4f}  {t:.1f}s')
    rows.append({'Model': 'MLP', **m, 'Train Time (s)': round(t, 2)})

    # SVM
    print(f'  [2/5] SVM ...', end=' ', flush=True)
    t0  = time.time()
    svm = LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_STATE)
    svm.fit(X_train, y_train)
    t   = time.time() - t0
    pred = svm.predict(X_test)
    m   = compute_metrics(y_test, pred)
    print(f'F1={m["F1-Score"]:.4f}  {t:.1f}s')
    rows.append({'Model': 'SVM', **m, 'Train Time (s)': round(t, 2)})

    # RF
    print(f'  [3/5] RF  ...', end=' ', flush=True)
    t0  = time.time()
    rf  = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=4)
    rf.fit(X_train, y_train)
    t   = time.time() - t0
    pred = rf.predict(X_test)
    m   = compute_metrics(y_test, pred)
    print(f'F1={m["F1-Score"]:.4f}  {t:.1f}s')
    rows.append({'Model': 'RF', **m, 'Train Time (s)': round(t, 2)})

    # KNN
    print(f'  [4/5] KNN ...', end=' ', flush=True)
    t0  = time.time()
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=4)
    knn.fit(X_train, y_train)
    t   = time.time() - t0
    pred = knn.predict(X_test)
    m   = compute_metrics(y_test, pred)
    print(f'F1={m["F1-Score"]:.4f}  {t:.1f}s')
    rows.append({'Model': 'KNN', **m, 'Train Time (s)': round(t, 2)})

    # NB
    print(f'  [5/5] NB  ...', end=' ', flush=True)
    t0  = time.time()
    nb  = GaussianNB()
    nb.fit(X_train, y_train)
    t   = time.time() - t0
    pred = nb.predict(X_test)
    m   = compute_metrics(y_test, pred)
    print(f'F1={m["F1-Score"]:.4f}  {t:.1f}s')
    rows.append({'Model': 'NB', **m, 'Train Time (s)': round(t, 2)})

    return rows


# ── Load reference từ DualEncoder experiments ─────────────────────────────────

def load_dual_ref(exp_name: str, sub_dir: str = None):
    """Load F1 results từ DualEncoder experiments để so sánh."""
    base = BASE_DIR / 'results/latent'
    paths = {
        'exp1': base / 'exp1_baseline_latent',
        'exp2': base / 'exp2_poisoning',
        'exp2b': base / 'exp2b_poisoning',
    }
    ref = {}
    if exp_name not in paths:
        return ref
    d = paths[exp_name] / sub_dir if sub_dir else paths[exp_name]
    csvs = sorted(d.glob('summary_*.csv')) if d.exists() else []
    if csvs:
        df = pd.read_csv(csvs[-1])
        ref = {row['Model']: row['F1-Score'] for _, row in df.iterrows()}
    return ref


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Exp11 LATENT: Single Encoder (compare SingleEnc vs DualEnc)'
    )
    parser.add_argument('--sub', choices=['exp11-1', 'exp11-2', 'exp11-2b', 'exp11-3', 'exp11-4', 'all'],
                        default='all', help='Sub-experiment to run')
    parser.add_argument('--poison-rates', nargs='+', type=int, default=[5, 10, 15, 50])
    parser.add_argument('--data-dir',   default=str(SINGLE_ENC_DATA))
    parser.add_argument('--output-dir', default=str(RESULTS_BASE))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.output_dir)

    if not data_dir.exists():
        print(f'\n❌ Exp11 data not found: {data_dir}')
        print('   Run: python pipelines/preprocessing/prepare_exp11_data.py')
        sys.exit(1)

    print('\n' + '='*80)
    print(' EXPERIMENT 11 LATENT: SINGLE ENCODER vs DUAL ENCODER '.center(80, '='))
    print('='*80)
    print("""
  SingleEncoder: 1 AE train trên ALL data (benign+malicious mixed) → 64-dim
  DualEncoder  : 2 AE riêng biệt (benign_AE + malicious_AE)       → 64-dim (32+32)

  Giả thuyết:
    H0: SingleEncoder ≈ DualEncoder (không có sự khác biệt đáng kể)
    H1: DualEncoder > SingleEncoder (dual-view representation có giá trị)
""")

    # ── EXP11-1: Baseline (clean) ─────────────────────────────────────────
    if args.sub in ['exp11-1', 'all']:
        print('\n' + '='*80)
        print('  EXP11-1: BASELINE (Single Encoder, Clean Data)'.center(80))
        print('='*80)

        base_dir = data_dir / 'exp11_baseline'
        if not (base_dir / 'X_train.npy').exists():
            print(f'  ⚠️  Baseline data not found: {base_dir}')
        else:
            X_tr = np.load(base_dir / 'X_train.npy')
            y_tr = np.load(base_dir / 'y_train.npy')
            X_te = np.load(base_dir / 'X_test.npy')
            y_te = np.load(base_dir / 'y_test.npy')
            print(f'  Train: {X_tr.shape}  Test: {X_te.shape}  (SingleEncoder 64-dim)\n')

            dual_ref = load_dual_ref('exp1')

            rows = train_eval_models(X_tr, y_tr, X_te, y_te, X_tr.shape[1])

            # Save
            out_base = out_dir / 'exp11_baseline'
            out_base.mkdir(parents=True, exist_ok=True)
            ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv = out_base / f'summary_{ts}.csv'
            pd.DataFrame(rows).to_csv(csv, index=False)

            # Print comparison
            print(f'\n  {"Model":<6} {"F1(SingleEnc)":>14}', end='')
            if dual_ref: print(f' {"F1(DualEnc)":>12} {"Δ(Dual-Single)":>15}', end='')
            print()
            print('  ' + '-'*55)
            for row in rows:
                nm   = row['Model']
                f1s  = row['F1-Score']
                line = f'  {nm:<6} {f1s:>14.4f}'
                if dual_ref:
                    f1d  = dual_ref.get(nm, float('nan'))
                    delta = f1d - f1s if not str(f1d) == 'nan' else float('nan')
                    line += f' {f1d:>12.4f} {delta:>+15.4f}'
                print(line)
            print(f'\n  📁 {csv}')

    # ── EXP11-2: Poisoning (clean AE) ──────────────────────────────────────
    if args.sub in ['exp11-2', 'all']:
        print('\n' + '='*80)
        print('  EXP11-2: POISONING — SingleEnc CLEAN AE, poisoned labels'.center(80))
        print('='*80)
        print('  CLEAN encoder — chỉ labels bị nhiễm (tương đương Exp2 Latent DualEnc)\n')

        poison_summary = []
        for rate in args.poison_rates:
            rs       = f'{rate:02d}'
            p_dir    = data_dir / f'exp11_poisoning/poison_{rs}'
            if not (p_dir / 'X_train.npy').exists():
                print(f'  ⚠️  Skipping poison_{rs} — not found')
                continue

            print(f'\n  ── Poison {rate}% ──')
            X_tr = np.load(p_dir / 'X_train.npy')
            y_tr = np.load(p_dir / 'y_train.npy')
            X_te = np.load(p_dir / 'X_test.npy')
            y_te = np.load(p_dir / 'y_test.npy')
            print(f'  Train: {X_tr.shape}  [clean AE, poisoned labels]')

            dual_ref   = load_dual_ref('exp2',  f'poison_{rs}')
            dual2b_ref = load_dual_ref('exp2b', f'poison_{rs}')

            rows = train_eval_models(X_tr, y_tr, X_te, y_te, X_tr.shape[1])

            out_p = out_dir / f'exp11_poisoning/poison_{rs}'
            out_p.mkdir(parents=True, exist_ok=True)
            ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv = out_p / f'summary_{ts}.csv'
            pd.DataFrame(rows).to_csv(csv, index=False)

            print()
            print('  {:6} {:>14} {:>12} {:>6} {:>12} {:>6}'.format(
                  'Model', 'SingleEnc(cl)', 'DualEnc(cl)', 'Δ1', 'DualEnc(ps)', 'Δ2'))
            print('  ' + '-'*64)
            for row in rows:
                nm  = row['Model']
                f1s = row['F1-Score']
                f1d = dual_ref.get(nm, float('nan'))
                f1b = dual2b_ref.get(nm, float('nan'))
                d1  = f1d - f1s if str(f1d) != 'nan' else float('nan')
                d2  = f1b - f1s if str(f1b) != 'nan' else float('nan')
                print(f'  {nm:<6} {f1s:>14.4f}'
                      f' {f1d:>12.4f} {d1:>+6.4f}'
                      f' {f1b:>12.4f} {d2:>+6.4f}')
            print(f'  📁 {csv}')
            poison_summary.append({'rate': rate, 'rows': rows})

    # ── EXP11-2b: Poisoning (poisoned AE) ────────────────────────────────
    if args.sub in ['exp11-2b', 'all']:
        print('\n' + '='*80)
        print('  EXP11-2b: POISONING — SingleEnc POISONED AE (↔ Exp2b DualEnc)'.center(80))
        print('='*80)
        print('  POISONED SingleEncoder: retrain AE trên poisoned data per rate')
        print('  Tương đương Exp2b Latent (DualEnc bị nhiễm) nhưng dùng 1 AE chung\n')

        for rate in args.poison_rates:
            rs      = f'{rate:02d}'
            pp_dir  = data_dir / f'exp11_poisoning_penc/poison_{rs}'
            if not (pp_dir / 'X_train.npy').exists():
                print(f'  ⚠️  Skipping poison_{rs} — exp11_poisoning_penc not found')
                print(f'       Chạy trước: python pipelines/preprocessing/prepare_exp11_data.py')
                continue

            print(f'\n  ── Poison {rate}% ──')
            X_tr = np.load(pp_dir / 'X_train.npy')
            y_tr = np.load(pp_dir / 'y_train.npy')
            X_te = np.load(pp_dir / 'X_test.npy')
            y_te = np.load(pp_dir / 'y_test.npy')
            print(f'  Train: {X_tr.shape}  [POISONED AE + poisoned labels]')

            # So sánh với Exp2b (DualEnc poisoned) và Exp11-2 (SingleEnc clean)
            dual2b_ref = load_dual_ref('exp2b', f'poison_{rs}')
            # Đọc kết quả Exp11-2 (clean AE) nếu có
            s11_clean_ref = {}
            csvs_clean = sorted((out_dir / f'exp11_poisoning/poison_{rs}').glob('summary_*.csv'))
            if csvs_clean:
                df_c = pd.read_csv(csvs_clean[-1])
                s11_clean_ref = {r['Model']: r['F1-Score'] for _, r in df_c.iterrows()}

            rows = train_eval_models(X_tr, y_tr, X_te, y_te, X_tr.shape[1])

            out_pp = out_dir / f'exp11_poisoning_penc/poison_{rs}'
            out_pp.mkdir(parents=True, exist_ok=True)
            ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv = out_pp / f'summary_{ts}.csv'
            pd.DataFrame(rows).to_csv(csv, index=False)

            print()
            print('  {:6} {:>14} {:>14} {:>6} {:>12} {:>6}'.format(
                  'Model', 'SingleEnc(ps)', 'SingleEnc(cl)', 'Δ1', 'DualEnc(ps)', 'Δ2'))
            print('  ' + '-'*70)
            for row in rows:
                nm   = row['Model']
                f1ps = row['F1-Score']            # SingleEnc poisoned AE
                f1cl = s11_clean_ref.get(nm, float('nan'))  # SingleEnc clean AE
                f1dp = dual2b_ref.get(nm, float('nan'))     # DualEnc poisoned AE
                d1   = f1cl - f1ps if str(f1cl) != 'nan' else float('nan')  # AE clean vs poisoned
                d2   = f1dp - f1ps if str(f1dp) != 'nan' else float('nan')  # Dual vs Single (both poisoned)
                print(f'  {nm:<6} {f1ps:>14.4f}'
                      f' {f1cl:>14.4f} {d1:>+6.4f}'
                      f' {f1dp:>12.4f} {d2:>+6.4f}')
            print()
            print('  Chú thích:')
            print(f'  . Δ1 = SingleEnc(cl) - SingleEnc(ps): tác động riêng của poisoned SingleAE')
            print(f'  . Δ2 = DualEnc(ps)   - SingleEnc(ps): lợi thế của DualEncoder khi cả hai bị nhiễm')
            print(f'  📁 {csv}')

    # ── EXP11-3: GAN Attack ───────────────────────────────────────────────
    if args.sub in ['exp11-3', 'all']:
        print('\n' + '='*80)
        print('  EXP11-3: GAN ATTACK TEST (Single Encoder)'.center(80))
        print('='*80)
        print('  Train trên clean, test trên GAN attack traffic\n')

        gan_dir  = data_dir / 'exp11_gan_attack'
        base_dir = data_dir / 'exp11_baseline'
        if not (gan_dir / 'X_test.npy').exists():
            print(f'  ⚠️  GAN data not found: {gan_dir}')
        else:
            # Train trên clean baseline, test trên GAN
            X_tr = np.load(base_dir / 'X_train.npy')
            y_tr = np.load(base_dir / 'y_train.npy')
            X_te = np.load(gan_dir / 'X_test.npy')
            y_te = np.load(gan_dir / 'y_test.npy')
            print(f'  Train: {X_tr.shape} (clean)  |  Test: {X_te.shape} (GAN attack)')

            rows = train_eval_models(X_tr, y_tr, X_te, y_te, X_tr.shape[1])

            out_gan = out_dir / 'exp11_gan_attack'
            out_gan.mkdir(parents=True, exist_ok=True)
            ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv = out_gan / f'summary_{ts}.csv'
            pd.DataFrame(rows).to_csv(csv, index=False)
            print(f'\n  📁 {csv}')

    # ── EXP11-4: Stacking on Poisoned SingleAE (↔ Exp10a DualEnc) ─────────
    if args.sub in ['exp11-4', 'all']:
        print('\n' + '='*80)
        print('  EXP11-4: STACKING — SingleEnc POISONED AE vs Exp10a DualEnc'.center(80))
        print('='*80)
        print('  Cấu trúc stacking giống Exp10a: standard + GAN-opt ensemble')
        print('  Khác biệt duy nhất: latent từ POISONED SingleAE (1) vs DualAE (2)\n')

        # Load Exp10a results để so sánh
        exp10a_csv = BASE_DIR / 'results/latent/exp10a_stacking/exp10a_latent_results.csv'
        exp10a_ref_std = {}
        exp10a_ref_gan = {}
        if exp10a_csv.exists():
            df10a = pd.read_csv(exp10a_csv)
            for sc in df10a['train_scenario'].unique():
                r_std = df10a[(df10a['train_scenario'] == sc)
                              & (df10a['test_type'] == 'clean')
                              & (df10a['stack_type'] == 'standard')]
                r_gan = df10a[(df10a['train_scenario'] == sc)
                              & (df10a['test_type'] == 'clean')
                              & (df10a['stack_type'] == 'ganopt')]
                if not r_std.empty:
                    exp10a_ref_std[sc] = r_std.iloc[0]['f1_score']
                if not r_gan.empty:
                    exp10a_ref_gan[sc] = r_gan.iloc[0]['f1_score']
            print('  ✓ Exp10a reference loaded: ' + str(list(exp10a_ref_std.keys())))
        else:
            print('  ⚠️  Exp10a results not found — Delta vs DualEnc sẽ hiển thị N/A')

        # GAN raw data (sẽ encode bởi POISONED SingleAE per rate — giống Exp10a)
        # Exp10a encode GAN test bởi poisoned DualEncoder của từng rate
        # → để so sánh công bằng, Exp11-4 cũng phải encode GAN bởi poisoned SingleAE
        raw_splits = BASE_DIR / 'datasets/splits/3.0_raw_from_latent'
        gan_raw_path = raw_splits / 'exp3_gan_attack/X_test.npy'
        gan_raw_y    = raw_splits / 'exp3_gan_attack/y_test.npy'
        X_gan_raw = np.load(gan_raw_path).astype(np.float32) if gan_raw_path.exists() else None
        y_te_gan  = np.load(gan_raw_y) if gan_raw_y.exists() else None
        if X_gan_raw is None:
            print('  ⚠️  GAN raw data not found — GAN test sẽ bị bỏ qua')

        stk_results = []
        cache_dir   = out_dir / 'exp11_stacking_cache'

        rates_to_run = [0] + args.poison_rates
        for rate in rates_to_run:
            if rate == 0:
                rs = 'clean'
                sc = 'clean'
                pp_dir = data_dir / 'exp11_baseline'
            else:
                rs = f'{rate:02d}'
                sc = f'poison_{rs}'
                pp_dir = data_dir / f'exp11_poisoning_penc/poison_{rs}'
                
            if not (pp_dir / 'X_train.npy').exists():
                print(f'  ⚠️  Skipping {sc} — chạy prepare_exp11_data.py trước')
                continue

            print(f'\n  ── Scenario: {sc} ──')
            X_tr = np.load(pp_dir / 'X_train.npy')
            y_tr = np.load(pp_dir / 'y_train.npy')
            X_te = np.load(pp_dir / 'X_test.npy')
            y_te = np.load(pp_dir / 'y_test.npy')
            input_dim = X_tr.shape[1]
            if rate == 0:
                print(f'  Train: {X_tr.shape}  Test: {X_te.shape}  [clean SingleAE]')
            else:
                print(f'  Train: {X_tr.shape}  Test: {X_te.shape}  [poisoned SingleAE]')

            # Encode GAN test
            X_te_gan = None
            if X_gan_raw is not None:
                if rate == 0:
                    p_enc_path = data_dir / 'models/single_encoder.h5'
                else:
                    p_enc_path = data_dir / f'models/poisoned_enc/poison_{rs}/single_encoder_poisoned.h5'
                    
                if p_enc_path.exists():
                    print(f'  Encoding GAN test...', end=' ', flush=True)
                    p_enc = tf.keras.models.load_model(str(p_enc_path))
                    X_te_gan = p_enc.predict(X_gan_raw, verbose=0)
                    print(f'✓ {X_te_gan.shape}')
                else:
                    gan_tdir  = data_dir / 'exp11_gan_attack'
                    X_te_gan  = np.load(gan_tdir / 'X_test.npy') if (gan_tdir / 'X_test.npy').exists() else None
                    print(f'  ⚠️  Encoder not found, using clean AE GAN test')

            def _metrics(yt, yp):
                return {
                    'accuracy':  round(float(accuracy_score(yt,  yp)), 6),
                    'precision': round(float(precision_score(yt, yp, zero_division=0)), 6),
                    'recall':    round(float(recall_score(yt,    yp, zero_division=0)), 6),
                    'f1_score':  round(float(f1_score(yt,         yp, zero_division=0)), 6),
                }

            # Standard stacking
            print('  [Std] Training standard stacking...', flush=True)
            t0  = time.time()
            std = create_stacking_ensemble(input_dim=input_dim)
            std.fit(X_tr, y_tr, verbose=False)
            t_std = time.time() - t0
            m_std_c = _metrics(y_te, std.predict(X_te))
            print(f'  [Std] F1(clean)={m_std_c["f1_score"]:.4f}  {t_std:.1f}s', end='')
            stk_results.append({'poison_rate': rate, 'stack_type': 'standard',
                                 'test_type': 'clean', 'train_time_s': round(t_std, 2),
                                 **m_std_c})
            if X_te_gan is not None:
                m_std_g = _metrics(y_te_gan, std.predict(X_te_gan))
                print(f'  F1(GAN)={m_std_g["f1_score"]:.4f}')
                stk_results.append({'poison_rate': rate, 'stack_type': 'standard',
                                     'test_type': 'gan_attack', 'train_time_s': round(t_std, 2),
                                     **m_std_g})
            else:
                print()

            # Save standard stack cache
            try:
                sc_path = cache_dir / f'std_poison_{rs}'
                sc_path.mkdir(parents=True, exist_ok=True)
                std.save(sc_path)
            except Exception:
                pass

            # GAN-optimized stacking
            print('  [GAN-Opt] Training GAN-opt stacking...', flush=True)
            t0  = time.time()
            gan_ens = create_stacking_ensemble_gan_optimized(input_dim=input_dim)
            gan_ens.fit(X_tr, y_tr, verbose=False)
            t_gan = time.time() - t0
            m_gan_c = _metrics(y_te, gan_ens.predict(X_te))
            print(f'  [GAN-Opt] F1(clean)={m_gan_c["f1_score"]:.4f}  {t_gan:.1f}s', end='')
            stk_results.append({'poison_rate': rate, 'stack_type': 'ganopt',
                                 'test_type': 'clean', 'train_time_s': round(t_gan, 2),
                                 **m_gan_c})
            if X_te_gan is not None:
                m_gan_g = _metrics(y_te_gan, gan_ens.predict(X_te_gan))
                print(f'  F1(GAN)={m_gan_g["f1_score"]:.4f}')
                stk_results.append({'poison_rate': rate, 'stack_type': 'ganopt',
                                     'test_type': 'gan_attack', 'train_time_s': round(t_gan, 2),
                                     **m_gan_g})
            else:
                print()

        # Save & print summary
        if stk_results:
            df_stk  = pd.DataFrame(stk_results)
            stk_out = out_dir / 'exp11_stacking'
            stk_out.mkdir(parents=True, exist_ok=True)
            ts_s    = datetime.now().strftime('%Y%m%d_%H%M%S')
            stk_csv = stk_out / f'summary_{ts_s}.csv'
            df_stk.to_csv(stk_csv, index=False)

            print('\n  ' + '-'*76)
            print('  {:12}  {:>10}  {:>12}  {:>11}  {:>11}  {:>9}'.format(
                  'Scenario', 'Std-S11', 'Std-D10a', 'GANOpt-S11', 'GANOpt-D10a', 'Delta-Std'))
            for rate in rates_to_run:
                if rate == 0:
                    sc = 'clean'
                else:
                    rs  = f'{rate:02d}'
                    sc  = f'poison_{rs}'
                sub = df_stk[(df_stk['poison_rate'] == rate) & (df_stk['test_type'] == 'clean')]
                f1s_vals = sub[sub['stack_type'] == 'standard']['f1_score'].values
                f1g_vals = sub[sub['stack_type'] == 'ganopt'  ]['f1_score'].values
                f1s = f1s_vals[0] if len(f1s_vals) else float('nan')
                f1g = f1g_vals[0] if len(f1g_vals) else float('nan')
                f1d_std = exp10a_ref_std.get(sc, float('nan'))
                f1d_gan = exp10a_ref_gan.get(sc, float('nan'))
                delta = f1d_std - f1s if str(f1d_std) != 'nan' else float('nan')
                delta_str = '{:+.4f}'.format(delta) if str(delta) != 'nan' else '   N/A'
                d10a_std_str = '{:.4f}'.format(f1d_std) if str(f1d_std) != 'nan' else '   N/A'
                d10a_gan_str = '{:.4f}'.format(f1d_gan) if str(f1d_gan) != 'nan' else '   N/A'
                print('  {:12}  {:>10.4f}  {:>12}  {:>11.4f}  {:>11}  {:>9}'.format(
                      sc, f1s, d10a_std_str, f1g, d10a_gan_str, delta_str))
            print()
            print('  Chú thích:')
            print('  . Std-S11    = SingleEnc poisoned AE + standard stack (Exp11-4)')
            print('  . Std-D10a   = DualEnc  poisoned AE + standard stack (Exp10a)')
            print('  . Delta-Std  = Dual - Single: lợi thế DualEncoder khi cùng bị nhiễm + cùng stacking')
            print('  . Delta > 0 → DualEncoder có lợi thế ngay cả sau khi bị nhiễm')
            print(f'\n  📁 {stk_csv}')

    # ── Final comparison summary ──────────────────────────────────────────
    print('\n' + '='*80)
    print('✅ EXP11 — SingleEncoder vs DualEncoder COMPARISON'.center(80))
    print('='*80)
    print("""
  So sánh đầy đủ (xem CSV files trong results/latent/exp11_single_encoder/):

  Scenario           SingleEnc  DualEnc   Δ (Dual-Single)
  ─────────────────────────────────────────────────────────
  Baseline (clean)   Exp11-1    Exp1-Lat  ?
  Poison (clean AE)  Exp11-2    Exp2-Lat  ?
  Poison (pois. AE)  Exp11-2b   Exp2b     ?
  Stacking+Poison    Exp11-4    Exp10a    ? ← thêm mới

  Δ > 0: DualEncoder tốt hơn → dual-view representation có giá trị  ✓
  Δ < 0: SingleEncoder tốt hơn → AE không cần tách biệt             ✗
  Δ ≈ 0: Tương đương → thiết kế đơn giản là đủ

  📁 Results: {out_dir}/
""".format(out_dir=out_dir))


if __name__ == '__main__':
    main()
