"""
Experiment 2b RAW: Data Poisoning Attack — RAW Features
=========================================================

Cặp đôi với exp2b_poisoning_latent.py để so sánh công bằng RAW vs Latent:

  Exp2b RAW   : RAW poisoned data → train/test trên RAW features (50-dim)
  Exp2b Latent: RAW poisoned data → POISONED encoder → latent (64-dim)

  (Cùng dữ liệu train/test — chỉ khác feature representation)

Mục đích:
  Δ(Exp2b Latent − Exp2b RAW) = tác động của poisoned LATENT SPACE
  so với raw features, trên cùng dữ liệu poisoned.

So sánh chuỗi đầy đủ:
  Exp2 RAW    ← RAW poisoned, clean (giống Exp2 latent với clean encoder)
  Exp2b RAW   ← RAW poisoned, standalone script (file này)
  Exp2 Latent ← RAW poisoned → CLEAN encoder → latent
  Exp2b Latent← RAW poisoned → POISONED encoder → latent

  Δ(Exp2b RAW  − Exp2 RAW)    ≈ 0 (same data, same models → verify consistency)
  Δ(Exp2 Lat   − Exp2b RAW)   = clean latent vs raw
  Δ(Exp2b Lat  − Exp2b RAW)   = poisoned latent vs raw  [key comparison!]

Data:
  Train: datasets/splits/3.0_raw_from_latent/exp2_poisoning/poison_XX/X_train.npy
  Test:  datasets/splits/3.0_raw_from_latent/exp2_poisoning/poison_XX/X_test.npy
  (Cùng source với Exp2b Latent — đảm bảo đánh giá công bằng)

Model config: giống run_model_evaluation.py (Exp2 gốc) và exp2b_poisoning_latent.py.

Output format: giống Exp2 (summary CSV)
  Model, Accuracy, Precision, Recall, F1-Score, Train Time (s)

Results: results/raw/exp2b_poisoning/poison_XX/summary_TIMESTAMP.csv

Usage:
    python experiments/exp2b_poisoning_raw.py
    python experiments/exp2b_poisoning_raw.py --poison-rates 5 10
"""

import sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_SPLITS   = BASE_DIR / "datasets/splits/3.0_raw_from_latent"
RESULTS_BASE = BASE_DIR / "results/raw/exp2b_poisoning"
RANDOM_STATE = 42


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    return {
        'Accuracy':  round(float(accuracy_score(y_true,  y_pred)), 4),
        'Precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'Recall':    round(float(recall_score(y_true,    y_pred, zero_division=0)), 4),
        'F1-Score':  round(float(f1_score(y_true,         y_pred, zero_division=0)), 4),
    }


# ── Model builders — GIỐNG HỆTCONFIG exp2b_poisoning_latent.py ───────────────

def build_mlp(input_dim):
    """Dense(50)→Dense(1) — giống run_model_evaluation.py (Exp2)."""
    model = Sequential([
        Dense(50, input_dim=input_dim, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_eval_models(X_train, y_train, X_test, y_test, input_dim):
    rows = []

    # ── MLP ──
    print(f'  [1/5] MLP  Dense(50)→Dense(1)  epochs=30  batch=64 ...', flush=True)
    t0  = time.time()
    mlp = build_mlp(input_dim)
    mlp.fit(X_train, y_train, epochs=30, batch_size=64,
            validation_split=0.2, verbose=0,
            callbacks=[EarlyStopping(monitor='val_loss', patience=8,
                                     restore_best_weights=True, verbose=0)])
    t_mlp = time.time() - t0
    pred  = (mlp.predict(X_test, verbose=0).flatten() >= 0.5).astype(int)
    m     = compute_metrics(y_test, pred)
    print(f'     ✓  F1={m["F1-Score"]:.4f}  time={t_mlp:.1f}s')
    rows.append({'Model': 'MLP', **m, 'Train Time (s)': round(t_mlp, 2)})

    # ── SVM ──
    print(f'  [2/5] SVM  LinearSVC(C=1.0) ...', flush=True)
    t0  = time.time()
    svm = LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_STATE)
    svm.fit(X_train, y_train)
    t_svm = time.time() - t0
    pred  = svm.predict(X_test)
    m     = compute_metrics(y_test, pred)
    print(f'     ✓  F1={m["F1-Score"]:.4f}  time={t_svm:.1f}s')
    rows.append({'Model': 'SVM', **m, 'Train Time (s)': round(t_svm, 2)})

    # ── RF ──
    print(f'  [3/5] RF   n_estimators=100 ...', flush=True)
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=4)
    rf.fit(X_train, y_train)
    t_rf = time.time() - t0
    pred = rf.predict(X_test)
    m    = compute_metrics(y_test, pred)
    print(f'     ✓  F1={m["F1-Score"]:.4f}  time={t_rf:.1f}s')
    rows.append({'Model': 'RF', **m, 'Train Time (s)': round(t_rf, 2)})

    # ── KNN ──
    print(f'  [4/5] KNN  n_neighbors=5 ...', flush=True)
    t0  = time.time()
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=4)
    knn.fit(X_train, y_train)
    t_knn = time.time() - t0
    pred  = knn.predict(X_test)
    m     = compute_metrics(y_test, pred)
    print(f'     ✓  F1={m["F1-Score"]:.4f}  time={t_knn:.1f}s')
    rows.append({'Model': 'KNN', **m, 'Train Time (s)': round(t_knn, 2)})

    # ── NB ──
    print(f'  [5/5] NB   GaussianNB ...', flush=True)
    t0 = time.time()
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    t_nb = time.time() - t0
    pred = nb.predict(X_test)
    m    = compute_metrics(y_test, pred)
    print(f'     ✓  F1={m["F1-Score"]:.4f}  time={t_nb:.1f}s')
    rows.append({'Model': 'NB', **m, 'Train Time (s)': round(t_nb, 2)})

    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Exp2b RAW: Base models trên RAW poisoned data (cặp đôi với exp2b latent)'
    )
    parser.add_argument('--poison-rates', nargs='+', type=int, default=[5, 10, 15, 50])
    parser.add_argument('--raw-dir',    default=str(RAW_SPLITS))
    parser.add_argument('--output-dir', default=str(RESULTS_BASE))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)

    print('\n' + '='*80)
    print(' EXP2b RAW: POISONING — RAW FEATURES (Cặp đôi Exp2b Latent) '.center(80, '='))
    print('='*80)
    print("""
  Dùng CÙNG DỮ LIỆU với Exp2b Latent để so sánh công bằng:
    Exp2b RAW   : poisoned RAW → train/test trên RAW (50-dim)
    Exp2b Latent: poisoned RAW → POISONED encoder → latent (64-dim)

  Δ(Exp2b Latent − Exp2b RAW) = tác động thêm của poisoned latent space
""")

    # Load Exp2 RAW results (original) để verify consistency
    exp2_raw_dir = BASE_DIR / 'results/raw_fair/exp2_poisoning'
    exp2b_lat_dir = BASE_DIR / 'results/latent/exp2b_poisoning'

    all_summary = []

    for rate in args.poison_rates:
        rate_str  = f'{rate:02d}'
        data_dir  = raw_dir / f'exp2_poisoning/poison_{rate_str}'

        if not (data_dir / 'X_train.npy').exists():
            print(f'\n  ⚠️  Skipping poison_{rate}% — data not found at {data_dir}')
            continue

        print('\n' + '='*80)
        print(f'  POISON RATE: {rate}%'.center(80))
        print('='*80)

        X_train = np.load(data_dir / 'X_train.npy')
        y_train = np.load(data_dir / 'y_train.npy')
        X_test  = np.load(data_dir / 'X_test.npy')   # clean test set
        y_test  = np.load(data_dir / 'y_test.npy')
        input_dim = X_train.shape[1]

        # So sánh với clean (để đếm label flips)
        clean_y = raw_dir / 'exp1_baseline/y_train.npy'
        if clean_y.exists():
            y_clean = np.load(clean_y)
            flips   = (y_train != y_clean).sum()
            print(f'  Train: {len(X_train):,} × {input_dim}  ({flips:,} flips = {flips/len(y_clean)*100:.1f}%)')
        else:
            print(f'  Train: {len(X_train):,} × {input_dim}')
        print(f'  Test : {len(X_test):,} × {input_dim}  (clean test set — same as Exp2b Latent)\n')

        # Tải reference từ Exp2 RAW gốc để verify
        exp2_csv = sorted((exp2_raw_dir / f'poison_{rate_str}').glob('summary_*.csv')) \
                   if (exp2_raw_dir / f'poison_{rate_str}').exists() else []
        exp2_ref = {}
        if exp2_csv:
            df2 = pd.read_csv(exp2_csv[-1])
            exp2_ref = {row['Model']: row['F1-Score'] for _, row in df2.iterrows()}

        # Tải reference từ Exp2b Latent nếu đã chạy
        exp2b_csv = sorted((exp2b_lat_dir / f'poison_{rate_str}').glob('summary_*.csv')) \
                    if (exp2b_lat_dir / f'poison_{rate_str}').exists() else []
        exp2b_lat_ref = {}
        if exp2b_csv:
            dfb = pd.read_csv(exp2b_csv[-1])
            exp2b_lat_ref = {row['Model']: row['F1-Score'] for _, row in dfb.iterrows()}

        # Train & evaluate
        rows = train_eval_models(X_train, y_train, X_test, y_test, input_dim)

        # Save
        rate_out = out_dir / f'poison_{rate_str}'
        rate_out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        df = pd.DataFrame(rows)
        csv_path = rate_out / f'summary_{ts}.csv'
        df.to_csv(csv_path, index=False)

        # Print comparison table
        has_exp2  = bool(exp2_ref)
        has_exp2b = bool(exp2b_lat_ref)

        header = f'\n  {"Model":<6} {"Acc":>8} {"Prec":>8} {"Rec":>8} {"F1":>8} {"Time":>7}'
        if has_exp2:  header += f' {"Exp2":>8}'
        if has_exp2b: header += f' {"Exp2bLat":>9} {"Δ(Lat-RAW)":>11}'
        print(header)
        print('  ' + '-' * (50 + (9 if has_exp2 else 0) + (21 if has_exp2b else 0)))

        for row in rows:
            name = row['Model']
            f1_raw = row['F1-Score']
            line   = (f'  {name:<6} {row["Accuracy"]:>8.4f} {row["Precision"]:>8.4f}'
                      f' {row["Recall"]:>8.4f} {f1_raw:>8.4f} {row["Train Time (s)"]:>7.2f}')
            if has_exp2:
                f1_2 = exp2_ref.get(name, float('nan'))
                line += f' {f1_2:>8.4f}' if not (isinstance(f1_2, float) and str(f1_2) == 'nan') else f' {"--":>8}'
            if has_exp2b:
                f1_lat = exp2b_lat_ref.get(name, None)
                if f1_lat is not None:
                    delta  = f1_lat - f1_raw
                    line  += f' {f1_lat:>9.4f} {delta:>+11.4f}'
                else:
                    line  += f' {"--":>9} {"--":>11}'
            print(line)

        print(f'\n  📁 Saved → {csv_path}')
        all_summary.append({'poison_rate': rate, 'rows': rows,
                            'exp2_ref': exp2_ref, 'exp2b_lat_ref': exp2b_lat_ref})

    # ── Bảng tổng kết ──────────────────────────────────────────────────────
    if len(all_summary) > 1:
        print('\n' + '='*80)
        print('✅ EXP2b RAW — TỔNG KẾT F1-Score'.center(80))
        print('='*80)
        model_names = [r['Model'] for r in all_summary[0]['rows']]

        # Header
        header = f'  {"Model":<6}'
        for s in all_summary:
            rs = f'{s["poison_rate"]:02d}'
            header += f'  P{rs}(raw)  P{rs}(lat)  P{rs}(Δ)'
        print(f'\n{header}')
        print('  ' + '-' * len(header))

        for name in model_names:
            row_txt = f'  {name:<6}'
            for s in all_summary:
                row_raw = next((r for r in s['rows'] if r['Model'] == name), {})
                f1_raw  = row_raw.get('F1-Score', float('nan'))
                f1_lat  = s['exp2b_lat_ref'].get(name, None)
                delta   = f'{f1_lat - f1_raw:+.4f}' if f1_lat is not None else '     --'
                lat_str = f'{f1_lat:.4f}' if f1_lat is not None else '     --'
                row_txt += f'   {f1_raw:.4f}  {lat_str}  {delta}'
            print(row_txt)

        print(f"""
  Giải thích:
    P{args.poison_rates[0]:02d}(raw) = Exp2b RAW F1 tại poison rate  (file này)
    P{args.poison_rates[0]:02d}(lat) = Exp2b Latent F1 (poisoned encoder)
    Δ     = lat − raw  →  Δ<0: poisoned encoder làm GIẢM thêm F1
                        →  Δ>0: latent space vẫn tốt hơn dù encoder bị nhiễm

  Verify consistency với Exp2 RAW gốc:
    Δ(Exp2b RAW − Exp2 RAW) ≈ 0  (cùng data, cùng models)
    Nếu Δ lớn → có vấn đề với data pipeline hoặc random seed
""")

        print(f'  📁 Results: {out_dir}/')


if __name__ == '__main__':
    main()
