"""
Experiment 2b: Data Poisoning Attack — LATENT Features (Poisoned Encoder)
=========================================================================

So sánh trực tiếp với Exp2 Latent:
  Exp2  Latent: RAW poisoned → CLEAN   encoder → latent → train/test
  Exp2b Latent: RAW poisoned → POISONED encoder → latent → train/test
                └─ Đây là điểm khác biệt duy nhất

Mục đích:
  Đo thêm tác động của việc encoder bị nhiễm (ngoài label bị nhiễm).
  Exp2 đo: "poisoned labels ảnh hưởng thế nào trong latent space"
  Exp2b đo: "poisoned labels + poisoned encoder ảnh hưởng thế nào"
  → Δ(Exp2b - Exp2) = tác động riêng của poisoned encoder lên base models

Data:
  Train: datasets/splits/exp10_latent/poison_XX/X_train.npy  ← poisoned encoder
  Test:  datasets/splits/exp10_latent/poison_XX/X_test.npy   ← poisoned encoder (realistic)
  (Tạo bởi: python pipelines/preprocessing/prepare_exp10_data.py)

Output format: giống Exp2 Latent
  results/latent/exp2b_poisoning/poison_XX/summary_TIMESTAMP.csv
  Columns: Model, Accuracy, Precision, Recall, F1-Score, Train Time (s)

Usage:
    python experiments/latent/exp2b_poisoning_latent.py
    python experiments/latent/exp2b_poisoning_latent.py --poison-rates 5 10
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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# ── Paths ─────────────────────────────────────────────────────────────────────
LATENT_EXP10 = BASE_DIR / "datasets/splits/exp10_latent"    # poisoned encoder data
RESULTS_BASE = BASE_DIR / "results/latent/exp2b_poisoning"
RANDOM_STATE = 42


# ── Metrics (giống Exp2) ──────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    return {
        'Accuracy':  round(float(accuracy_score(y_true,  y_pred)), 4),
        'Precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'Recall':    round(float(recall_score(y_true,    y_pred, zero_division=0)), 4),
        'F1-Score':  round(float(f1_score(y_true,         y_pred, zero_division=0)), 4),
    }


# ── Model builders (giống run_ensemble_evaluation.py / run_model_evaluation.py) ──

def build_mlp(input_dim):
    model = Sequential([
        Dense(50, input_dim=input_dim, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_eval_models(X_train, y_train, X_test, y_test, input_dim):
    """Train và evaluate từng model, trả về list of row dicts."""
    rows = []

    # ── MLP ──
    print(f'  [1/5] MLP  Dense(50)→Dense(1)  epochs=30  batch=64 ...', flush=True)
    t0 = time.time()
    mlp = build_mlp(input_dim)
    mlp.fit(X_train, y_train,
            epochs=30, batch_size=64,
            validation_split=0.2, verbose=0,
            callbacks=[EarlyStopping(monitor='val_loss', patience=8,
                                     restore_best_weights=True, verbose=0)])
    t_mlp = time.time() - t0
    pred_mlp = (mlp.predict(X_test, verbose=0).flatten() >= 0.5).astype(int)
    m = compute_metrics(y_test, pred_mlp)
    print(f'     ✓  F1={m["F1-Score"]:.4f}  time={t_mlp:.1f}s')
    rows.append({'Model': 'MLP', **m, 'Train Time (s)': round(t_mlp, 2)})

    # ── SVM ──
    print(f'  [2/5] SVM  LinearSVC(C=1.0) ...', flush=True)
    t0 = time.time()
    svm = LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_STATE)
    svm.fit(X_train, y_train)
    t_svm = time.time() - t0
    pred_svm = svm.predict(X_test)
    m = compute_metrics(y_test, pred_svm)
    print(f'     ✓  F1={m["F1-Score"]:.4f}  time={t_svm:.1f}s')
    rows.append({'Model': 'SVM', **m, 'Train Time (s)': round(t_svm, 2)})

    # ── RF ──
    print(f'  [3/5] RF   n_estimators=100 ...', flush=True)
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=4)
    rf.fit(X_train, y_train)
    t_rf = time.time() - t0
    pred_rf = rf.predict(X_test)
    m = compute_metrics(y_test, pred_rf)
    print(f'     ✓  F1={m["F1-Score"]:.4f}  time={t_rf:.1f}s')
    rows.append({'Model': 'RF', **m, 'Train Time (s)': round(t_rf, 2)})

    # ── KNN ──
    print(f'  [4/5] KNN  n_neighbors=5 ...', flush=True)
    t0 = time.time()
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=4)
    knn.fit(X_train, y_train)
    t_knn = time.time() - t0
    pred_knn = knn.predict(X_test)
    m = compute_metrics(y_test, pred_knn)
    print(f'     ✓  F1={m["F1-Score"]:.4f}  time={t_knn:.1f}s')
    rows.append({'Model': 'KNN', **m, 'Train Time (s)': round(t_knn, 2)})

    # ── NB ──
    print(f'  [5/5] NB   GaussianNB ...', flush=True)
    t0 = time.time()
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    t_nb = time.time() - t0
    pred_nb = nb.predict(X_test)
    m = compute_metrics(y_test, pred_nb)
    print(f'     ✓  F1={m["F1-Score"]:.4f}  time={t_nb:.1f}s')
    rows.append({'Model': 'NB', **m, 'Train Time (s)': round(t_nb, 2)})

    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Exp2b LATENT: Base models trên poisoned latent (poisoned encoder)'
    )
    parser.add_argument('--poison-rates', nargs='+', type=int, default=[5, 10, 15, 50])
    parser.add_argument('--latent-dir',  default=str(LATENT_EXP10))
    parser.add_argument('--output-dir',  default=str(RESULTS_BASE))
    args = parser.parse_args()

    lat_dir = Path(args.latent_dir)
    out_dir = Path(args.output_dir)

    if not lat_dir.exists():
        print(f'\n❌ Exp10 latent data not found: {lat_dir}')
        print('   Hãy chạy trước: python pipelines/preprocessing/prepare_exp10_data.py')
        sys.exit(1)

    print('\n' + '='*80)
    print(' EXP2b LATENT: POISONING (Poisoned Encoder) '.center(80, '='))
    print('='*80)
    print("""
  So sánh với Exp2 Latent (clean encoder):
    Exp2  : RAW poisoned → CLEAN   encoder → latent
    Exp2b : RAW poisoned → POISONED encoder → latent  ← đây!

  Train trên POISONED latent (label nhiễm + feature space nhiễm)
  Test  trên POISONED latent (cùng encoder — realistic deployment)

  Δ(Exp2b − Exp2) = tác động riêng của poisoned encoder
""")

    # Đọc Exp2 results để so sánh inline
    exp2_results = {}
    exp2_dir = BASE_DIR / 'results/latent/exp2_poisoning'

    all_summary = []   # để print bảng tổng kết cuối

    for rate in args.poison_rates:
        rate_str  = f'{rate:02d}'
        data_dir  = lat_dir / f'poison_{rate_str}'

        if not (data_dir / 'X_train.npy').exists():
            print(f'\n  ⚠️  Skipping poison_{rate}% — data not found at {data_dir}')
            print(f'     Run: python pipelines/preprocessing/prepare_exp10_data.py')
            continue

        print('\n' + '='*80)
        print(f'  POISON RATE: {rate}%'.center(80))
        print('='*80)

        X_train = np.load(data_dir / 'X_train.npy')
        y_train = np.load(data_dir / 'y_train.npy')
        X_test  = np.load(data_dir / 'X_test.npy')   # encoded by POISONED encoder
        y_test  = np.load(data_dir / 'y_test.npy')
        input_dim = X_train.shape[1]

        print(f'  Train: {len(X_train):,} × {input_dim}  (poisoned encoder)')
        print(f'  Test : {len(X_test):,} × {input_dim}   (poisoned encoder — realistic)')
        print(f'  Encoder: POISONED (AE trained on poisoned labels)\n')

        # Load Exp2 reference nếu có
        exp2_csv = sorted((exp2_dir / f'poison_{rate_str}').glob('summary_*.csv'))
        exp2_ref = {}
        if exp2_csv:
            df2 = pd.read_csv(exp2_csv[-1])
            exp2_ref = {row['Model']: row['F1-Score'] for _, row in df2.iterrows()}

        # Train & evaluate
        rows = train_eval_models(X_train, y_train, X_test, y_test, input_dim)

        # Save
        rate_out = out_dir / f'poison_{rate_str}'
        rate_out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        df = pd.DataFrame(rows)
        csv_path = rate_out / f'summary_{ts}.csv'
        df.to_csv(csv_path, index=False)

        # Print summary giống Exp2
        print(f'\n  {"Model":<6} {"Accuracy":>10} {"Precision":>10}'
              f' {"Recall":>8} {"F1-Score":>9} {"Time(s)":>8}'
              + (f' {"Exp2 F1":>9} {"Δ":>8}' if exp2_ref else ''))
        print('  ' + '-' * (63 + (18 if exp2_ref else 0)))
        for row in rows:
            name = row['Model']
            f1b  = row['F1-Score']
            f1_2 = exp2_ref.get(name, None)
            delta = f'{f1b - f1_2:+.4f}' if f1_2 is not None else ''
            exp2s = f'{f1_2:.4f}' if f1_2 is not None else ''
            line = (f'  {name:<6} {row["Accuracy"]:>10.4f} {row["Precision"]:>10.4f}'
                    f' {row["Recall"]:>8.4f} {f1b:>9.4f} {row["Train Time (s)"]:>8.2f}')
            if exp2_ref:
                line += f' {exp2s:>9} {delta:>8}'
            print(line)

        print(f'\n  📁 Saved → {csv_path}')
        all_summary.append({'poison_rate': rate, 'rows': rows,
                            'exp2_ref': exp2_ref})

    # ── Bảng tổng kết cuối (giống format Exp2) ──────────────────────────
    if len(all_summary) > 1:
        print('\n' + '='*80)
        print('✅ EXP2b LATENT — TỔNG KẾT F1-Score'.center(80))
        print('='*80)
        models = [r['rows'] for r in all_summary][0]
        model_names = [r['Model'] for r in models]

        header = f'  {"Model":<6}' + ''.join(
            f'  P{s["poison_rate"]:02d}(b)  P{s["poison_rate"]:02d}(Δ)' 
            for s in all_summary
        )
        print(f'\n{header}')
        print('  ' + '-' * len(header))
        for name in model_names:
            row_txt = f'  {name:<6}'
            for s in all_summary:
                row_b = next((r for r in s['rows'] if r['Model'] == name), {})
                f1b   = row_b.get('F1-Score', float('nan'))
                f1_2  = s['exp2_ref'].get(name, None)
                delta = f'{f1b - f1_2:+.4f}' if f1_2 is not None else '     --'
                row_txt += f'   {f1b:.4f}  {delta}'
            print(row_txt)

        print(f'\n  Chú giải: (b) = Exp2b (poisoned enc)  |  (Δ) = Exp2b − Exp2')
        print(f'           Δ < 0 → poisoned encoder làm GIẢM thêm F1')
        print(f'\n  📁 Results: {out_dir}/')


if __name__ == '__main__':
    main()
