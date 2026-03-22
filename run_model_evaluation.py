import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

RANDOM_STATE = 42

def compute_metrics(y_true, y_pred):
    return {
        'Accuracy':  round(float(accuracy_score(y_true,  y_pred)), 4),
        'Precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'Recall':    round(float(recall_score(y_true,    y_pred, zero_division=0)), 4),
        'F1-Score':  round(float(f1_score(y_true,         y_pred, zero_division=0)), 4),
    }

def build_mlp(input_dim):
    model = Sequential([
        Dense(50, input_dim=input_dim, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_eval_models(X_train, y_train, X_test, y_test, input_dim, save_dir=None):
    rows = []
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        import joblib

    # ── MLP ──
    print(f'  [1/5] MLP  Dense({input_dim})→Dense(1)  epochs=30  batch=64 ...', flush=True)
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
    rows.append({'Model': 'MLP', **m, 'Time(s)': round(t_mlp, 2)})
    if save_dir:
        mlp.save(str(save_dir / 'mlp.h5'))

    # ── SVM ──
    print(f'  [2/5] SVM  LinearSVC(C=1.0) ...', flush=True)
    t0  = time.time()
    svm = LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_STATE)
    svm.fit(X_train, y_train)
    t_svm = time.time() - t0
    pred  = svm.predict(X_test)
    m     = compute_metrics(y_test, pred)
    print(f'     ✓  F1={m["F1-Score"]:.4f}  time={t_svm:.1f}s')
    rows.append({'Model': 'SVM', **m, 'Time(s)': round(t_svm, 2)})
    if save_dir:
        joblib.dump(svm, save_dir / 'svm.pkl')

    # ── RF ──
    print(f'  [3/5] RF   n_estimators=100 ...', flush=True)
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=4)
    rf.fit(X_train, y_train)
    t_rf = time.time() - t0
    pred = rf.predict(X_test)
    m    = compute_metrics(y_test, pred)
    print(f'     ✓  F1={m["F1-Score"]:.4f}  time={t_rf:.1f}s')
    rows.append({'Model': 'RF', **m, 'Time(s)': round(t_rf, 2)})
    if save_dir:
        joblib.dump(rf, save_dir / 'rf.pkl')

    # ── KNN ──
    print(f'  [4/5] KNN  n_neighbors=5 ...', flush=True)
    t0  = time.time()
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=4)
    knn.fit(X_train, y_train)
    t_knn = time.time() - t0
    pred  = knn.predict(X_test)
    m     = compute_metrics(y_test, pred)
    print(f'     ✓  F1={m["F1-Score"]:.4f}  time={t_knn:.1f}s')
    rows.append({'Model': 'KNN', **m, 'Time(s)': round(t_knn, 2)})
    if save_dir:
        joblib.dump(knn, save_dir / 'knn.pkl')

    # ── NB ──
    print(f'  [5/5] NB   GaussianNB ...', flush=True)
    t0  = time.time()
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    t_nb = time.time() - t0
    pred  = nb.predict(X_test)
    m     = compute_metrics(y_test, pred)
    print(f'     ✓  F1={m["F1-Score"]:.4f}  time={t_nb:.1f}s')
    rows.append({'Model': 'NB', **m, 'Time(s)': round(t_nb, 2)})
    if save_dir:
        joblib.dump(nb, save_dir / 'nb.pkl')

    return rows

def main():
    parser = argparse.ArgumentParser(description='Model Evaluation Script')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--exp-name', required=True)
    parser.add_argument('--models-save-dir', required=False, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)

    print('\n' + '='*80)
    print(f' {args.exp_name} '.center(80, '='))
    print('='*80)
    
    if not (data_dir / 'X_train.npy').exists():
        print(f"❌ Error: data not found at {data_dir}")
        sys.exit(1)

    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    X_test  = np.load(data_dir / 'X_test.npy')
    y_test  = np.load(data_dir / 'y_test.npy')
    
    input_dim = X_train.shape[1]
    print(f'  Train: {len(X_train):,} × {input_dim}')
    print(f'  Test : {len(X_test):,} × {input_dim}\n')

    rows = train_eval_models(X_train, y_train, X_test, y_test, input_dim, save_dir=args.models_save_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    df = pd.DataFrame(rows)
    csv_path = out_dir / f'summary_{ts}.csv'
    df.to_csv(csv_path, index=False)
    
    # Save full details to json to be consistent with other files
    json_path = out_dir / f'results_{ts}.json'
    df.to_json(json_path, orient='records', indent=4)

    header = f'\n  {"Model":<6} {"Accuracy":>8} {"Precision":>9} {"Recall":>8} {"F1-Score":>8} {"Time(s)":>8}'
    print(header)
    print('  ' + '-' * 55)

    for row in rows:
        name = row['Model']
        line = (f'  {name:<6} {row["Accuracy"]:>8.4f} {row["Precision"]:>9.4f}'
                f' {row["Recall"]:>8.4f} {row["F1-Score"]:>8.4f} {row["Time(s)"]:>8.2f}')
        print(line)

    print(f'\n  📁 Saved results to: {out_dir}/')

if __name__ == "__main__":
    main()
