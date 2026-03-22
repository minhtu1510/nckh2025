import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model

def compute_metrics(y_true, y_pred):
    return {
        'Accuracy':  round(float(accuracy_score(y_true,  y_pred)), 4),
        'Precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'Recall':    round(float(recall_score(y_true,    y_pred, zero_division=0)), 4),
        'F1-Score':  round(float(f1_score(y_true,         y_pred, zero_division=0)), 4),
    }

def main():
    parser = argparse.ArgumentParser(description='Ensemble Evaluation Script')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--models-load-dir', required=True)
    parser.add_argument('--exp-name', required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    models_dir = Path(args.models_load_dir)

    print('\n' + '='*80)
    print(f' {args.exp_name} '.center(80, '='))
    print('='*80)
    
    if not (data_dir / 'X_test.npy').exists():
        print(f"❌ Error: data not found at {data_dir}")
        sys.exit(1)

    X_test  = np.load(data_dir / 'X_test.npy')
    y_test  = np.load(data_dir / 'y_test.npy')
    
    input_dim = X_test.shape[1]
    print(f'  Test : {len(X_test):,} × {input_dim}\n')
    
    print(f'  [+] Loading models from {models_dir}...')
    try:
        mlp = load_model(models_dir / 'mlp.h5')
        svm = joblib.load(models_dir / 'svm.pkl')
        rf = joblib.load(models_dir / 'rf.pkl')
        knn = joblib.load(models_dir / 'knn.pkl')
        nb = joblib.load(models_dir / 'nb.pkl')
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        sys.exit(1)

    models = {'MLP': mlp, 'SVM': svm, 'RF': rf, 'KNN': knn, 'NB': nb}
    rows = []
    
    print(f'  [+] Evaluating Individual Models...')
    probas = {}
    preds = {}
    metrics_log = {}
    for name, model in models.items():
        if name == 'MLP':
            prob = mlp.predict(X_test, verbose=0).flatten()
            pred = (prob >= 0.5).astype(int)
            # SVM outputting decision_function scaled roughly or just raw prob
            # we use 0.5 as prob threshold.
            prob_arr = np.vstack([1 - prob, prob]).T
        else:
            pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                prob_arr = model.predict_proba(X_test)
            else:
                # SVM doesn't have predict_proba natively with LinearSVC mostly
                d = model.decision_function(X_test)
                # Platt scaling approximation using sigmoid
                p = 1 / (1 + np.exp(-d))
                prob_arr = np.vstack([1 - p, p]).T
                
        metrics = compute_metrics(y_test, pred)
        metrics_log[name] = metrics
        rows.append({'Model': name, **metrics})
        probas[name] = prob_arr
        preds[name] = pred
        print(f"     ✓ {name:4s} F1={metrics['F1-Score']:.4f}")

    print(f'\n  [+] Evaluating Ensembles...')
    
    # 1. Ensemble Hard Voting
    all_preds = np.array([preds['MLP'], preds['SVM'], preds['RF'], preds['KNN'], preds['NB']])
    # Majority vote
    hard_pred = (np.sum(all_preds, axis=0) >= 3).astype(int)
    m_hard = compute_metrics(y_test, hard_pred)
    rows.append({'Model': 'Ensemble Hard', **m_hard})
    print(f"     ✓ Hard Voting     F1={m_hard['F1-Score']:.4f}")
    
    # 2. Ensemble Soft Voting
    all_probs = np.array([probas['MLP'][:, 1], probas['SVM'][:, 1], probas['RF'][:, 1], probas['KNN'][:, 1], probas['NB'][:, 1]])
    soft_prob = np.mean(all_probs, axis=0)
    soft_pred = (soft_prob >= 0.5).astype(int)
    m_soft = compute_metrics(y_test, soft_pred)
    rows.append({'Model': 'Ensemble Soft', **m_soft})
    print(f"     ✓ Soft Voting     F1={m_soft['F1-Score']:.4f}")
    
    # 3. Ensemble Weighted Soft Voting (Weights based on individual F1-score)
    weights = np.array([metrics_log[name]['F1-Score'] for name in ['MLP', 'SVM', 'RF', 'KNN', 'NB']])
    weights = weights / np.sum(weights) # Normalize to 1
    weighted_prob = np.average(all_probs, axis=0, weights=weights)
    weighted_pred = (weighted_prob >= 0.5).astype(int)
    m_weighted = compute_metrics(y_test, weighted_pred)
    rows.append({'Model': 'Ensemble Weighted Soft', **m_weighted})
    print(f"     ✓ Weighted Soft   F1={m_weighted['F1-Score']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    df = pd.DataFrame(rows)
    csv_path = out_dir / f'summary_{ts}.csv'
    df.to_csv(csv_path, index=False)
    
    json_path = out_dir / f'results_{ts}.json'
    df.to_json(json_path, orient='records', indent=4)

    header = f'\n  {"Model":<25} {"Accuracy":>8} {"Precision":>9} {"Recall":>8} {"F1-Score":>8}'
    print(header)
    print('  ' + '-' * 65)

    for row in rows:
        name = row['Model']
        line = (f'  {name:<25} {row["Accuracy"]:>8.4f} {row["Precision"]:>9.4f}'
                f' {row["Recall"]:>8.4f} {row["F1-Score"]:>8.4f}')
        print(line)

    print(f'\n  📁 Saved results to: {out_dir}/')

if __name__ == "__main__":
    main()
