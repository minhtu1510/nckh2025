"""
Script chung để train và evaluate 5 models trên bất kỳ dataset nào.
Dùng chung cho cả 3 thực nghiệm.

Usage:
    python run_model_evaluation.py --data-dir datasets/splits/exp1_baseline --output-dir results/exp1_baseline --exp-name "Baseline"
    python run_model_evaluation.py --data-dir datasets/splits/exp2_poisoning/poison_05 --output-dir results/exp2_poison_05 --exp-name "Poisoning 5%"
    python run_model_evaluation.py --data-dir datasets/splits/exp3_gan --output-dir results/exp3_gan --exp-name "GAN Attack"
"""

import argparse
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# TensorFlow imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Reshape, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import Callback


def load_data(data_dir):
    """
    Load training and test data from directory.
    Expected files: X_train.npy, y_train.npy, X_test.npy, y_test.npy
    """
    data_dir = Path(data_dir)
    
    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    X_test = np.load(data_dir / 'X_test.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    
    print(f"📊 Loaded data from: {data_dir}")
    print(f"  Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"    - Benign: {(y_train==0).sum():,}")
    print(f"    - Malicious: {(y_train==1).sum():,}")
    print(f"  Test: {X_test.shape[0]:,} samples")
    print(f"    - Benign: {(y_test==0).sum():,}")
    print(f"    - Malicious: {(y_test==1).sum():,}")
    
    return X_train, y_train, X_test, y_test


class TrainingProgressCallback(Callback):
    """Custom callback để hiển thị progress mỗi 10 epochs."""
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            print(f"  📊 Epoch {epoch+1}: "
                  f"Loss={logs['loss']:.4f}, Acc={logs['accuracy']:.4f}, "
                  f"Val_Loss={logs['val_loss']:.4f}, Val_Acc={logs['val_accuracy']:.4f}")


def train_mlp(X_train, y_train, X_test, y_test):
    """Train MLP classifier."""
    print(f"\n{'='*80}")
    print("[MODEL 1/5] Multi-Layer Perceptron (MLP)")
    print(f"{'='*80}")
    
    # Flatten y
    y_train = np.array(y_train).flatten()
    y_test = np.array(y_test).flatten()
    
    # Build model
    model = Sequential([
        Dense(50, input_dim=X_train.shape[1], activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print(f"  Architecture: Input({X_train.shape[1]}) → Dense(50) → Dense(1)")
    print(f"  Total params: {model.count_params():,}")
    print(f"  Training for 30 epochs, batch_size=8...")
    
    start = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=64,
        validation_split=0.2,
        verbose=0,
        callbacks=[TrainingProgressCallback()]
    )
    train_time = time.time() - start
    
    # Predict
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'train_time': train_time,
    }
    
    print(f"  ✓ Completed in {train_time:.2f}s - Accuracy: {results['accuracy']:.4f}")
    return results


def train_svm(X_train, y_train, X_test, y_test):
    """Train SVM classifier."""
    print(f"\n{'='*80}")
    print("[MODEL 2/5] Support Vector Machine (SVM)")
    print(f"{'='*80}")
    
    clf = LinearSVC(C=1.0, max_iter=3000, random_state=42, verbose=0)
    
    print(f"  Configuration: LinearSVC(C=1.0, max_iter=3000)")
    print(f"  Training...")
    
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start
    
    y_pred = clf.predict(X_test)
    
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'train_time': train_time,
    }
    
    print(f"  ✓ Completed in {train_time:.2f}s - Accuracy: {results['accuracy']:.4f}")
    return results


def train_rf(X_train, y_train, X_test, y_test):
    """Train Random Forest classifier."""
    print(f"\n{'='*80}")
    print("[MODEL 3/5] Random Forest (RF)")
    print(f"{'='*80}")
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=4, verbose=0)
    
    print(f"  Configuration: n_estimators=100, n_jobs=4")
    print(f"  Training...")
    
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start
    
    y_pred = clf.predict(X_test)
    
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'train_time': train_time,
    }
    
    print(f"  ✓ Completed in {train_time:.2f}s - Accuracy: {results['accuracy']:.4f}")
    return results


def train_knn(X_train, y_train, X_test, y_test):
    """Train KNN classifier."""
    print(f"\n{'='*80}")
    print("[MODEL 4/5] K-Nearest Neighbors (KNN)")
    print(f"{'='*80}")
    
    clf = KNeighborsClassifier(n_neighbors=5, n_jobs=4)
    
    print(f"  Configuration: n_neighbors=5, n_jobs=4")
    print(f"  Training...")
    
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start
    
    y_pred = clf.predict(X_test)
    
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'train_time': train_time,
    }
    
    print(f"  ✓ Completed in {train_time:.2f}s - Accuracy: {results['accuracy']:.4f}")
    return results


def train_cnn(X_train, y_train, X_test, y_test):
    """Train CNN classifier."""
    print(f"\n{'='*80}")
    print("[MODEL 5/5] Convolutional Neural Network (CNN)")
    print(f"{'='*80}")
    
    # Flatten y
    y_train = np.array(y_train).flatten()
    y_test = np.array(y_test).flatten()
    
    # Reshape for CNN: (samples, features, 1)
    X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Build model
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print(f"  Architecture: Conv1D → MaxPool → Dense(64) → Dense(1)")
    print(f"  Total params: {model.count_params():,}")
    print(f"  Training for 15 epochs, batch_size=256...")
    
    start = time.time()
    history = model.fit(
        X_train_cnn, y_train,
        epochs=15,
        batch_size=256,
        validation_split=0.2,
        verbose=0,
        callbacks=[TrainingProgressCallback()]
    )
    train_time = time.time() - start
    
    # Predict
    y_pred_proba = model.predict(X_test_cnn, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'train_time': train_time,
    }
    
    print(f"  ✓ Completed in {train_time:.2f}s - Accuracy: {results['accuracy']:.4f}")
    return results


def run_evaluation(data_dir, output_dir, exp_name, models_to_run=None):
    """
    Main evaluation function - train selected models.
    
    Args:
        data_dir: Directory containing X_train.npy, y_train.npy, X_test.npy, y_test.npy
        output_dir: Directory to save results
        exp_name: Experiment name for display
        models_to_run: List of model names to run (e.g., ['mlp', 'svm']). If None, run all.
    """
    print("\n" + "="*80)
    print(f"{exp_name.center(80)}")
    print("="*80)
    
    # Available models
    available_models = {
        'mlp': train_mlp,
        'svm': train_svm,
        'rf': train_rf,
        'knn': train_knn,
        'cnn': train_cnn,
    }
    
    # Determine which models to run
    if models_to_run is None:
        models_to_run = list(available_models.keys())
    else:
        # Validate model names
        models_to_run = [m.lower() for m in models_to_run]
        invalid = [m for m in models_to_run if m not in available_models]
        if invalid:
            print(f"❌ Invalid model names: {invalid}")
            print(f"   Available: {list(available_models.keys())}")
            sys.exit(1)
    
    print(f"\n📋 Models to run: {', '.join([m.upper() for m in models_to_run])}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n[STEP 1] Loading data...")
    X_train, y_train, X_test, y_test = load_data(data_dir)
    
    # Train selected models
    print(f"\n[STEP 2] Training models...")
    
    all_results = {}
    
    for idx, model_name in enumerate(models_to_run, 1):
        try:
            print(f"\n{'='*80}")
            print(f"[MODEL {idx}/{len(models_to_run)}] {model_name.upper()}")
            print(f"{'='*80}")
            
            train_func = available_models[model_name]
            all_results[model_name.upper()] = train_func(X_train, y_train, X_test, y_test)
        except Exception as e:
            print(f"  ✗ {model_name.upper()} failed: {e}")
            all_results[model_name.upper()] = {'error': str(e)}
    
    # Save results
    print(f"\n[STEP 3] Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create summary table
    summary_data = []
    for model_name, results in all_results.items():
        if 'error' not in results:
            summary_data.append({
                'Model': model_name,
                'Accuracy': f"{results['accuracy']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}",
                'Train Time (s)': f"{results['train_time']:.2f}",
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / f"summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    
    # Save detailed JSON
    results_json_path = output_dir / f"results_{timestamp}.json"
    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {exp_name}")
    print(f"{'='*80}")
    print(summary_df.to_string(index=False))
    print(f"\n✅ Results saved:")
    print(f"  - {summary_path}")
    print(f"  - {results_json_path}")
    print(f"{'='*80}\n")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate models on any dataset')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing X_train.npy, y_train.npy, X_test.npy, y_test.npy')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--exp-name', type=str, default='Experiment',
                        help='Experiment name for display')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Models to run (mlp, svm, rf, knn, cnn). Default: all')
    
    args = parser.parse_args()
    
    run_evaluation(args.data_dir, args.output_dir, args.exp_name, args.models)


if __name__ == '__main__':
    main()
