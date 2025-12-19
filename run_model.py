#!/usr/bin/env python3
"""
Script chạy RIÊNG BIỆT từng model với lựa chọn dataset.
Usage: python run_model.py --model mlp --experiment 1 --dataset cicids2018

Datasets hỗ trợ:
- cicids2018: CICIDS 2018 dataset
- cicids2017: CICIDS 2017 dataset (nếu có)
- trabid: TRAbID dataset (nếu có)
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.experiment_data_preparation import load_experiment_data


def load_dataset_for_experiment(dataset_name, experiment_type):
    """Load data cho dataset và experiment cụ thể."""
    
    if dataset_name == 'cicids2018':
        return load_experiment_data(experiment_type)
    elif dataset_name == 'cicids2017':
        # TODO: Implement CICIDS2017 loader
        print(f"  [WARNING] CICIDS2017 chưa implement. Dùng CICIDS2018.")
        return load_experiment_data(experiment_type)
    elif dataset_name == 'trabid':
        # TODO: Implement TRAbID loader
        print(f"  [WARNING] TRAbID chưa implement. Dùng CICIDS2018.")
        return load_experiment_data(experiment_type)
    else:
        print(f"  [ERROR] Unknown dataset: {dataset_name}")
        print("  Available: cicids2018, cicids2017, trabid")
        raise ValueError(f"Unknown dataset: {dataset_name}")


def run_single_model(model_name, experiment_type, dataset_name='cicids2018'):
    """
    Chạy 1 model với experiment và dataset cụ thể.
    
    Args:
        model_name: 'mlp', 'svm', 'rf', 'knn', 'cnn'
        experiment_type: 1, 2, or 3
        dataset_name: 'cicids2018' (mở rộng sau)
    """
    
    print("\n" + "="*80)
    print(f"CHẠY MODEL: {model_name.upper()}")
    print(f"THỰC NGHIỆM: {experiment_type}")
    print(f"DATASET: {dataset_name.upper()}")
    print("="*80)
    
    # Determine result directory
    BASE_DIR = Path(__file__).parent.parent
    RESULTS_DIR = BASE_DIR / "results" / f"exp{experiment_type}_{get_exp_name(experiment_type)}" / model_name
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data based on experiment
    print(f"\n[1] Loading data for Experiment {experiment_type}...")
    print(f"    Dataset: {dataset_name}")
    
    if experiment_type == 1:
        # Baseline: benign + malicious (clean)
        X_train, y_train, X_test, y_test = load_dataset_for_experiment(dataset_name, 'baseline')
        print(f"    Train: {len(X_train):,} samples")
        print(f"    Test: {len(X_test):,} samples")
        
    elif experiment_type == 2:
        # Data poisoning: need to ask for poison rate
        print("\n  Chọn poison rate:")
        print("    1. 5%")
        print("    2. 10%")
        print("    3. 15%")
        choice = input("  Nhập lựa chọn (1-3): ").strip()
        poison_rates = {
            '1': 0.05,
            '2': 0.10,
            '3': 0.15
        }
        poison_rate = poison_rates.get(choice, 0.05)
        
        # Load and poison
        X_train, y_train_clean, X_test, y_test = load_dataset_for_experiment(dataset_name, 'baseline')
        
        from experiments.exp2_data_poisoning import poison_labels
        y_train = poison_labels(y_train_clean, poison_rate)
        
        print(f"    Poisoned {int(poison_rate*100)}% of labels")
        print(f"    Train: {len(X_train):,} samples")
        print(f"    Test: {len(X_test):,} samples (clean)")
        
    elif experiment_type == 3:
        print("  [WARNING] Experiment 3 (GAN Attack) chưa implement")
        print("  Vui lòng chọn Experiment 1 hoặc 2")
        return
    
    # Train model
    print(f"\n[2] Training {model_name.upper()}...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{model_name}_exp{experiment_type}_{timestamp}"
    
    # Check actual number of classes
    unique_train = np.unique(y_train)
    unique_test = np.unique(y_test)
    print(f"    Training labels: {unique_train}")
    print(f"    Test labels: {unique_test}")
    is_binary = len(unique_train) == 2 and len(unique_test) == 2
    avg_type = 'binary' if is_binary else 'weighted'
    
    if model_name == 'mlp':
        import time
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.callbacks import Callback, EarlyStopping
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Flatten y if needed
        y_train_flat = np.array(y_train).flatten()
        y_test_flat = np.array(y_test).flatten()
        
        # Custom callback to show progress
        class TrainingProgress(Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % 10 == 0:  # Print summary every 10 epochs
                    print(f"\n  📊 Epoch {epoch+1} Summary:")
                    print(f"     Train Loss: {logs['loss']:.4f} | Train Acc: {logs['accuracy']:.4f}")
                    print(f"     Val Loss:   {logs['val_loss']:.4f} | Val Acc:   {logs['val_accuracy']:.4f}")
        
        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,  # Reduced for faster stopping
            restore_best_weights=True,
            verbose=1
        )
        
        # Create deeper model
        n_classes = len(np.unique(y_train_flat))
        print(f"\n    Number of classes: {n_classes}")
        print(f"    Classes: {np.unique(y_train_flat)}")
        
        mlp = Sequential()
        mlp.add(Dense(units=128, input_dim=X_train.shape[1], activation='relu'))
        mlp.add(Dropout(0.3))
        mlp.add(Dense(units=64, activation='relu'))
        mlp.add(Dropout(0.2))
        mlp.add(Dense(units=n_classes if n_classes > 2 else 1, activation='softmax' if n_classes > 2 else 'sigmoid'))
        
        loss_fn = 'sparse_categorical_crossentropy' if n_classes > 2 else 'binary_crossentropy'
        mlp.compile(
            loss=loss_fn,
            optimizer='adam',
            metrics=['accuracy']
        )
        
        print(f"\n    🔧 Model Configuration:")
        print(f"       Architecture: Input({X_train.shape[1]}) → Dense(128) → Dropout(0.3)")
        print(f"                     → Dense(64) → Dropout(0.2) → Output({n_classes if n_classes > 2 else 1})")
        print(f"       Total params: {mlp.count_params():,}")
        print(f"\n    📚 Training Configuration:")
        print(f"       Task: {'Binary' if n_classes == 2 else 'Multi-class'} classification ({n_classes} classes)")
        print(f"       Epochs: 200 (with early stopping patience=30)")
        print(f"       Batch size: 8")
        print(f"       Validation split: 0.2 ({int(len(X_train)*0.2):,} samples)")
        print(f"       Train samples: {int(len(X_train)*0.8):,}")
        print(f"\n    🏃 Starting training...")
        print(f"    (Deeper model + smaller batch = slower, more thorough learning)\n")
        
        # Train
        start = time.time()
        history = mlp.fit(
            X_train, y_train_flat,
            epochs=200,
            batch_size=8,  # Very small batch for slower, more granular training
            validation_split=0.2,
            verbose=2,
            callbacks=[TrainingProgress(), early_stop]
        )
        train_time = time.time() - start
        
        # Training summary
        print(f"\n    ✓ Training completed in {train_time:.2f}s ({train_time/60:.2f} minutes)")
        print(f"\n    📈 Training History Summary:")
        print(f"       Final Train Loss: {history.history['loss'][-1]:.4f}")
        print(f"       Final Train Acc:  {history.history['accuracy'][-1]:.4f}")
        print(f"       Final Val Loss:   {history.history['val_loss'][-1]:.4f}")
        print(f"       Final Val Acc:    {history.history['val_accuracy'][-1]:.4f}")
        print(f"       Best Val Acc:     {max(history.history['val_accuracy']):.4f} (epoch {np.argmax(history.history['val_accuracy'])+1})")
        
        # Evaluate
        print(f"\n    🧪 Evaluating on test set ({len(X_test):,} samples)...")
        test_results = mlp.evaluate(X_test, y_test_flat, verbose=0)
        print(f"       Test Loss:     {test_results[0]:.4f}")
        print(f"       Test Accuracy: {test_results[1]:.4f}")
        
        # Predict - different logic for binary vs multi-class
        y_pred_proba = mlp.predict(X_test, verbose=0)
        
        if n_classes == 2:
            # Binary classification: use threshold 0.5
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            avg_type = 'binary'
        else:
            # Multi-class: use argmax
            y_pred = np.argmax(y_pred_proba, axis=1)
            avg_type = 'weighted'
        
        # Calculate metrics
        results = {
            'model': model_name,
            'experiment': experiment_type,
            'dataset': dataset_name,
            'n_classes': int(n_classes),
            'accuracy': accuracy_score(y_test_flat, y_pred),
            'precision': precision_score(y_test_flat, y_pred, average=avg_type, zero_division=0),
            'recall': recall_score(y_test_flat, y_pred, average=avg_type, zero_division=0),
            'f1_score': f1_score(y_test_flat, y_pred, average=avg_type, zero_division=0),
            'train_time_seconds': train_time,
            'training_history': {
                'loss': [float(x) for x in history.history['loss']],
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']]
            }
        }
        
    elif model_name == 'svm':
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
        import time
        
        print(f"    Creating SVM (kernel='linear', gamma='auto')...")
        # Using SVC instead of LinearSVC for more detailed multi-class support
        clf = SVC(kernel='linear', gamma='auto', verbose=True, random_state=42)
        
        print(f"    Training on {len(X_train):,} samples...")
        print(f"    (SVC with linear kernel - slower but more detailed than LinearSVC)")
        print(f"    (This may take 5-15 minutes depending on dataset size...)")
        start = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start
        print(f"    ✓ Training completed in {train_time:.2f} seconds ({train_time/60:.1f} minutes)")
        
        print(f"    Predicting...")
        y_pred = clf.predict(X_test)
        
        # Calculate metrics
        n_classes = len(np.unique(y_train))
        avg_type = 'binary' if n_classes == 2 else 'weighted'
        
        acc = accuracy_score(y_test, y_pred) * 100
        print(f"\n    SVM Accuracy: {acc:.2f}%")
        
        # Classification report
        print(f"\n    Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        results = {
            'model': model_name,
            'experiment': experiment_type,
            'dataset': dataset_name,
            'n_classes': int(n_classes),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average=avg_type, zero_division=0),
            'recall': recall_score(y_test, y_pred, average=avg_type, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average=avg_type, zero_division=0),
            'train_time_seconds': train_time
        }
        
    elif model_name == 'rf':
        from models.baselines.random_forest import train_random_forest
        results = train_random_forest(X_train, y_train, X_test, y_test, run_id, n_estimators=500)
        
    elif model_name == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        import time
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        print(f"    Creating KNN (k=5)...")
        clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        
        print(f"    Indexing {len(X_train):,} samples...")
        print(f"    (KNN doesn't train, just indexes data - fast)")
        start = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start
        print(f"    ✓ Indexing completed in {train_time:.2f} seconds")
        
        print(f"    Predicting on {len(X_test):,} samples...")
        print(f"    (This will be slow - O(n) per prediction)")
        y_pred = clf.predict(X_test)
        
        # Determine if binary or multi-class
        n_classes = len(np.unique(y_train))
        avg_type = 'binary' if n_classes == 2 else 'weighted'
        
        results = {
            'model': model_name,
            'experiment': experiment_type,
            'dataset': dataset_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average=avg_type, zero_division=0),
            'recall': recall_score(y_test, y_pred, average=avg_type, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average=avg_type, zero_division=0),
            'train_time_seconds': train_time
        }
    
    elif model_name == 'lda':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
        import time
        
        print(f"    Creating LDA (Linear Discriminant Analysis)...")
        n_classes = len(np.unique(y_train))
        print(f"    Classes: {n_classes}")
        
        # LDA has max n_classes-1 components
        n_components = min(n_classes - 1, X_train.shape[1])
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        
        print(f"    Training on {len(X_train):,} samples...")
        print(f"    Dimensionality reduction: {X_train.shape[1]} → {n_components} components")
        start = time.time()
        lda.fit(X_train, y_train)
        train_time = time.time() - start
        print(f"    ✓ Training completed in {train_time:.2f} seconds")
        
        print(f"    Predicting...")
        y_pred = lda.predict(X_test)
        
        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred) * 100
        print(f"\n    LDA Accuracy: {acc:.2f}%")
        
        # Determine average type
        avg_type = 'binary' if n_classes == 2 else 'weighted'
        
        # Classification report (verbose)
        print(f"\n    Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        results = {
            'model': model_name,
            'experiment': experiment_type,
            'dataset': dataset_name,
            'n_classes': int(n_classes),
            'n_components': int(n_components),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average=avg_type, zero_division=0),
            'recall': recall_score(y_test, y_pred, average=avg_type, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average=avg_type, zero_division=0),
            'train_time_seconds': train_time
        }
        
    elif model_name == 'cnn':
        from models.deep_learning.cnn_classifier import train_cnn_classifier
        results = train_cnn_classifier(X_train, y_train, X_test, y_test, run_id, epochs=30, batch_size=8)
        
    else:
        print(f"  [ERROR] Unknown model: {model_name}")
        print("  Available: mlp, svm, rf, knn, cnn")
        return
    
    # Save results
    print(f"\n[3] Saving results to: {RESULTS_DIR}")
    
    # Summary
    summary_file = RESULTS_DIR / f"summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # CSV for easy viewing
    import pandas as pd
    summary_csv = RESULTS_DIR / f"summary_{timestamp}.csv"
    pd.DataFrame([results]).to_csv(summary_csv, index=False)
    
    print(f"  ✓ Summary: {summary_csv}")
    
    # Print results
    print(f"\n[4] RESULTS:")
    print(f"  Accuracy:  {results.get('accuracy', 'N/A'):.4f}")
    print(f"  Precision: {results.get('precision', 'N/A'):.4f}")
    print(f"  Recall:    {results.get('recall', 'N/A'):.4f}")
    print(f"  F1-Score:  {results.get('f1_score', 'N/A'):.4f}")
    print(f"  Train Time: {results.get('train_time_seconds', results.get('train_time', 'N/A')):.2f}s")
    
    print(f"\n{'='*80}")
    print("✓ Hoàn thành!")
    print(f"{'='*80}\n")
    
    return results


def get_exp_name(exp_num):
    """Get experiment folder name."""
    names = {
        1: 'baseline',
        2: 'poisoning',
        3: 'gan'
    }
    return names.get(exp_num, 'unknown')


def main():
    parser = argparse.ArgumentParser(description='Chạy 1 model cho 1 experiment')
    parser.add_argument('--model', '-m', required=True, choices=['mlp', 'svm', 'rf', 'knn', 'cnn'],
                        help='Model to train')
    parser.add_argument('--experiment', '-e', type=int, required=True, choices=[1, 2, 3],
                        help='Experiment number (1=baseline, 2=poisoning, 3=gan)')
    parser.add_argument('--dataset', '-d', default='cicids2018',
                        help='Dataset name (default: cicids2018)')
    
    args = parser.parse_args()
    
    run_single_model(args.model, args.experiment, args.dataset)


if __name__ == '__main__':
    main()
