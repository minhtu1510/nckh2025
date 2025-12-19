#!/usr/bin/env python3
"""
BASELINE - 6 MODELS trên CICIDS2017

Models:
1. MLP (Multi-Layer Perceptron)
2. LSVM (Linear SVM)
3. QSVM (Quadratic SVM / RBF SVM)
4. KNN (K-Nearest Neighbors)
5. RF (Random Forest)
6. AE-MLP (Autoencoder + MLP)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Try TensorFlow for deep learning models
try:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not available - will skip MLP and AE-MLP")

# Try scikit-learn for traditional ML
try:
    from sklearn.svm import LinearSVC, SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import classification_report, confusion_matrix
    SKL_AVAILABLE = True
except ImportError:
    SKL_AVAILABLE = False
    print("⚠️  scikit-learn not available - will skip LSVM, QSVM, KNN, RF")

if not TF_AVAILABLE and not SKL_AVAILABLE:
    print("❌ No ML libraries available! Install tensorflow or scikit-learn")
    exit(1)

# Paths
BASE_DIR = Path(__file__).resolve().parent
SPLITS_DIR = BASE_DIR / "datasets" / "splits" / "cicids2017"
RESULTS_DIR = BASE_DIR / "results" / "baseline_6models_cicids2017"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ===== LOAD DATA =====
print("="*80)
print("📂 LOADING CICIDS2017 DATA (46 features)")
print("="*80)

X_train = np.load(SPLITS_DIR / "train_X.npy")
y_train = np.load(SPLITS_DIR / "train_y.npy")
X_test = np.load(SPLITS_DIR / "test_X.npy")
y_test = np.load(SPLITS_DIR / "test_y.npy")

print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")
print(f"✓ Features: {X_train.shape[1]}")
print()

results = {}

# ===== MODEL 1: MLP =====
if TF_AVAILABLE:
    print("="*80)
    print("🧠 MODEL 1: MLP (Multi-Layer Perceptron)")
    print("="*80)
    
    mlp = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    mlp.fit(X_train, y_train, epochs=30, batch_size=64, verbose=0, validation_split=0.1)
    
    y_pred = (mlp.predict(X_test, verbose=0) > 0.5).astype(int).ravel()
    
    results['MLP'] = {
        'accuracy': accuracy_score(y_test, y_pred) if SKL_AVAILABLE else 0,
        'precision': precision_score(y_test, y_pred) if SKL_AVAILABLE else 0,
        'recall': recall_score(y_test, y_pred) if SKL_AVAILABLE else 0,
        'f1': f1_score(y_test, y_pred) if SKL_AVAILABLE else 0,
    }
    print(f"✓ Accuracy: {results['MLP']['accuracy']:.4f}, F1: {results['MLP']['f1']:.4f}")
    print()

# ===== MODEL 2: Linear SVM =====
if SKL_AVAILABLE:
    print("="*80)
    print("⚖️  MODEL 2: LSVM (Linear SVM)")
    print("="*80)
    
    lsvm = LinearSVC(max_iter=1000, random_state=42)
    lsvm.fit(X_train, y_train)
    y_pred = lsvm.predict(X_test)
    
    results['LSVM'] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
    }
    print(f"✓ Accuracy: {results['LSVM']['accuracy']:.4f}, F1: {results['LSVM']['f1']:.4f}")
    print()

# ===== MODEL 3: RBF SVM (Quadratic-like) =====
if SKL_AVAILABLE:
    print("="*80)
    print("🔮 MODEL 3: QSVM (RBF SVM)")
    print("="*80)
    
    # Use subset for faster training
    sample_size = min(5000, len(X_train))
    sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
    
    qsvm = SVC(kernel='rbf', gamma='scale', random_state=42)
    qsvm.fit(X_train[sample_idx], y_train[sample_idx])
    y_pred = qsvm.predict(X_test)
    
    results['QSVM'] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
    }
    print(f"✓ Accuracy: {results['QSVM']['accuracy']:.4f}, F1: {results['QSVM']['f1']:.4f}")
    print(f"  (trained on {sample_size} samples for speed)")
    print()

# ===== MODEL 4: KNN =====
if SKL_AVAILABLE:
    print("="*80)
    print("👥 MODEL 4: KNN (K-Nearest Neighbors)")
    print("="*80)
    
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    results['KNN'] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
    }
    print(f"✓ Accuracy: {results['KNN']['accuracy']:.4f}, F1: {results['KNN']['f1']:.4f}")
    print()

# ===== MODEL 5: Random Forest =====
if SKL_AVAILABLE:
    print("="*80)
    print("🌲 MODEL 5: RF (Random Forest)")
    print("="*80)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    results['RF'] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
    }
    print(f"✓ Accuracy: {results['RF']['accuracy']:.4f}, F1: {results['RF']['f1']:.4f}")
    print()

# ===== MODEL 6: Autoencoder + MLP =====
if TF_AVAILABLE:
    print("="*80)
    print("🔐 MODEL 6: AE-MLP (Autoencoder + MLP)")
    print("="*80)
    
    # Train autoencoder
    latent_dim = 16
    
    input_layer = Input(shape=(X_train.shape[1],))
    encoded = Dense(32, activation='relu')(input_layer)
    encoded = Dense(latent_dim, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(X_train.shape[1], activation='sigmoid')(decoded)
    
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_train, X_train, epochs=20, batch_size=256, verbose=0, validation_split=0.1)
    
    # Extract latent features
    X_train_latent = encoder.predict(X_train, verbose=0)
    X_test_latent = encoder.predict(X_test, verbose=0)
    
    # Train classifier on latent
    ae_mlp = Sequential([
        Dense(16, activation='relu', input_dim=latent_dim),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    ae_mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    ae_mlp.fit(X_train_latent, y_train, epochs=30, batch_size=64, verbose=0, validation_split=0.1)
    
    y_pred = (ae_mlp.predict(X_test_latent, verbose=0) > 0.5).astype(int).ravel()
    
    results['AE-MLP'] = {
        'accuracy': accuracy_score(y_test, y_pred) if SKL_AVAILABLE else 0,
        'precision': precision_score(y_test, y_pred) if SKL_AVAILABLE else 0,
        'recall': recall_score(y_test, y_pred) if SKL_AVAILABLE else 0,
        'f1': f1_score(y_test, y_pred) if SKL_AVAILABLE else 0,
    }
    print(f"✓ Accuracy: {results['AE-MLP']['accuracy']:.4f}, F1: {results['AE-MLP']['f1']:.4f}")
    print(f"  (latent dim: {latent_dim})")
    print()

# ===== SAVE RESULTS =====
print("="*80)
print("💾 SAVING RESULTS")
print("="*80)

results_df = pd.DataFrame(results).T
results_df.index.name = 'model'
results_path = RESULTS_DIR / f"all_models_metrics_{TIMESTAMP}.csv"
results_df.to_csv(results_path)
print(f"✓ Saved: {results_path}")
print()

# ===== SUMMARY =====
print("="*80)
print("📊 RESULTS SUMMARY")
print("="*80)
print(results_df.to_string())
print()

# Best model
best_accuracy = results_df['accuracy'].max()
best_f1 = results_df['f1'].max()
best_model_acc = results_df['accuracy'].idxmax()
best_model_f1 = results_df['f1'].idxmax()

print(f"🏆 Best Accuracy: {best_model_acc} ({best_accuracy:.4f})")
print(f"🏆 Best F1 Score: {best_model_f1} ({best_f1:.4f})")
print()

print("="*80)
print("✅ ALL MODELS COMPLETE!")
print("="*80)
