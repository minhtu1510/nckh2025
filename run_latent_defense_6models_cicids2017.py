#!/usr/bin/env python3
"""
LATENT DEFENSE - 6 MODELS trên CICIDS2017

Workflow:
1. Train 6 models trên TRAIN_LATENT (32-dim compressed)
2. Test models trên TEST ORIGINAL (46-dim)  
3. GAN attack trên test original
4. Compare: Latent Defense vs Baseline robustness

Hypothesis: Models trained on latent space will be more robust to adversarial attacks
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Concatenate, Dense, Input, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

np.random.seed(42)
tf.random.set_seed(42)

# Paths
BASE_DIR = Path(__file__).resolve().parent
SPLITS_DIR = BASE_DIR / "datasets" / "splits" / "cicids2017"
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "latent_defense_6models_cicids2017"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

print("="*80)
print("🛡️  LATENT DEFENSE - 6 MODELS - CICIDS2017")
print("="*80)
print()

# ===== LOAD DATA =====
print("📂 Step 1: Loading data...")

# Load ORIGINAL test data (46-dim)
X_test_original = np.load(SPLITS_DIR / "test_X.npy")
y_test = np.load(SPLITS_DIR / "test_y.npy")
test_mal_X_original = np.load(SPLITS_DIR / "test_malicious_X.npy")

# Load LATENT train data (32-dim) - find latest
latent_files = list(PROCESSED_DIR.glob("cicids2017_train_latent_X_*.npy"))
if not latent_files:
    print("❌ No latent files found! Please run: python extract_train_latent_combined.py")
    exit(1)

latest_latent = sorted(latent_files)[-1]
print(f"✓ Using latent file: {latest_latent.name}")

X_train_latent = np.load(latest_latent)
y_train_latent_file = latest_latent.parent / latest_latent.name.replace("_X_", "_y_")
y_train = np.load(y_train_latent_file)

print(f"✓ Train LATENT: {X_train_latent.shape} (compressed)")
print(f"✓ Test ORIGINAL: {X_test_original.shape} (original)")
print()

latent_dim = X_train_latent.shape[1]
feature_dim = X_test_original.shape[1]

# ===== TRAIN 6 MODELS ON LATENT =====
print("="*80)
print(f"🧠 Step 2: Train 6 models on LATENT space ({latent_dim}-dim)")
print("="*80)
print()

models_latent = {}

# Model 1: MLP on latent
print("Training MLP on latent...")
mlp = Sequential([
    Dense(32, activation='relu', input_dim=latent_dim),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
mlp.fit(X_train_latent, y_train, epochs=30, batch_size=64, verbose=0, validation_split=0.1)
models_latent['MLP'] = mlp
print("✓ MLP trained")

# Model 2: LSVM on latent  
print("Training LSVM on latent...")
lsvm = LinearSVC(max_iter=1000, random_state=42)
lsvm.fit(X_train_latent, y_train)
models_latent['LSVM'] = lsvm
print("✓ LSVM trained")

# Model 3: QSVM on latent (full dataset, latent is smaller)
print("Training QSVM on latent...")
qsvm = SVC(kernel='rbf', gamma='scale', random_state=42)
qsvm.fit(X_train_latent, y_train)
models_latent['QSVM'] = qsvm
print("✓ QSVM trained")

# Model 4: KNN on latent
print("Training KNN on latent...")
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train_latent, y_train)
models_latent['KNN'] = knn
print("✓ KNN trained")

# Model 5: RF on latent
print("Training RF on latent...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_latent, y_train)
models_latent['RF'] = rf
print("✓ RF trained")

# Model 6: Simple MLP (no extra AE, since we're already in latent)
print("Training Deep MLP on latent...")
deep_mlp = Sequential([
    Dense(64, activation='relu', input_dim=latent_dim),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
deep_mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
deep_mlp.fit(X_train_latent, y_train, epochs=30, batch_size=64, verbose=0, validation_split=0.1)
models_latent['DeepMLP'] = deep_mlp
print("✓ DeepMLP trained")
print()

# ===== TRAIN AUTOENCODER (for encoding test data) =====
print("="*80)
print(f"🔐 Step 3: Load pre-trained Autoencoder for encoding")
print("="*80)
print()

# Find latest encoder from metadata
metadata_files = list(PROCESSED_DIR.glob("cicids2017_train_latent_metadata_*.json"))
if not metadata_files:
    print("❌ No metadata found! Please run: python extract_train_latent_combined.py")
    exit(1)

latest_metadata = sorted(metadata_files)[-1]
print(f"✓ Using metadata: {latest_metadata.name}")

import json
with open(latest_metadata, 'r') as f:
    metadata = json.load(f)

encoder_model_path = metadata.get('encoder_model')
if not encoder_model_path or not Path(encoder_model_path).exists():
    print("❌ Encoder model not found in metadata!")
    print(f"   Expected: {encoder_model_path}")
    exit(1)

print(f"✓ Loading encoder from: {Path(encoder_model_path).name}")
encoder = tf.keras.models.load_model(encoder_model_path)
print("✓ Encoder loaded successfully")
print()

# ===== TEST ON ORIGINAL DATA =====
print("="*80)
print(f"📊 Step 4: Test models on ORIGINAL test data ({feature_dim}-dim)")
print("="*80)
print()

# Encode test data to latent
X_test_latent = encoder.predict(X_test_original, verbose=0)
print(f"✓ Encoded test data: {X_test_original.shape} → {X_test_latent.shape}")
print()

baseline_results = {}

# Evaluate each model
for name, model in models_latent.items():
    if name in ['MLP', 'DeepMLP']:
        y_pred = (model.predict(X_test_latent, verbose=0) > 0.5).astype(int).ravel()
    else:
        y_pred = model.predict(X_test_latent)
    
    baseline_results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    print(f"✓ {name}: Acc={baseline_results[name]['accuracy']:.4f}, F1={baseline_results[name]['f1']:.4f}")

print()

# ===== GAN ATTACK =====
print("="*80)
print("⚔️  Step 5: GAN Attack on ORIGINAL test data")
print("="*80)
print()

# Train GAN on ORIGINAL space (not latent!)
print("Building & training GAN...")

# Use MLP as target
target_classifier = mlp

def build_generator(feature_dim, noise_dim, epsilon):
    orig_input = Input(shape=(feature_dim,))
    noise_input = Input(shape=(noise_dim,))
    x = Concatenate()([orig_input, noise_input])
    x = Dense(128, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    perturbation = Dense(feature_dim, activation="tanh")(x)
    perturbation = Lambda(lambda p: epsilon * p)(perturbation)
    adv_sample = Lambda(lambda inputs: K.clip(inputs[0] + inputs[1], 0.0, 1.0))([orig_input, perturbation])
    return Model([orig_input, noise_input], adv_sample)

def build_discriminator(feature_dim):
    model = Sequential([
        Dense(128, activation="relu", input_dim=feature_dim),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy")
    return model

noise_dim = 32
epsilon = 0.1

generator = build_generator(feature_dim, noise_dim, epsilon)
discriminator = build_discriminator(feature_dim)

# Combined
for layer in discriminator.layers:
    layer.trainable = False
encoder.trainable = False
target_classifier.trainable = False

orig_input = Input(shape=(feature_dim,))
noise_input = Input(shape=(noise_dim,))
adv_sample = generator([orig_input, noise_input])

# Encode adversarial to latent for classifier
adv_latent = encoder(adv_sample)
validity = discriminator(adv_sample)
benign_pred = target_classifier(adv_latent)

combined = Model([orig_input, noise_input], [validity, benign_pred])
combined.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=["binary_crossentropy", "binary_crossentropy"],
    loss_weights=[1.0, 5.0]
)

# Train GAN
# Load train original data
print("Loading train original data for GAN...")
train_ben_X = np.load(SPLITS_DIR / "train_benign_X.npy")
train_mal_X = np.load(SPLITS_DIR / "train_malicious_X.npy")
train_ben_y = np.load(SPLITS_DIR / "train_benign_y.npy")
train_mal_y = np.load(SPLITS_DIR / "train_malicious_y.npy")
X_train_original = np.vstack([train_ben_X, train_mal_X])
y_train_full = np.hstack([train_ben_y, train_mal_y])
print(f"✓ Train original: {X_train_original.shape}")
print()

train_benign_mask = y_train_full == 0
train_malicious_mask = y_train_full == 1
X_benign_orig = X_train_original[train_benign_mask]
X_malicious_orig = X_train_original[train_malicious_mask]

batch_size = 128
epochs = 1000  # Shorter for speed

for epoch in range(1, epochs + 1):
    mal_idx = np.random.randint(0, X_malicious_orig.shape[0], batch_size)
    ben_idx = np.random.randint(0, X_benign_orig.shape[0], batch_size)
    
    noise = np.random.uniform(-1.0, 1.0, (batch_size, noise_dim))
    generated = generator.predict([X_malicious_orig[mal_idx], noise], verbose=0)
    
    valid_label = np.ones((batch_size, 1))
    fake_label = np.zeros((batch_size, 1))
    
    discriminator.train_on_batch(X_benign_orig[ben_idx], valid_label)
    discriminator.train_on_batch(generated, fake_label)
    
    combined.train_on_batch([X_malicious_orig[mal_idx], noise], [valid_label, np.zeros((batch_size, 1))])
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}/{epochs}")

print("✓ GAN training complete")
print()

# Generate adversarial samples
print("Generating adversarial samples...")
noise = np.random.uniform(-1.0, 1.0, (test_mal_X_original.shape[0], noise_dim))
test_adv_mal_X = generator.predict([test_mal_X_original, noise], verbose=0)

# Create adversarial test set
test_malicious_mask = y_test == 1
X_test_adv_original = X_test_original.copy()
X_test_adv_original[test_malicious_mask] = test_adv_mal_X

# Encode to latent
X_test_adv_latent = encoder.predict(X_test_adv_original, verbose=0)
print(f"✓ Generated {test_adv_mal_X.shape[0]} adversarial samples")
print()

# ===== EVALUATE ON ADVERSARIAL =====
print("="*80)
print("📊 Step 6: Evaluate on ADVERSARIAL data")
print("="*80)
print()

attack_results = {}

for name, model in models_latent.items():
    if name in ['MLP', 'DeepMLP']:
        y_pred = (model.predict(X_test_adv_latent, verbose=0) > 0.5).astype(int).ravel()
    else:
        y_pred = model.predict(X_test_adv_latent)
    
    attack_results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    drop = baseline_results[name]['accuracy'] - attack_results[name]['accuracy']
    print(f"✓ {name}: Acc={attack_results[name]['accuracy']:.4f} (drop: {drop:.4f})")

print()

# ===== SAVE RESULTS =====
print("="*80)
print("💾 Saving results...")
print("="*80)

comparison = []
for name in baseline_results.keys():
    comparison.append({
        'model': name,
        'baseline_accuracy': baseline_results[name]['accuracy'],
        'baseline_f1': baseline_results[name]['f1'],
        'attack_accuracy': attack_results[name]['accuracy'],
        'attack_f1': attack_results[name]['f1'],
        'accuracy_drop': baseline_results[name]['accuracy'] - attack_results[name]['accuracy'],
        'f1_drop': baseline_results[name]['f1'] - attack_results[name]['f1'],
    })

df = pd.DataFrame(comparison)
results_path = RESULTS_DIR / f"latent_baseline_vs_attack_{TIMESTAMP}.csv"
df.to_csv(results_path, index=False)
print(f"✓ Results: {results_path}")
print()

# ===== SUMMARY =====
print("="*80)
print("📈 LATENT DEFENSE SUMMARY")
print("="*80)
print(df.to_string(index=False))
print()

print(f"🏆 Most Robust: {df.loc[df['accuracy_drop'].idxmin(), 'model']}")
print(f"⚠️  Most Vulnerable: {df.loc[df['accuracy_drop'].idxmax(), 'model']}")
print()

print("💡 Compare with baseline experiment to see if latent defense is more robust!")
print("="*80)
print("✅ LATENT DEFENSE COMPLETE!")
print("="*80)
