#!/usr/bin/env python3
"""
GAN ATTACK - 6 MODELS trên CICIDS2017

Train GAN để tạo adversarial samples, sau đó test 6 models:
1. MLP, 2. LSVM, 3. QSVM, 4. KNN, 5. RF, 6. AE-MLP
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Imports
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
RESULTS_DIR = BASE_DIR / "results" / "gan_attack_6models_cicids2017"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

print("="*80)
print("⚔️  GAN ATTACK - 6 MODELS - CICIDS2017")
print("="*80)
print()

# ===== LOAD DATA =====
print("📂 Loading data...")
X_train = np.load(SPLITS_DIR / "train_X.npy")
y_train = np.load(SPLITS_DIR / "train_y.npy")
X_test = np.load(SPLITS_DIR / "test_X.npy")
y_test = np.load(SPLITS_DIR / "test_y.npy")

train_mal_X = np.load(SPLITS_DIR / "train_malicious_X.npy")
test_mal_X = np.load(SPLITS_DIR / "test_malicious_X.npy")

print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")
print(f"✓ Test malicious: {test_mal_X.shape}")
print()

feature_dim = X_train.shape[1]
noise_dim = 32
epsilon = 0.1

# ===== BUILD 6 MODELS AND TRAIN ON BASELINE =====
print("="*80)
print("🧠 STEP 1: Train 6 models on ORIGINAL data")
print("="*80)
print()

models = {}
baseline_results = {}

# Model 1: MLP
print("Training MLP...")
mlp = Sequential([
    Dense(64, activation='relu', input_dim=feature_dim),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
mlp.fit(X_train, y_train, epochs=30, batch_size=64, verbose=0, validation_split=0.1)
models['MLP'] = mlp

y_pred = (mlp.predict(X_test, verbose=0) > 0.5).astype(int).ravel()
baseline_results['MLP'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred)
}
print(f"✓ MLP: {baseline_results['MLP']['accuracy']:.4f}")

# Model 2: LSVM
print("Training LSVM...")
lsvm = LinearSVC(max_iter=1000, random_state=42)
lsvm.fit(X_train, y_train)
models['LSVM'] = lsvm

y_pred = lsvm.predict(X_test)
baseline_results['LSVM'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred)
}
print(f"✓ LSVM: {baseline_results['LSVM']['accuracy']:.4f}")

# Model 3: QSVM (sample for speed)
print("Training QSVM (on 5000 samples)...")
sample_idx = np.random.choice(len(X_train), min(5000, len(X_train)), replace=False)
qsvm = SVC(kernel='rbf', gamma='scale', random_state=42)
qsvm.fit(X_train[sample_idx], y_train[sample_idx])
models['QSVM'] = qsvm

y_pred = qsvm.predict(X_test)
baseline_results['QSVM'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred)
}
print(f"✓ QSVM: {baseline_results['QSVM']['accuracy']:.4f}")

# Model 4: KNN
print("Training KNN...")
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train, y_train)
models['KNN'] = knn

y_pred = knn.predict(X_test)
baseline_results['KNN'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred)
}
print(f"✓ KNN: {baseline_results['KNN']['accuracy']:.4f}")

# Model 5: Random Forest
print("Training RF...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
models['RF'] = rf

y_pred = rf.predict(X_test)
baseline_results['RF'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred)
}
print(f"✓ RF: {baseline_results['RF']['accuracy']:.4f}")

# Model 6: AE-MLP
print("Training AE-MLP...")
latent_dim = 16

# Autoencoder
input_layer = Input(shape=(feature_dim,))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(latent_dim, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(feature_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=20, batch_size=256, verbose=0, validation_split=0.1)

X_train_latent = encoder.predict(X_train, verbose=0)
X_test_latent = encoder.predict(X_test, verbose=0)

ae_mlp = Sequential([
    Dense(16, activation='relu', input_dim=latent_dim),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
ae_mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ae_mlp.fit(X_train_latent, y_train, epochs=30, batch_size=64, verbose=0, validation_split=0.1)
models['AE-MLP'] = (encoder, ae_mlp)

y_pred = (ae_mlp.predict(X_test_latent, verbose=0) > 0.5).astype(int).ravel()
baseline_results['AE-MLP'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred)
}
print(f"✓ AE-MLP: {baseline_results['AE-MLP']['accuracy']:.4f}")
print()

# ===== TRAIN GAN =====
print("="*80)
print("🔥 STEP 2: Train GAN to generate adversarial samples")
print("="*80)
print()

# Use MLP as target classifier for GAN
target_classifier = mlp

# Build GAN components
def build_generator(feature_dim, noise_dim, epsilon):
    orig_input = Input(shape=(feature_dim,), name="orig_sample")
    noise_input = Input(shape=(noise_dim,), name="noise_vector")
    
    x = Concatenate()([orig_input, noise_input])
    x = Dense(128, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    
    perturbation = Dense(feature_dim, activation="tanh")(x)
    perturbation = Lambda(lambda p: epsilon * p)(perturbation)
    
    adv_sample = Lambda(
        lambda inputs: K.clip(inputs[0] + inputs[1], 0.0, 1.0)
    )([orig_input, perturbation])
    
    return Model([orig_input, noise_input], adv_sample)

def build_discriminator(feature_dim):
    model = Sequential()
    model.add(Dense(128, activation="relu", input_dim=feature_dim))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(lr=1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model

generator = build_generator(feature_dim, noise_dim, epsilon)
discriminator = build_discriminator(feature_dim)

# Combined model
for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False

target_classifier.trainable = False
for layer in target_classifier.layers:
    layer.trainable = False

orig_input = Input(shape=(feature_dim,))
noise_input = Input(shape=(noise_dim,))
adv_sample = generator([orig_input, noise_input])
validity = discriminator(adv_sample)
benign_pred = target_classifier(adv_sample)

combined = Model([orig_input, noise_input], [validity, benign_pred])
combined.compile(
    optimizer=Adam(lr=1e-3),
    loss=["binary_crossentropy", "binary_crossentropy"],
    loss_weights=[1.0, 5.0]
)

# Train GAN
print("Training GAN (2000 epochs)...")
train_benign_mask = y_train == 0
train_malicious_mask = y_train == 1
X_benign = X_train[train_benign_mask]
X_malicious = X_train[train_malicious_mask]

batch_size = 128
epochs = 2000

for epoch in range(1, epochs + 1):
    # Sample
    mal_idx = np.random.randint(0, X_malicious.shape[0], batch_size)
    ben_idx = np.random.randint(0, X_benign.shape[0], batch_size)
    
    malicious_batch = X_malicious[mal_idx]
    benign_batch = X_benign[ben_idx]
    
    noise = np.random.uniform(-1.0, 1.0, (batch_size, noise_dim))
    generated_batch = generator.predict([malicious_batch, noise], verbose=0)
    
    # Train discriminator
    valid_label = np.ones((batch_size, 1))
    fake_label = np.zeros((batch_size, 1))
    
    d_loss_real = discriminator.train_on_batch(benign_batch, valid_label)
    d_loss_fake = discriminator.train_on_batch(generated_batch, fake_label)
    
    # Train generator
    target_benign = np.zeros((batch_size, 1))
    g_loss = combined.train_on_batch([malicious_batch, noise], [valid_label, target_benign])
    
    if epoch % 500 == 0:
        benign_pred = target_classifier.predict(generated_batch, verbose=0)
        benign_success = (benign_pred < 0.5).mean()
        print(f"Epoch {epoch:04d} | Benign success: {benign_success:.2%}")

print("✓ GAN training complete!")
print()

# ===== GENERATE ADVERSARIAL SAMPLES =====
print("="*80)
print("⚔️  STEP 3: Generate adversarial samples from TEST malicious")
print("="*80)
print()

noise = np.random.uniform(-1.0, 1.0, (test_mal_X.shape[0], noise_dim))
test_adv_mal_X = generator.predict([test_mal_X, noise], verbose=0)

# Create adversarial test set
test_malicious_mask = y_test == 1
X_test_adv = X_test.copy()
X_test_adv[test_malicious_mask] = test_adv_mal_X

print(f"✓ Generated {test_adv_mal_X.shape[0]} adversarial samples")
print()

# ===== EVALUATE ALL MODELS ON ADVERSARIAL DATA =====
print("="*80)
print("📊 STEP 4: Evaluate 6 models on ADVERSARIAL data")
print("="*80)
print()

attack_results = {}

# MLP
y_pred = (models['MLP'].predict(X_test_adv, verbose=0) > 0.5).astype(int).ravel()
attack_results['MLP'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred)
}
print(f"✓ MLP: {attack_results['MLP']['accuracy']:.4f} (drop: {baseline_results['MLP']['accuracy'] - attack_results['MLP']['accuracy']:.4f})")

# LSVM
y_pred = models['LSVM'].predict(X_test_adv)
attack_results['LSVM'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred)
}
print(f"✓ LSVM: {attack_results['LSVM']['accuracy']:.4f} (drop: {baseline_results['LSVM']['accuracy'] - attack_results['LSVM']['accuracy']:.4f})")

# QSVM
y_pred = models['QSVM'].predict(X_test_adv)
attack_results['QSVM'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred)
}
print(f"✓ QSVM: {attack_results['QSVM']['accuracy']:.4f} (drop: {baseline_results['QSVM']['accuracy'] - attack_results['QSVM']['accuracy']:.4f})")

# KNN
y_pred = models['KNN'].predict(X_test_adv)
attack_results['KNN'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred)
}
print(f"✓ KNN: {attack_results['KNN']['accuracy']:.4f} (drop: {baseline_results['KNN']['accuracy'] - attack_results['KNN']['accuracy']:.4f})")

# RF
y_pred = models['RF'].predict(X_test_adv)
attack_results['RF'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred)
}
print(f"✓ RF: {attack_results['RF']['accuracy']:.4f} (drop: {baseline_results['RF']['accuracy'] - attack_results['RF']['accuracy']:.4f})")

# AE-MLP
encoder_ae, ae_mlp_model = models['AE-MLP']
X_test_adv_latent = encoder_ae.predict(X_test_adv, verbose=0)
y_pred = (ae_mlp_model.predict(X_test_adv_latent, verbose=0) > 0.5).astype(int).ravel()
attack_results['AE-MLP'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred)
}
print(f"✓ AE-MLP: {attack_results['AE-MLP']['accuracy']:.4f} (drop: {baseline_results['AE-MLP']['accuracy'] - attack_results['AE-MLP']['accuracy']:.4f})")
print()

# ===== SAVE RESULTS =====
print("="*80)
print("💾 Saving results...")
print("="*80)

# Comparison table
comparison = []
for model_name in baseline_results.keys():
    comparison.append({
        'model': model_name,
        'baseline_accuracy': baseline_results[model_name]['accuracy'],
        'baseline_f1': baseline_results[model_name]['f1'],
        'attack_accuracy': attack_results[model_name]['accuracy'],
        'attack_f1': attack_results[model_name]['f1'],
        'accuracy_drop': baseline_results[model_name]['accuracy'] - attack_results[model_name]['accuracy'],
        'f1_drop': baseline_results[model_name]['f1'] - attack_results[model_name]['f1'],
    })

df = pd.DataFrame(comparison)
results_path = RESULTS_DIR / f"baseline_vs_attack_{TIMESTAMP}.csv"
df.to_csv(results_path, index=False)
print(f"✓ Results: {results_path}")

# Save adversarial samples
adv_path = RESULTS_DIR / f"test_adversarial_malicious_X_{TIMESTAMP}.npy"
np.save(adv_path, test_adv_mal_X)
print(f"✓ Adversarial samples: {adv_path}")
print()

# ===== SUMMARY =====
print("="*80)
print("📈 ATTACK SUMMARY")
print("="*80)
print(df.to_string(index=False))
print()

print(f"🏆 Most Robust (smallest drop): {df.loc[df['accuracy_drop'].idxmin(), 'model']}")
print(f"⚠️  Most Vulnerable (largest drop): {df.loc[df['accuracy_drop'].idxmax(), 'model']}")
print()

print("="*80)
print("✅ GAN ATTACK COMPLETE!")
print("="*80)
