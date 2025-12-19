#!/usr/bin/env python3
"""
Script để sinh mẫu adversarial từ test malicious CICIDS2017 bằng GAN

Quy trình:
1. Load train/test malicious từ CICIDS2017 splits
2. Train GAN trên train malicious
3. Sinh adversarial samples từ test malicious
4. Lưu kết quả và đánh giá
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Concatenate, Dense, Input, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Paths
BASE_DIR = Path(__file__).resolve().parent
SPLITS_DIR = BASE_DIR / "datasets" / "splits" / "cicids2017"
RESULTS_DIR = BASE_DIR / "results" / "gan_adversarial_cicids2017"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def load_cicids2017_data():
    """Load CICIDS2017 train/test malicious and benign data"""
    print("\n" + "="*70)
    print("📂 LOADING CICIDS2017 DATA")
    print("="*70)
    
    # Load train data
    train_mal_X = np.load(SPLITS_DIR / "train_malicious_X.npy")
    train_mal_y = np.load(SPLITS_DIR / "train_malicious_y.npy")
    train_ben_X = np.load(SPLITS_DIR / "train_benign_X.npy")
    train_ben_y = np.load(SPLITS_DIR / "train_benign_y.npy")
    
    # Load test data
    test_mal_X = np.load(SPLITS_DIR / "test_malicious_X.npy")
    test_mal_y = np.load(SPLITS_DIR / "test_malicious_y.npy")
    test_ben_X = np.load(SPLITS_DIR / "test_benign_X.npy")
    test_ben_y = np.load(SPLITS_DIR / "test_benign_y.npy")
    
    # Combine train
    X_train = np.vstack([train_ben_X, train_mal_X])
    y_train = np.hstack([train_ben_y, train_mal_y])
    
    # Combine test
    X_test = np.vstack([test_ben_X, test_mal_X])
    y_test = np.hstack([test_ben_y, test_mal_y])
    
    print(f"✓ Train shape: {X_train.shape}, Malicious: {(y_train==1).sum():,}, Benign: {(y_train==0).sum():,}")
    print(f"✓ Test shape: {X_test.shape}, Malicious: {(y_test==1).sum():,}, Benign: {(y_test==0).sum():,}")
    print(f"✓ Feature dimension: {X_train.shape[1]}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'train_mal_X': train_mal_X,
        'test_mal_X': test_mal_X,
    }


def build_classifier(input_dim):
    """Build MLP classifier"""
    model = Sequential(name="ids_classifier")
    model.add(Dense(units=64, activation="relu", input_dim=input_dim))
    model.add(Dense(units=32, activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_generator(feature_dim, noise_dim=32, epsilon=0.1):
    """Build conditional generator for adversarial perturbations"""
    orig_input = Input(shape=(feature_dim,), name="orig_sample")
    noise_input = Input(shape=(noise_dim,), name="noise_vector")
    
    x = Concatenate()([orig_input, noise_input])
    x = Dense(128, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    
    # Generate perturbation
    perturbation = Dense(feature_dim, activation="tanh")(x)
    perturbation = Lambda(lambda p: epsilon * p, name="scaled_perturbation")(perturbation)
    
    # Add perturbation to original (clipped to [0,1])
    adv_sample = Lambda(
        lambda inputs: K.clip(inputs[0] + inputs[1], 0.0, 1.0),
        name="adv_sample",
    )([orig_input, perturbation])
    
    return Model([orig_input, noise_input], adv_sample, name="generator")


def build_discriminator(feature_dim):
    """Build discriminator to distinguish real benign from generated samples"""
    model = Sequential(name="discriminator")
    model.add(Dense(128, activation="relu", input_dim=feature_dim))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_combined(generator, discriminator, classifier, feature_dim, noise_dim=32, cls_weight=5.0):
    """Build combined model for training generator"""
    # Freeze discriminator and classifier
    for layer in discriminator.layers:
        layer.trainable = False
    discriminator.trainable = False
    
    classifier.trainable = False
    for layer in classifier.layers:
        layer.trainable = False
    
    # Build combined model
    orig_input = Input(shape=(feature_dim,), name="combined_orig")
    noise_input = Input(shape=(noise_dim,), name="combined_noise")
    
    adv_sample = generator([orig_input, noise_input])
    validity = discriminator(adv_sample)
    benign_pred = classifier(adv_sample)
    
    combined = Model(
        inputs=[orig_input, noise_input],
        outputs=[validity, benign_pred],
        name="combined_model",
    )
    combined.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=["binary_crossentropy", "binary_crossentropy"],
        loss_weights=[1.0, cls_weight],
    )
    return combined


def train_gan(
    generator,
    discriminator,
    combined,
    classifier,
    X_malicious,
    X_benign,
    noise_dim=32,
    epochs=2000,
    batch_size=128,
):
    """Train GAN to generate adversarial samples"""
    print("\n" + "="*70)
    print("🔥 TRAINING GAN")
    print("="*70)
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Noise dim: {noise_dim}")
    print(f"Malicious samples: {X_malicious.shape[0]:,}")
    print(f"Benign samples: {X_benign.shape[0]:,}")
    
    valid_label = np.ones((batch_size, 1))
    fake_label = np.zeros((batch_size, 1))
    target_benign = np.zeros((batch_size, 1))  # Generator wants classifier to predict benign (0)
    
    for epoch in range(1, epochs + 1):
        # Sample batches
        mal_idx = np.random.randint(0, X_malicious.shape[0], batch_size)
        ben_idx = np.random.randint(0, X_benign.shape[0], batch_size)
        
        malicious_batch = X_malicious[mal_idx]
        benign_batch = X_benign[ben_idx]
        
        # Generate adversarial samples
        noise = np.random.uniform(-1.0, 1.0, size=(batch_size, noise_dim))
        generated_batch = generator.predict([malicious_batch, noise], verbose=0)
        
        # Train discriminator
        d_loss_real = discriminator.train_on_batch(benign_batch, valid_label)
        d_loss_fake = discriminator.train_on_batch(generated_batch, fake_label)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train generator
        g_loss = combined.train_on_batch(
            [malicious_batch, noise], [valid_label, target_benign]
        )
        
        # Log progress
        if epoch % 100 == 0 or epoch == 1:
            benign_pred = classifier.predict(generated_batch, verbose=0)
            benign_success = (benign_pred < 0.5).mean()  # Fraction classified as benign
            
            print(
                f"Epoch {epoch:04d} | "
                f"D loss: {d_loss[0]:.4f} acc: {d_loss[1]:.4f} | "
                f"G loss: {g_loss[0]:.4f} | "
                f"Benign success: {benign_success:.2%}"
            )
    
    print("\n✅ GAN training complete!")


def generate_adversarial_samples(generator, X_malicious, noise_dim=32):
    """Generate adversarial samples from malicious samples"""
    noise = np.random.uniform(-1.0, 1.0, size=(X_malicious.shape[0], noise_dim))
    return generator.predict([X_malicious, noise], verbose=0)


def evaluate_model(model, X, y, label):
    """Evaluate model and print metrics"""
    y_pred_prob = model.predict(X, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).ravel()
    
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    print(f"\n{label}:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'y_pred': y_pred,
    }


def main():
    print("\n" + "="*70)
    print("🎯 GAN ADVERSARIAL ATTACK - CICIDS2017")
    print("="*70)
    
    # Load data
    data = load_cicids2017_data()
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    train_mal_X = data['train_mal_X']
    test_mal_X = data['test_mal_X']
    
    feature_dim = X_train.shape[1]
    noise_dim = 32
    epsilon = 0.1
    
    # Build and train classifier
    print("\n" + "="*70)
    print("🧠 TRAINING CLASSIFIER")
    print("="*70)
    classifier = build_classifier(feature_dim)
    classifier.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=50,
        batch_size=64,
        verbose=1
    )
    
    # Evaluate baseline
    baseline_results = evaluate_model(classifier, X_test, y_test, "📊 BASELINE (Original Test)")
    
    # Build GAN components
    generator = build_generator(feature_dim, noise_dim=noise_dim, epsilon=epsilon)
    discriminator = build_discriminator(feature_dim)
    combined = build_combined(
        generator, discriminator, classifier, 
        feature_dim, noise_dim=noise_dim
    )
    
    # Train GAN on TRAIN malicious
    train_ben_mask = y_train == 0
    train_mal_mask = y_train == 1
    
    train_gan(
        generator,
        discriminator,
        combined,
        classifier,
        X_train[train_mal_mask],  # Train malicious
        X_train[train_ben_mask],  # Train benign
        noise_dim=noise_dim,
        epochs=2000,
        batch_size=128,
    )
    
    # Generate adversarial samples from TEST malicious
    print("\n" + "="*70)
    print("⚔️  GENERATING ADVERSARIAL SAMPLES FROM TEST MALICIOUS")
    print("="*70)
    print(f"Test malicious samples: {test_mal_X.shape[0]:,}")
    
    test_adv_mal_X = generate_adversarial_samples(generator, test_mal_X, noise_dim=noise_dim)
    
    # Create adversarial test set: benign (unchanged) + adversarial malicious
    test_malicious_mask = y_test == 1
    X_test_adv = X_test.copy()
    X_test_adv[test_malicious_mask] = test_adv_mal_X
    
    # Evaluate on adversarial test
    adv_results = evaluate_model(classifier, X_test_adv, y_test, "⚔️  ADVERSARIAL (GAN Attack)")
    
    # Calculate perturbation statistics
    perturbations = test_adv_mal_X - test_mal_X
    l2_norms = np.linalg.norm(perturbations, axis=1)
    linf_norms = np.max(np.abs(perturbations), axis=1)
    
    print("\n" + "="*70)
    print("📊 PERTURBATION STATISTICS")
    print("="*70)
    print(f"L2 norm:   mean={l2_norms.mean():.4f}, std={l2_norms.std():.4f}, max={l2_norms.max():.4f}")
    print(f"L∞ norm:   mean={linf_norms.mean():.4f}, std={linf_norms.std():.4f}, max={linf_norms.max():.4f}")
    
    # Save results
    print("\n" + "="*70)
    print("💾 SAVING RESULTS")
    print("="*70)
    
    # Save adversarial samples
    adv_samples_path = RESULTS_DIR / f"test_adversarial_malicious_X_{TIMESTAMP}.npy"
    np.save(adv_samples_path, test_adv_mal_X)
    print(f"✓ Adversarial malicious samples: {adv_samples_path}")
    
    # Save full adversarial test set
    adv_test_path = RESULTS_DIR / f"test_adversarial_full_X_{TIMESTAMP}.npy"
    np.save(adv_test_path, X_test_adv)
    print(f"✓ Full adversarial test set: {adv_test_path}")
    
    # Save metrics
    metrics_df = pd.DataFrame([
        {
            'scenario': 'baseline',
            'accuracy': baseline_results['accuracy'],
            'precision': baseline_results['precision'],
            'recall': baseline_results['recall'],
            'f1': baseline_results['f1'],
        },
        {
            'scenario': 'gan_attack',
            'accuracy': adv_results['accuracy'],
            'precision': adv_results['precision'],
            'recall': adv_results['recall'],
            'f1': adv_results['f1'],
        }
    ])
    metrics_path = RESULTS_DIR / f"metrics_{TIMESTAMP}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✓ Metrics: {metrics_path}")
    
    # Save perturbation stats
    perturbation_df = pd.DataFrame({
        'l2_norm': l2_norms,
        'linf_norm': linf_norms,
    })
    pert_path = RESULTS_DIR / f"perturbation_stats_{TIMESTAMP}.csv"
    perturbation_df.to_csv(pert_path, index=False)
    print(f"✓ Perturbation statistics: {pert_path}")
    
    # Summary
    print("\n" + "="*70)
    print("📈 ATTACK SUMMARY")
    print("="*70)
    print(f"Baseline Accuracy:     {baseline_results['accuracy']:.4f}")
    print(f"Adversarial Accuracy:  {adv_results['accuracy']:.4f}")
    print(f"Accuracy Drop:         {baseline_results['accuracy'] - adv_results['accuracy']:.4f}")
    print(f"Baseline F1:           {baseline_results['f1']:.4f}")
    print(f"Adversarial F1:        {adv_results['f1']:.4f}")
    print(f"F1 Drop:               {baseline_results['f1'] - adv_results['f1']:.4f}")
    
    print("\n✅ DONE! All results saved to:", RESULTS_DIR)
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
