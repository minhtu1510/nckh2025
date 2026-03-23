#!/usr/bin/env python3
"""
Memory-Efficient GAN Adversarial Sample Generation
Optimizations:
- Batch generation (không generate toàn bộ cùng lúc)
- Stream encoding (encode và save theo batch)
- Memory-mapped output (ghi trực tiếp ra disk)
- Minimal copying

Author: Research Team
Date: 2026-01-29
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "datasets" / "splits" / "3.0_raw_from_latent"
LATENT_DATA_DIR = BASE_DIR / "datasets" / "splits" / "3.1_latent"
MODELS_DIR = LATENT_DATA_DIR / "models"

# GAN Configuration (MEMORY-OPTIMIZED)
LATENT_DIM = 64  # Reduced from 100
GAN_EPOCHS = 30  # Reduced from 50 (still effective)
GAN_BATCH_SIZE = 64  # Reduced from 256 for low RAM
RANDOM_STATE = 42

# Memory optimization (aggressive)
GENERATION_BATCH_SIZE = 500  # Generate in smaller batches
ENCODING_BATCH_SIZE = 500    # Encode in smaller batches


np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def create_memmap_array(filepath, shape, dtype='float32', mode='w+'):
    """Create memory-mapped array"""
    return np.memmap(filepath, dtype=dtype, mode=mode, shape=shape)


def build_generator(latent_dim, feature_dim, epsilon=0.25):
    """Build GAN generator (Evasion-Optimized)"""
    mal_input = layers.Input(shape=(feature_dim,), name='malicious_input')
    noise_input = layers.Input(shape=(latent_dim,), name='noise_input')
    
    x = layers.Concatenate()([mal_input, noise_input])
    x = layers.Dense(64, activation='relu')(x)  # Reduced
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)  # Reduced
    x = layers.BatchNormalization()(x)
    
    # Generate perturbation in [-1, 1]
    perturbation = layers.Dense(feature_dim, activation='tanh')(x)
    
    # Scale perturbation strictly by epsilon (e.g., max 12.5% change)
    perturbation_scaled = layers.Lambda(lambda p: p * epsilon)(perturbation)
    
    # Add perturbation to original malicious sample
    adv_x = layers.Add()([mal_input, perturbation_scaled])
    
    # Clip to valid scaled range [-1, 1]
    import tensorflow as tf
    adv_x_clipped = layers.Lambda(lambda val: tf.clip_by_value(val, -1.0, 1.0))(adv_x)
    
    model = models.Model(inputs=[mal_input, noise_input], outputs=adv_x_clipped, name='generator')
    return model


def build_discriminator(input_dim):
    """Build GAN discriminator (MEMORY-OPTIMIZED)"""
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),  # Reduced
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),  # Reduced
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid'),
    ], name='discriminator')
    return model


def train_gan_memory_efficient(X_malicious, X_benign, latent_dim=100, epochs=30, batch_size=256):
    """
    Train Evasion GAN with memory-efficient batching
    D tries to distinguish: Real Benign (1) vs Fake Adversarial (0)
    G tries to perturb Real Malicious so D predicts (1)
    """
    feature_dim = X_malicious.shape[1]
    
    # Normalize globally to save memory and maintain scale
    X_combined = np.vstack([X_malicious, X_benign])
    X_min = X_combined.min(axis=0)
    X_max = X_combined.max(axis=0)
    del X_combined
    
    X_mal_scaled = (X_malicious - X_min) / (X_max - X_min + 1e-8)
    X_mal_scaled = X_mal_scaled * 2 - 1  # [-1, 1]
    
    X_ben_scaled = (X_benign - X_min) / (X_max - X_min + 1e-8)
    X_ben_scaled = X_ben_scaled * 2 - 1  # [-1, 1]
    
    # Build models
    generator = build_generator(latent_dim, feature_dim, epsilon=0.25)
    discriminator = build_discriminator(feature_dim)
    
    discriminator.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Build GAN
    discriminator.trainable = False
    mal_input = layers.Input(shape=(feature_dim,))
    noise_input = layers.Input(shape=(latent_dim,))
    gan_output = discriminator(generator([mal_input, noise_input]))
    gan = models.Model(inputs=[mal_input, noise_input], outputs=gan_output, name='gan')
    gan.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        loss='binary_crossentropy'
    )
    
    print(f"  Generator params: {generator.count_params():,}")
    print(f"  Training for {epochs} epochs...")
    
    n_samples_mal = len(X_mal_scaled)
    n_samples_ben = len(X_ben_scaled)
    batches_per_epoch = min(n_samples_mal, n_samples_ben) // batch_size
    
    for epoch in range(epochs):
        epoch_d_loss = []
        epoch_g_loss = []
        
        for batch in range(batches_per_epoch):
            # 1. Train Discriminator
            idx_ben = np.random.randint(0, n_samples_ben, batch_size)
            idx_mal = np.random.randint(0, n_samples_mal, batch_size)
            
            real_benign = X_ben_scaled[idx_ben]
            base_malicious = X_mal_scaled[idx_mal]
            
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            fake_adversarial = generator.predict([base_malicious, noise], verbose=0)
            
            discriminator.trainable = True
            # D wants to predict 1 for true Benign and 0 for fake Adversarial
            d_loss_real = discriminator.train_on_batch(real_benign, np.ones((batch_size, 1)))
            d_loss_fake = discriminator.train_on_batch(fake_adversarial, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            epoch_d_loss.append(d_loss[0])
            
            # 2. Train Generator
            discriminator.trainable = False
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            base_malicious = X_mal_scaled[np.random.randint(0, n_samples_mal, batch_size)]
            
            # G wants D to predict 1 (Benign) for fake_adversarial
            g_loss = gan.train_on_batch([base_malicious, noise], np.ones((batch_size, 1)))
            epoch_g_loss.append(g_loss)
        
        # Clear memory after each epoch
        if (epoch + 1) % 5 == 0:
            tf.keras.backend.clear_session()
            import gc
            gc.collect()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"D_loss={np.mean(epoch_d_loss):.4f}, G_loss={np.mean(epoch_g_loss):.4f}")
    
    print(f"  ✓ GAN training completed (Evasion Mode)")
    return generator, X_min, X_max


def generate_adversarial_in_batches(generator, X_mal_test_raw, min_vals, max_vals,
                                    output_path, latent_dim=100, batch_size=GENERATION_BATCH_SIZE):
    """
    Generate adversarial samples by perturbing test malicious samples
    """
    n_samples, feature_dim = X_mal_test_raw.shape
    
    # Create output memmap
    X_adversarial = create_memmap_array(output_path, shape=(n_samples, feature_dim))
    
    print(f"  Generating {n_samples:,} adversarial samples in batches...")
    
    # Scale test data beforehand
    X_mal_test_scaled = (X_mal_test_raw - min_vals) / (max_vals - min_vals + 1e-8)
    X_mal_test_scaled = X_mal_test_scaled * 2 - 1
    
    for i in tqdm(range(0, n_samples, batch_size), desc="  Generating"):
        end_idx = min(i + batch_size, n_samples)
        batch_size_actual = end_idx - i
        
        # Generate noise
        noise = np.random.normal(0, 1, (batch_size_actual, latent_dim))
        batch_mal = X_mal_test_scaled[i:end_idx]
        
        # Generate adversarial samples (in [-1, 1])
        X_adv_scaled = generator.predict([batch_mal, noise], verbose=0)
        
        # Inverse transform
        X_adv = (X_adv_scaled + 1) / 2  # [-1,1] → [0,1]
        X_adv = X_adv * (max_vals - min_vals + 1e-8) + min_vals
        
        # Write to memmap
        X_adversarial[i:end_idx] = X_adv
    
    # Flush
    del X_adversarial
    
    return np.memmap(output_path, dtype='float32', mode='r', shape=(n_samples, feature_dim))


def encode_adversarial_in_batches(benign_enc, mal_enc, X_input, output_path,
                                  batch_size=ENCODING_BATCH_SIZE):
    """
    Dual-encode adversarial samples in batches
    """
    n_samples = len(X_input)
    latent_dim = benign_enc.output_shape[1]
    
    # Create output memmap
    X_encoded = create_memmap_array(output_path, shape=(n_samples, latent_dim * 2))
    
    print(f"  Encoding {n_samples:,} samples in batches...")
    
    for i in tqdm(range(0, n_samples, batch_size), desc="  Encoding"):
        end_idx = min(i + batch_size, n_samples)
        batch = X_input[i:end_idx]
        
        # Dual-encode
        z_b = benign_enc.predict(batch, verbose=0)
        z_m = mal_enc.predict(batch, verbose=0)
        
        X_encoded[i:end_idx] = np.hstack([z_b, z_m])
    
    # Flush
    del X_encoded
    
    return np.memmap(output_path, dtype='float32', mode='r', shape=(n_samples, latent_dim * 2))


def main():
    print("\n" + "="*80)
    print(" "*10 + "🤖 MEMORY-EFFICIENT GAN GENERATION")
    print(" "*15 + "Adversarial Sample Generation")
    print("="*80)
    print(f"\n  Optimizations:")
    print(f"  ✓ Batch generation ({GENERATION_BATCH_SIZE:,} samples/batch)")
    print(f"  ✓ Stream encoding ({ENCODING_BATCH_SIZE:,} samples/batch)")
    print(f"  ✓ Memory-mapped output (direct to disk)")
    print(f"  ✓ Minimal copying\n")
    
    # ========================================================================
    # STEP 1: Load RAW baseline
    # ========================================================================
    print(f"[STEP 1] Loading RAW baseline...")
    
    raw_baseline_dir = RAW_DATA_DIR / "exp1_baseline"
    if not raw_baseline_dir.exists():
        print(f"  ❌ Error: RAW baseline not found")
        sys.exit(1)
    
    X_train_raw = np.load(raw_baseline_dir / 'X_train.npy')
    y_train_raw = np.load(raw_baseline_dir / 'y_train.npy')
    X_test_raw = np.load(raw_baseline_dir / 'X_test.npy')
    y_test_raw = np.load(raw_baseline_dir / 'y_test.npy')
    
    # BƯỚC SỬA MỤC TIÊU: Tránh Rò rỉ Data (Data Leakage)
    # Lấy tập Test để Train GAN (Vì Attacker không được quyền truy cập tập Train nội bộ của hệ thống)
    malicious_test_mask = (y_test_raw == 1)
    benign_test_mask = (y_test_raw == 0)
    X_gan_train_malicious = X_test_raw[malicious_test_mask]
    X_gan_train_benign = X_test_raw[benign_test_mask]
    
    print(f"  ✓ Loaded {X_gan_train_malicious.shape[0]:,} malicious TEST samples for GAN origin")
    print(f"  ✓ Loaded {X_gan_train_benign.shape[0]:,} benign TEST samples for GAN target")
    
    # ========================================================================
    # STEP 2: Train GAN
    # ========================================================================
    print(f"\n[STEP 2] Training Targeted EVASION GAN (memory-efficient)...")
    
    generator, min_vals, max_vals = train_gan_memory_efficient(
        X_gan_train_malicious,
        X_gan_train_benign,
        latent_dim=LATENT_DIM,
        epochs=GAN_EPOCHS,
        batch_size=GAN_BATCH_SIZE
    )
    
    # Clear memory
    del X_gan_train_malicious
    del X_gan_train_benign
    
    # ========================================================================
    # STEP 3: Generate adversarial in batches
    # ========================================================================
    print(f"\n[STEP 3] Generating adversarial samples (batch mode)...")
    
    benign_test_mask = (y_test_raw == 0)
    malicious_test_mask = (y_test_raw == 1)
    
    n_benign_test = benign_test_mask.sum()
    n_malicious_test = malicious_test_mask.sum()
    
    X_test_benign = X_test_raw[benign_test_mask]
    X_test_malicious = X_test_raw[malicious_test_mask]
    
    # Generate adversarial malicious (in batches to memmap)
    temp_dir = RAW_DATA_DIR / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    X_test_mal_adversarial = generate_adversarial_in_batches(
        generator, X_test_malicious, min_vals, max_vals,
        temp_dir / "adversarial_malicious.mmap",
        LATENT_DIM, GENERATION_BATCH_SIZE
    )
    
    print(f"  ✓ Generated {n_malicious_test:,} adversarial samples")
    
    # ========================================================================
    # STEP 4: Save RAW Exp3 (efficient combine)
    # ========================================================================
    print(f"\n[STEP 4] Saving RAW Exp3...")
    
    raw_exp3_dir = RAW_DATA_DIR / "exp3_gan_attack"
    raw_exp3_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine benign + adversarial
    n_test_total = n_benign_test + n_malicious_test
    X_test_adv_raw = np.vstack([X_test_benign, X_test_mal_adversarial[:]])
    y_test_adv = np.hstack([np.zeros(n_benign_test), np.ones(n_malicious_test)])
    
    # Shuffle
    rng = np.random.RandomState(RANDOM_STATE)
    shuffle_idx = rng.permutation(n_test_total)
    
    np.save(raw_exp3_dir / 'X_train.npy', X_train_raw)
    np.save(raw_exp3_dir / 'y_train.npy', y_train_raw)
    np.save(raw_exp3_dir / 'X_test.npy', X_test_adv_raw[shuffle_idx])
    np.save(raw_exp3_dir / 'y_test.npy', y_test_adv[shuffle_idx])
    
    print(f"  ✓ RAW Exp3 saved: {raw_exp3_dir}/")
    
    # ========================================================================
    # STEP 5: Encode to LATENT (in batches)
    # ========================================================================
    print(f"\n[STEP 5] Encoding to LATENT (batch mode)...")
    
    if not MODELS_DIR.exists():
        print(f"  ❌ Error: Encoders not found")
        sys.exit(1)
    
    benign_encoder = load_model(MODELS_DIR / 'benign_encoder.h5')
    malicious_encoder = load_model(MODELS_DIR / 'malicious_encoder.h5')
    
    print(f"  ✓ Loaded encoders")
    
    # Encode adversarial test (in batches to memmap)
    X_test_adv_latent = encode_adversarial_in_batches(
        benign_encoder, malicious_encoder,
        X_test_adv_raw,
        temp_dir / "adversarial_test_latent.mmap",
        ENCODING_BATCH_SIZE
    )
    
    print(f"  ✓ Encoded to latent: {X_test_adv_latent.shape}")
    
    # ========================================================================
    # STEP 6: Save LATENT Exp3
    # ========================================================================
    print(f"\n[STEP 6] Saving LATENT Exp3...")
    
    latent_exp3_dir = LATENT_DATA_DIR / "exp3_gan_attack"
    latent_exp3_dir.mkdir(parents=True, exist_ok=True)
    
    # Load clean train from Exp1
    latent_exp1_dir = LATENT_DATA_DIR / "exp1_baseline_latent"
    X_train_latent = np.load(latent_exp1_dir / 'X_train.npy')
    y_train_latent = np.load(latent_exp1_dir / 'y_train.npy')
    
    # Save
    np.save(latent_exp3_dir / 'X_train.npy', X_train_latent)
    np.save(latent_exp3_dir / 'y_train.npy', y_train_latent)
    np.save(latent_exp3_dir / 'X_test.npy', X_test_adv_latent[shuffle_idx])
    np.save(latent_exp3_dir / 'y_test.npy', y_test_adv[shuffle_idx])
    
    print(f"  ✓ LATENT Exp3 saved: {latent_exp3_dir}/")
    
    # Summary
    print("\n" + "="*80)
    print("✅ MEMORY-EFFICIENT GAN GENERATION COMPLETED!")
    print("="*80)
    print(f"\n💾 Memory optimizations used:")
    print(f"  ✓ Batch generation ({GENERATION_BATCH_SIZE:,} samples)")
    print(f"  ✓ Stream encoding ({ENCODING_BATCH_SIZE:,} samples)")
    print(f"  ✓ Memory-mapped output files")
    print(f"  ✓ Minimal data copying")
    print(f"\n📊 Output:")
    print(f"  • RAW Exp3: {raw_exp3_dir}/")
    print(f"  • LATENT Exp3: {latent_exp3_dir}/\n")


if __name__ == '__main__':
    main()
