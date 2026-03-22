"""
Prepare Exp11: Single Encoder Latent Data
==========================================

Khác biệt với prepare_data.py (DualEncoder):
  DualEncoder: benign_AE train trên benign only
               malicious_AE train trên malicious only
               z = [benign_enc(X) | malicious_enc(X)]  64-dim

  SingleEncoder (Exp11): 1 AE train trên TẤT CẢ data (benign + malicious)
                         z = single_enc(X)              64-dim

Mục đích: So sánh DualEncoder vs SingleEncoder
  Nếu DualEncoder tốt hơn → chứng minh việc tách riêng 2 AE có giá trị
  Nếu không khác biệt    → DualEncoder không cần thiết

Same data splits như Exp1/2/3 để so sánh công bằng:
  Exp11 Baseline  ↔ Exp1  Latent (clean)
  Exp11 Poisoning ↔ Exp2  Latent (clean encoder) & Exp2b (poisoned encoder)
  Exp11 GAN       ↔ Exp3  Latent

AE Architecture (giống 1 nhánh của DualEncoder):
  input(50) → Dense(256) → Dense(128) → Dense(64) → latent(64)
  latent(64)→ Dense(128) → Dense(256) → output(50)

Output: datasets/splits/3.2_latent_single_enc/

Usage:
    python pipelines/preprocessing/prepare_exp11_data.py
    python pipelines/preprocessing/prepare_exp11_data.py --latent-dim 64
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_SPLITS  = BASE_DIR / "datasets/splits/3.0_raw_from_latent"   # cùng raw data
OUTPUT_BASE = BASE_DIR / "datasets/splits/3.2_latent_single_enc"  # output

# ── Constants (giống prepare_data.py) ─────────────────────────────────────────
LATENT_DIM  = 64   # 64-dim để match DualEncoder output (32+32=64)
AE_EPOCHS   = 100  # giống prepare_data.py
AE_BATCH    = 256  # giống prepare_data.py
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ── Single AE Architecture ────────────────────────────────────────────────────

def build_single_encoder(input_dim: int, latent_dim: int):
    """
    1 AE chung train trên TẤT CẢ data (benign + malicious).
    Architecture giống 1 nhánh của DualEncoder.
    input(50) → 256 → 128 → 64 → latent(64)
    """
    inp = keras.Input(shape=(input_dim,), name='single_enc_input')
    noisy = layers.GaussianNoise(0.1)(inp)
    x   = layers.Dense(256, activation='relu', name='enc1')(noisy)
    x   = layers.Dense(128, activation='relu', name='enc2')(x)
    x   = layers.Dense(64,  activation='relu', name='enc3')(x)
    out = layers.Dense(latent_dim, activation='relu', name='latent')(x)
    encoder = keras.Model(inp, out, name='single_encoder')

    dec_inp = keras.Input(shape=(latent_dim,), name='dec_input')
    d = layers.Dense(64,  activation='relu', name='dec1')(dec_inp)
    d = layers.Dense(128, activation='relu', name='dec2')(d)
    d = layers.Dense(256, activation='relu', name='dec3')(d)
    d = layers.Dense(input_dim, activation='sigmoid', name='dec_out')(d)
    decoder = keras.Model(dec_inp, d, name='single_decoder')

    ae_out = decoder(encoder(inp))
    ae = keras.Model(inp, ae_out, name='single_autoencoder')
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber())

    return encoder, ae


def train_ae(ae, X_tr, X_val, save_dir: Path, name='single'):
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = str(save_dir / f'{name}_ae_best.weights.h5')
    cbs = [
        EarlyStopping(monitor='val_loss', patience=10,
                      restore_best_weights=True, verbose=0),
        ModelCheckpoint(ckpt, monitor='val_loss', save_best_only=True,
                        save_weights_only=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4,
                          min_lr=1e-6, verbose=0),
    ]
    print(f'  Training AE ({len(X_tr):,} × {X_tr.shape[1]}, max {AE_EPOCHS} epochs)...',
          flush=True)
    hist = ae.fit(X_tr, X_tr,
                  validation_data=(X_val, X_val),
                  epochs=AE_EPOCHS, batch_size=AE_BATCH,
                  callbacks=cbs, verbose=0)
    best = min(hist.history['val_loss'])
    print(f'  ✓ best_val_loss={best:.6f}  '
          f'({len(hist.history["val_loss"])} epochs ran)')
    return best


def encode_and_save(encoder, X_train, y_train, X_test, y_test, output_dir: Path,
                    label: str):
    """Encode và save dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'  Encoding [{label}]...', end=' ', flush=True)
    Z_train = encoder.predict(X_train.astype(np.float32), verbose=0)
    Z_test  = encoder.predict(X_test.astype(np.float32),  verbose=0)
    np.save(output_dir / 'X_train.npy', Z_train)
    np.save(output_dir / 'y_train.npy', y_train)
    np.save(output_dir / 'X_test.npy',  Z_test)
    np.save(output_dir / 'y_test.npy',  y_test)
    print(f'✓  train={Z_train.shape}  test={Z_test.shape}')
    return Z_train, Z_test


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Prepare Exp11: Single Encoder Latent (compare with DualEncoder)'
    )
    parser.add_argument('--latent-dim',   type=int, default=LATENT_DIM,
                        help=f'Latent dimension (default: {LATENT_DIM}, same as DualEncoder)')
    parser.add_argument('--poison-rates', nargs='+', type=int, default=[5, 10, 15, 50])
    parser.add_argument('--raw-dir',      default=str(RAW_SPLITS))
    parser.add_argument('--output-dir',   default=str(OUTPUT_BASE))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)

    print('\n' + '='*80)
    print(' PREPARE EXP11: SINGLE ENCODER LATENT '.center(80, '='))
    print('='*80)
    print(f"""
  DualEncoder: train 2 AEs riêng (benign / malicious) → z=64 (32+32)
  SingleEncoder: train 1 AE chung (all data)           → z={args.latent_dim}
  
  Cùng data như Exp1/2/3 để so sánh công bằng.
  Architecture: input({raw_dir/'exp1_baseline/X_train.npy'})→256→128→64→latent({args.latent_dim})
""")

    # ── Load baseline data ─────────────────────────────────────────────────
    baseline_dir = raw_dir / 'exp1_baseline'
    X_train_base = np.load(baseline_dir / 'X_train.npy').astype(np.float32)
    y_train_base = np.load(baseline_dir / 'y_train.npy')
    X_test_base  = np.load(baseline_dir / 'X_test.npy').astype(np.float32)
    y_test_base  = np.load(baseline_dir / 'y_test.npy')
    input_dim    = X_train_base.shape[1]
    print(f'  Baseline: train={X_train_base.shape}  test={X_test_base.shape}')

    # ── STEP 1: Train Single AE trên ALL clean data ────────────────────────
    print('\n[STEP 1] Training Single AE on ALL clean training data...')
    print(f'  (Baseline: {len(X_train_base):,} samples — benign + malicious mixed)')
    enc_dir  = out_dir / 'models'
    encoder, ae = build_single_encoder(input_dim, args.latent_dim)
    print(f'  AE params: {ae.count_params():,}')
    best_loss = train_ae(ae, X_train_base, X_test_base, enc_dir, name='single')

    # Save encoder model
    enc_dir.mkdir(parents=True, exist_ok=True)
    encoder.save(str(enc_dir / 'single_encoder.h5'))
    import json
    with open(enc_dir / 'config.json', 'w') as f:
        json.dump({'input_dim': int(input_dim), 'latent_dim': int(args.latent_dim),
                   'architecture': '256-128-64-latent',
                   'trained_on': 'all_data (benign+malicious)',
                   'best_val_loss': float(best_loss),
                   'ae_epochs': AE_EPOCHS, 'ae_batch': AE_BATCH}, f, indent=2)
    print(f'  ✓ Encoder saved → {enc_dir / "single_encoder.h5"}')

    # ── STEP 2: Exp11_Baseline — encode clean data ──────────────────────────
    print('\n[STEP 2] Encoding EXP11 Baseline (clean data)...')
    exp11_base = out_dir / 'exp11_baseline'
    encode_and_save(encoder, X_train_base, y_train_base,
                    X_test_base,  y_test_base,
                    exp11_base, 'exp11_baseline')

    # ── STEP 3: Exp11_Poisoning (clean AE) — encode poisoned labels, clean encoder ──
    print('\n[STEP 3] Encoding EXP11 Poisoning — CLEAN encoder, poisoned labels...')
    print('  (Tương đương Exp2 Latent: chỉ labels bị nhiễm, AE giữ sạch)')
    for rate in args.poison_rates:
        rs         = f'{rate:02d}'
        poison_dir = raw_dir / f'exp2_poisoning/poison_{rs}'
        if not (poison_dir / 'X_train.npy').exists():
            print(f'  ⚠️  Skipping poison_{rs} — not found')
            continue
        X_p_train = np.load(poison_dir / 'X_train.npy').astype(np.float32)
        y_p_train = np.load(poison_dir / 'y_train.npy')
        out_p = out_dir / f'exp11_poisoning/poison_{rs}'
        encode_and_save(encoder, X_p_train, y_p_train,
                        X_test_base, y_test_base,
                        out_p, f'exp11_poison_{rs}')
        # CLEAN encoder → giống Exp2 Latent (DualEnc clean)

    # ── STEP 3b: Exp11_Poisoning_PEnc (poisoned AE) — retrain AE per poison rate ──
    print('\n[STEP 3b] EXP11 Poisoning — POISONED SingleEncoder (↔ Exp2b DualEnc poisoned)...')
    print('  Retrain 1 AE chung trên poisoned data per rate → encode → latent bị nhiễm')
    for rate in args.poison_rates:
        rs         = f'{rate:02d}'
        poison_dir = raw_dir / f'exp2_poisoning/poison_{rs}'
        if not (poison_dir / 'X_train.npy').exists():
            print(f'  ⚠️  Skipping poison_{rs} — not found')
            continue

        print(f'\n  ── Poison {rate}% — retrain SingleAE on poisoned data ──')
        X_p_train = np.load(poison_dir / 'X_train.npy').astype(np.float32)
        y_p_train = np.load(poison_dir / 'y_train.npy')

        # Train NEW SingleAE trên poisoned training data (all poisoned, mixed)
        # → AE bị nhiễm: học reconstruction trên data có nhãn bị flip
        #   → latent space méo, giống Exp2b (poisoned DualEncoder)
        penc_dir = out_dir / f'models/poisoned_enc/poison_{rs}'
        penc_dir.mkdir(parents=True, exist_ok=True)
        p_encoder, p_ae = build_single_encoder(input_dim, args.latent_dim)
        print(f'    Training poisoned SingleAE on {len(X_p_train):,} samples...')
        # validation split: 10% của poisoned train
        n_val = max(200, int(len(X_p_train) * 0.1))
        rng   = np.random.RandomState(RANDOM_STATE + rate)
        idx   = rng.permutation(len(X_p_train))
        X_p_val  = X_p_train[idx[:n_val]]
        X_p_fit  = X_p_train[idx[n_val:]]
        best_loss = train_ae(p_ae, X_p_fit, X_p_val, penc_dir, name=f'single_poison_{rs}')
        p_encoder.save(str(penc_dir / 'single_encoder_poisoned.h5'))
        print(f'    ✓ Poisoned SingleAE saved  best_val_loss={best_loss:.6f}')

        # Encode: train qua poisoned AE, test qua poisoned AE (realistic)
        out_pp = out_dir / f'exp11_poisoning_penc/poison_{rs}'
        encode_and_save(p_encoder, X_p_train, y_p_train,
                        X_test_base, y_test_base,
                        out_pp, f'exp11_penc_{rs}')
        # Note: X_test encode bởi POISONED SingleAE → realistic deployment
        #       → Exp11_penc so sánh ↔ Exp2b (poisoned DualEncoder)

    # ── STEP 4: Exp11_GAN — encode GAN attack test data ─────────────────────
    print('\n[STEP 4] Encoding EXP11 GAN Attack test data...')
    gan_dir = raw_dir / 'exp3_gan_attack'
    if (gan_dir / 'X_test.npy').exists():
        X_gan_test = np.load(gan_dir / 'X_test.npy').astype(np.float32)
        y_gan_test = np.load(gan_dir / 'y_test.npy')
        X_gan_tr   = np.load(gan_dir / 'X_train.npy').astype(np.float32) \
                     if (gan_dir / 'X_train.npy').exists() else X_train_base
        y_gan_tr   = np.load(gan_dir / 'y_train.npy') \
                     if (gan_dir / 'y_train.npy').exists() else y_train_base
        out_gan = out_dir / 'exp11_gan_attack'
        encode_and_save(encoder, X_gan_tr, y_gan_tr,
                        X_gan_test, y_gan_test,
                        out_gan, 'exp11_gan')
    else:
        print('  ⚠️  GAN data not found, skipping')

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '='*80)
    print('✅ EXP11 DATA READY'.center(80))
    print('='*80)
    print(f"""
  Output: {out_dir}/
    models/single_encoder.h5              ← 1 clean encoder (all clean data)
    models/poisoned_enc/poison_XX/        ← poisoned SingleEncoder per rate
    exp11_baseline/X_train.npy            ← {args.latent_dim}-dim latent (clean)
    exp11_poisoning/poison_XX/            ← {args.latent_dim}-dim latent (clean AE, poisoned labels)
    exp11_poisoning_penc/poison_XX/       ← {args.latent_dim}-dim latent (POISONED AE + poisoned labels)
    exp11_gan_attack/                     ← {args.latent_dim}-dim latent (GAN)

  So sánh đầy đủ:
    Exp11 baseline            ↔ Exp1   Latent (DualEncoder, clean)
    Exp11 poisoning           ↔ Exp2   Latent (DualEncoder, clean AE, poisoned labels)
    Exp11 poisoning_penc      ↔ Exp2b  Latent (DualEncoder, POISONED AE + poisoned labels) ← MỚI!
    Exp11 GAN attack test     ↔ Exp3   Latent (DualEncoder, GAN)

    SingleEncoder {args.latent_dim}-dim vs DualEncoder 64-dim (32+32)
    → Nếu DualEncoder tốt hơn: dual-view representation có giá trị!

  Chạy thực nghiệm:
    python experiments/latent/exp11_single_enc_latent.py
""")


if __name__ == '__main__':
    main()
