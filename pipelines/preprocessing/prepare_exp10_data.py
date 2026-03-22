#!/usr/bin/env python3
"""
Prepare Data cho Experiment 10: Poisoned Full-System Retraining
===============================================================

Vấn đề với prepare_data.py cũ
-------------------------------
prepare_data.py tạo latent poisoned data (exp2_poisoning/) bằng cách
encode qua clean encoder:

    X_lat = clean_benign_enc(X_raw) + clean_malicious_enc(X_raw)

→ Đây là đúng cho Exp2/Exp9 (defender biết DeDe/DualEnc clean).
→ SAI cho Exp10: defender đã train lại DualEncoder trên poisoned data.

Cách tiếp cận của script này (Exp10)
--------------------------------------
Với mỗi poison rate:
  1. Lấy X_train_raw (50-dim, đã scale/select) từ RAW splits hiện có
  2. Lấy y_train_poisoned (label đã flip) từ RAW splits hiện có
  3. Train benign_encoder   CHỈ trên X[y_poison==0] (bao gồm cả malicious bị đổi nhãn)
  4. Train malicious_encoder CHỈ trên X[y_poison==1] (thiếu các poisoned samples)
  5. Encode ALL X_train_raw qua cặp encoder bị nhiễm → latent_poisoned (64-dim)
  6. Save vào datasets/splits/exp10_latent/poison_{rate}/


"""

import sys, json, argparse
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_SPLITS    = BASE_DIR / "datasets/splits/3.0_raw_from_latent"
LATENT_CLEAN  = BASE_DIR / "datasets/splits/3.1_latent"
CLEAN_MODELS  = LATENT_CLEAN / "models"
OUT_BASE      = BASE_DIR / "datasets/splits/exp10_latent"
RANDOM_STATE  = 42


# ── Encoder Architecture ──────────────────────────────────────────────────────

def build_encoder(input_dim: int, latent_dim: int, name: str) -> keras.Model:
    """
    Kiến trúc giống prepare_data.py gốc (4-layer MLP encoder).
    input_dim → 256 → 128 → 64 → latent_dim
    """
    inp = keras.Input(shape=(input_dim,), name=f'{name}_input')
    x   = layers.Dense(256, activation='relu', name=f'{name}_enc1')(inp)
    x   = layers.Dense(128, activation='relu', name=f'{name}_enc2')(x)
    x   = layers.Dense(64,  activation='relu', name=f'{name}_enc3')(x)
    out = layers.Dense(latent_dim, activation='relu', name=f'{name}_latent')(x)
    model = keras.Model(inp, out, name=name)

    # Full AE for training (decoder)
    dec_inp = keras.Input(shape=(latent_dim,), name=f'{name}_dec_input')
    d = layers.Dense(64,  activation='relu', name=f'{name}_dec1')(dec_inp)
    d = layers.Dense(128, activation='relu', name=f'{name}_dec2')(d)
    d = layers.Dense(256, activation='relu', name=f'{name}_dec3')(d)
    d = layers.Dense(input_dim, activation='sigmoid', name=f'{name}_output')(d)
    decoder = keras.Model(dec_inp, d, name=f'{name}_decoder')

    ae_out = decoder(model(inp))
    ae = keras.Model(inp, ae_out, name=f'{name}_autoencoder')
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')

    return model, ae  # trả về (encoder, autoencoder)


def train_encoder(ae, X_train, X_val, epochs, batch_size, name, save_dir: Path):
    """Train autoencoder và lưu weights của encoder."""
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = str(save_dir / f'{name}_ae_best.weights.h5')
    cbs  = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
        ModelCheckpoint(ckpt, monitor='val_loss', save_best_only=True,
                        save_weights_only=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4,
                          min_lr=1e-6, verbose=0),
    ]
    hist = ae.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=cbs, verbose=0,
    )
    best_loss = min(hist.history['val_loss'])
    print(f'    ✓ {name}: best_val_loss={best_loss:.6f}  ({len(X_train):,} samples)')
    return best_loss


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Prepare latent data for Exp10 using poisoned DualEncoder'
    )
    parser.add_argument('--poison-rates', nargs='+', type=int, default=[5, 10, 15, 50])
    parser.add_argument('--epochs',     type=int, default=100,
                        help='Max epochs when training poisoned encoders (same as prepare_data.py=100)')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--latent-dim', type=int, default=32,
                        help='Latent dim per encoder; total = 2x (default: 32)')
    parser.add_argument('--raw-dir',    default=str(RAW_SPLITS))
    parser.add_argument('--out-dir',    default=str(OUT_BASE))
    args = parser.parse_args()

    raw_dir  = Path(args.raw_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    enc_dir  = out_dir / 'encoders'
    enc_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    print('\n' + '='*80)
    print(' PREPARE EXP10 LATENT DATA (Poisoned DualEncoder) '.center(80, '='))
    print('='*80)
    print(f"""
  Mục tiêu: Tạo latent features cho Exp10 Scenario C
  - Train DualEncoder trên poisoned data (theo từng poison rate)
  - Encode ALL X_train → latent bằng poisoned encoder
  - X_test: cũng encode bởi POISONED encoder (realistic: toàn bộ hệ thống bị nhiễm)
  - X_test_cleanenc.npy: encode bởi CLEAN encoder (chỉ lưu để so sánh)

  RAW splits    : {raw_dir}
  Output        : {out_dir}
  Latent dim    : {args.latent_dim} per encoder → {args.latent_dim*2} total
  Poison rates  : {args.poison_rates}
""")

    # ── Load clean test set ─────────────────────────────────────────────
    print('[INIT] Loading clean test set (RAW 50-dim)...')
    X_test_clean  = np.load(raw_dir / 'exp1_baseline/X_test.npy')
    y_test_clean  = np.load(raw_dir / 'exp1_baseline/y_test.npy')
    X_train_clean = np.load(raw_dir / 'exp1_baseline/X_train.npy')
    y_train_clean = np.load(raw_dir / 'exp1_baseline/y_train.npy')
    input_dim     = X_train_clean.shape[1]
    print(f'  Train: {len(X_train_clean):,}×{input_dim}  |  Test: {len(X_test_clean):,}×{input_dim}')

    # ── Load clean encoder để encode test set ───────────────────────────
    print('\n[INIT] Loading clean DualEncoder...')
    clean_benc = tf.keras.models.load_model(str(CLEAN_MODELS / 'benign_encoder.h5'))
    clean_menc = tf.keras.models.load_model(str(CLEAN_MODELS / 'malicious_encoder.h5'))
    print('  ✓ Clean encoders loaded')

    # Encode clean test set với clean encoder → dùng chung cho tất cả poison rates
    print('\n[INIT] Encoding clean test set with clean DualEncoder...')
    X_test_lat_clean = np.hstack([
        clean_benc.predict(X_test_clean.astype(np.float32), verbose=0),
        clean_menc.predict(X_test_clean.astype(np.float32), verbose=0),
    ])
    print(f'  ✓ X_test_latent(clean): {X_test_lat_clean.shape}')

    # ── Copy baseline (clean latent) ────────────────────────────────────
    baseline_dir = out_dir / 'baseline'
    baseline_dir.mkdir(exist_ok=True)
    # Encode clean train với clean encoder
    print('\n[INIT] Encoding clean train set with clean DualEncoder...')
    X_train_lat_clean = np.hstack([
        clean_benc.predict(X_train_clean.astype(np.float32), verbose=0),
        clean_menc.predict(X_train_clean.astype(np.float32), verbose=0),
    ])
    np.save(baseline_dir / 'X_train.npy', X_train_lat_clean)
    np.save(baseline_dir / 'y_train.npy', y_train_clean)
    np.save(baseline_dir / 'X_test.npy',  X_test_lat_clean)
    np.save(baseline_dir / 'y_test.npy',  y_test_clean)
    print(f'  ✓ Baseline latent saved → {baseline_dir}')

    # ── Per poison rate ─────────────────────────────────────────────────
    manifest = []

    for rate in args.poison_rates:
        rate_str   = f'{rate:02d}'
        poison_raw = raw_dir / f'exp2_poisoning/poison_{rate_str}'

        if not (poison_raw / 'X_train.npy').exists():
            print(f'\n  ⚠️  Skipping poison_{rate}% — RAW data not found at {poison_raw}')
            continue

        print('\n' + '='*80)
        print(f'  POISON RATE: {rate}%'.center(80))
        print('='*80)

        # Load poisoned RAW training data
        X_tr_raw  = np.load(poison_raw / 'X_train.npy').astype(np.float32)
        y_tr_pois = np.load(poison_raw / 'y_train.npy')  # poisoned labels
        flips     = (y_tr_pois != y_train_clean).sum()
        print(f'  {len(X_tr_raw):,} train samples  |  {flips:,} label flips ({flips/len(y_train_clean)*100:.1f}%)')

        # Phân tách theo poisoned label
        # QUAN TRỌNG:  y_tr_pois[i]==0 có thể là benign thật HOẶC malicious bị flip
        #             Defender không phân biệt được → encode như nhau
        X_tr_b = X_tr_raw[y_tr_pois == 0]   # "benign" theo defender (bao gồm poisoned)
        X_tr_m = X_tr_raw[y_tr_pois == 1]   # "malicious" theo defender (thiếu poisoned)

        print(f'  Label 0 (defender thinks benign): {len(X_tr_b):,}'
              f'  [thực tế có {flips:,} là malicious bị flip]')
        print(f'  Label 1 (defender thinks malicious): {len(X_tr_m):,}')

        # Validation split: dùng 10% của mỗi nhóm
        n_val_b = max(200, int(len(X_tr_b) * 0.1))
        n_val_m = max(200, int(len(X_tr_m) * 0.1))
        rng  = np.random.RandomState(RANDOM_STATE + rate)
        idx_b = rng.permutation(len(X_tr_b))
        idx_m = rng.permutation(len(X_tr_m))
        X_val_b, X_fit_b = X_tr_b[idx_b[:n_val_b]], X_tr_b[idx_b[n_val_b:]]
        X_val_m, X_fit_m = X_tr_m[idx_m[:n_val_m]], X_tr_m[idx_m[n_val_m:]]

        # ── Train poisoned encoders ───────────────────────────────────
        poison_enc_dir = enc_dir / f'poison_{rate_str}'

        print(f'\n  [1/3] Training BENIGN encoder on poisoned "label-0" data...')
        benc, bae = build_encoder(input_dim, args.latent_dim, 'benign_encoder')
        train_encoder(bae, X_fit_b, X_val_b,
                      args.epochs, args.batch_size, 'benign', poison_enc_dir)

        print(f'  [2/3] Training MALICIOUS encoder on poisoned "label-1" data...')
        menc, mae = build_encoder(input_dim, args.latent_dim, 'malicious_encoder')
        train_encoder(mae, X_fit_m, X_val_m,
                      args.epochs, args.batch_size, 'malicious', poison_enc_dir)

        # Save encoders
        benc.save(str(poison_enc_dir / 'benign_encoder.h5'))
        menc.save(str(poison_enc_dir / 'malicious_encoder.h5'))
        print(f'  ✓ Poisoned encoders saved → {poison_enc_dir}')

        # ── Encode ALL train với poisoned encoder ─────────────────────
        print(f'\n  [3/4] Encoding ALL {len(X_tr_raw):,} train samples with poisoned DualEncoder...')
        X_tr_lat = np.hstack([
            benc.predict(X_tr_raw, verbose=0),
            menc.predict(X_tr_raw, verbose=0),
        ])
        print(f'  ✓ Latent train shape: {X_tr_lat.shape}')

        # ── Encode X_test với POISONED encoder (REALISTIC!) ─────────────
        # Trong deployment thực tế, TOÀN BỘ hệ thống dùng poisoned encoder
        # → traffic mới (kể cả test) cũng qua POISONED encoder mới đúng scenario
        print(f'  [4/4] Encoding X_test with POISONED encoder (realistic deployment)...')
        X_te_lat_poison = np.hstack([
            benc.predict(X_test_clean.astype(np.float32), verbose=0),
            menc.predict(X_test_clean.astype(np.float32), verbose=0),
        ])
        # Cũng lưu clean-encoded test để so sánh (optional reference)
        X_te_lat_cleanref = np.hstack([
            clean_benc.predict(X_test_clean.astype(np.float32), verbose=0),
            clean_menc.predict(X_test_clean.astype(np.float32), verbose=0),
        ])
        print(f'  ✓ Latent test (poisoned enc): {X_te_lat_poison.shape}')

        # ── Save output ───────────────────────────────────────────────
        out_rate = out_dir / f'poison_{rate_str}'
        out_rate.mkdir(exist_ok=True)

        np.save(out_rate / 'X_train.npy', X_tr_lat)
        np.save(out_rate / 'y_train.npy', y_tr_pois)
        # X_test encode bởi POISONED encoder → đánh giá realistic
        np.save(out_rate / 'X_test.npy',  X_te_lat_poison)
        np.save(out_rate / 'y_test.npy',  y_test_clean)
        # X_test encode bởi CLEAN encoder → chỉ để so sánh
        np.save(out_rate / 'X_test_cleanenc.npy', X_te_lat_cleanref)

        # Meta
        meta = {
            'poison_rate': rate,
            'n_train': int(len(X_tr_raw)),
            'n_label_flips': int(flips),
            'latent_dim_per_enc': args.latent_dim,
            'latent_dim_total': args.latent_dim * 2,
            'encoder_trained_on': f'poisoned_{rate}pct_labels',
            'X_train_encoded_by': 'poisoned_encoder',
            'X_test_encoded_by': 'poisoned_encoder (REALISTIC deployment)',
            'X_test_cleanenc_encoded_by': 'clean_encoder (reference only)',
            'input_dim': int(input_dim),
            'created_at': ts,
            'note': (
                'X_train AND X_test.npy encoded by POISONED DualEncoder '
                '(realistic: entire deployed system uses poisoned encoder). '
                'X_test_cleanenc.npy = clean encoder, available for optional comparison.'
            ),
        }
        with open(out_rate / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        manifest.append({
            'rate': rate,
            'path': str(out_rate),
            'X_train_shape': list(X_tr_lat.shape),
            'flips': int(flips),
        })
        print(f'  ✓ Saved → {out_rate}')
        print(f'    X_train: {X_tr_lat.shape}  (poisoned latent)')
        print(f'    X_test : {X_te_lat_poison.shape}  (poisoned enc — realistic)')
        print(f'    X_test_cleanenc.npy: {X_te_lat_cleanref.shape}  (clean enc — reference)')

    # ── Save manifest ────────────────────────────────────────────────────
    with open(out_dir / 'manifest.json', 'w') as f:
        json.dump({'created_at': ts, 'datasets': manifest}, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────
    print('\n' + '='*80)
    print('✅ HOÀN THÀNH! Dữ liệu Exp10 Latent đã sẵn sàng'.center(80))
    print('='*80)
    print(f"""
  Output: {out_dir}/
    baseline/        ← clean DualEncoder (dùng cho Scenario A, B)
    encoders/
      poison_05/     ← poisoned DualEncoder cho 5%
      poison_10/     ...
    poison_05/       ← X_train latent (poisoned enc) + y_train (poisoned labels)
    poison_10/       ...

  Dùng với:
    python experiments/latent/exp10_poisoned_retrain_latent.py \\
        --latent-dir {out_dir}

  Lưu ý:
    • X_train được encode bởi DualEncoder train trên poisoned labels
      → Latent space bị skewed: poisoned malicious nằm trong "benign" cluster
    • X_test được encode bởi POISONED encoder (realistic deployment)
      → F1 phản ánh đúng hiệu năng thực tế của hệ thống bị nhiễm
    • X_test_cleanenc.npy: encode bởi clean encoder (chỉ để so sánh)
    • So sánh với 3.1_latent/exp2_poisoning/ (đưa và clean latent)
      → Thấy mức ảnh hưởng của poisoned encoder lên latent space
""")


if __name__ == '__main__':
    main()
