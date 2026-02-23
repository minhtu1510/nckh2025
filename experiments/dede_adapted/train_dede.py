"""
Training script cho DeDe-Adapted model

Train masked autoencoder trên clean data (Exp1 baseline)
để học reconstruct network features

Usage:
    python experiments/dede_adapted/train_dede.py \
        --data-dir datasets/splits/raw_scaled/exp1_baseline \
        --output-dir experiments/dede_adapted/models \
        --epochs 100 \
        --mask-ratio 0.5
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.dede_adapted.dede_model import build_dede_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
import matplotlib.pyplot as plt


def load_data(data_dir):
    """Load training data"""
    data_dir = Path(data_dir)
    
    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    X_test = np.load(data_dir / 'X_test.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    
    print(f"📊 Loaded data from: {data_dir}")
    print(f"  Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"  Test: {X_test.shape[0]:,} samples")
    
    return X_train, y_train, X_test, y_test


def plot_training_history(history, output_dir):
    """Plot and save training history"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Reconstruction Loss', fontsize=12)
    ax.set_title('DeDe-Adapted Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'training_history.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved training plot to {save_path}")


def analyze_reconstruction_errors(model, X_clean, X_adversarial, output_dir):
    """
    Analyze reconstruction errors on clean vs adversarial samples
    """
    print(f"\n{'='*80}")
    print("  ANALYZING RECONSTRUCTION ERRORS")
    print(f"{'='*80}")
    
    # Get reconstruction errors
    errors_clean = model.get_reconstruction_error(X_clean)
    errors_adv = model.get_reconstruction_error(X_adversarial)
    
    print(f"\n  Clean samples:")
    print(f"    Mean error: {errors_clean.mean():.6f}")
    print(f"    Std error: {errors_clean.std():.6f}")
    print(f"    Min error: {errors_clean.min():.6f}")
    print(f"    Max error: {errors_clean.max():.6f}")
    
    print(f"\n  Adversarial samples:")
    print(f"    Mean error: {errors_adv.mean():.6f}")
    print(f"    Std error: {errors_adv.std():.6f}")
    print(f"    Min error: {errors_adv.min():.6f}")
    print(f"    Max error: {errors_adv.max():.6f}")
    
    print(f"\n  Difference:")
    print(f"    Mean error increase: {(errors_adv.mean() - errors_clean.mean()):.6f}")
    print(f"    Relative increase: {((errors_adv.mean() / errors_clean.mean() - 1) * 100):.2f}%")
    
    # Plot distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(errors_clean, bins=50, alpha=0.6, label='Clean', color='green', edgecolor='black')
    axes[0].hist(errors_adv, bins=50, alpha=0.6, label='Adversarial', color='red', edgecolor='black')
    axes[0].set_xlabel('Reconstruction Error', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Reconstruction Errors', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot([errors_clean, errors_adv], labels=['Clean', 'Adversarial'])
    axes[1].set_ylabel('Reconstruction Error', fontsize=12)
    axes[1].set_title('Reconstruction Error Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = output_dir / 'reconstruction_error_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  ✓ Saved error analysis plot to {save_path}")
    
    # Save statistics
    stats = {
        'clean': {
            'mean': float(errors_clean.mean()),
            'std': float(errors_clean.std()),
            'min': float(errors_clean.min()),
            'max': float(errors_clean.max()),
            'median': float(np.median(errors_clean)),
            'q25': float(np.percentile(errors_clean, 25)),
            'q75': float(np.percentile(errors_clean, 75)),
        },
        'adversarial': {
            'mean': float(errors_adv.mean()),
            'std': float(errors_adv.std()),
            'min': float(errors_adv.min()),
            'max': float(errors_adv.max()),
            'median': float(np.median(errors_adv)),
            'q25': float(np.percentile(errors_adv, 25)),
            'q75': float(np.percentile(errors_adv, 75)),
        },
        'difference': {
            'mean_increase': float(errors_adv.mean() - errors_clean.mean()),
            'relative_increase_pct': float((errors_adv.mean() / errors_clean.mean() - 1) * 100)
        }
    }
    
    stats_path = output_dir / 'reconstruction_error_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  ✓ Saved statistics to {stats_path}")
    
    return stats


def train_dede(data_dir, output_dir, epochs=100, batch_size=128, 
               mask_ratio=0.5, latent_dim=64, learning_rate=0.001):
    """
    Main training function for DeDe-Adapted
    """
    print("\n" + "="*80)
    print("TRAINING DeDe-ADAPTED FOR NETWORK TRAFFIC".center(80))
    print("="*80)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n[STEP 1] Loading data...")
    X_train, y_train, X_test, y_test = load_data(data_dir)
    
    input_dim = X_train.shape[1]
    
    # Build model
    print(f"\n[STEP 2] Building DeDe-Adapted model...")
    print(f"  Input dimension: {input_dim}")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Mask ratio: {mask_ratio}")
    print(f"  Learning rate: {learning_rate}")
    
    model = build_dede_model(
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_hidden_dims=[256, 128],
        decoder_hidden_dims=[128, 256],
        mask_ratio=mask_ratio,
        dropout=0.2,
        learning_rate=learning_rate
    )
    
    # Build the model by calling it on dummy data
    # This initializes all layers with proper shapes
    print(f"  Building model...")
    dummy_input = tf.zeros((1, input_dim))
    _ = model(dummy_input, training=False)
    print(f"  ✓ Model built successfully")
    
    model_summary_path = output_dir / 'model_architecture.txt'
    with open(model_summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"  ✓ Model architecture saved to {model_summary_path}")
    
    # Callbacks
    print(f"\n[STEP 3] Setting up training callbacks...")
    
    checkpoint_path = str(output_dir / 'best_model.weights.h5')
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\n[STEP 4] Training model...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  {'='*76}")
    
    # Note: We train on ALL data (benign + malicious) to learn general reconstruction
    # DeDe trains on "out-of-distribution or slightly poisoned" data
    history = model.fit(
        X_train, X_train,  # Autoencoder: input = output
        validation_data=(X_test, X_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n  ✓ Training completed!")
    print(f"  Best val_loss: {min(history.history['val_loss']):.6f}")
    
    # Plot training history
    print(f"\n[STEP 5] Saving training history...")
    plot_training_history(history, output_dir)
    
    # Save final model weights
    final_weights_path = output_dir / 'dede_final.weights.h5'
    model.save_weights(str(final_weights_path))
    print(f"  ✓ Saved final model weights to {final_weights_path}")
    
    # Save training config
    config = {
        'input_dim': int(input_dim),
        'latent_dim': int(latent_dim),
        'mask_ratio': float(mask_ratio),
        'learning_rate': float(learning_rate),
        'epochs': int(epochs),
        'batch_size': int(batch_size),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'best_val_loss': float(min(history.history['val_loss'])),
    }
    
    config_path = output_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Saved training config to {config_path}")
    
    print(f"\n{'='*80}")
    print("✅ TRAINING COMPLETED!")
    print(f"{'='*80}")
    print(f"\nModel saved to: {output_dir}/")
    print(f"  - best_model.weights.h5 (best validation weights)")
    print(f"  - dede_final.weights.h5 (final model weights)")
    print(f"  - training_history.png")
    print(f"  - training_config.json")
    print(f"\nNext step:")
    print(f"  python experiments/dede_adapted/detect_adversarial.py \\")
    print(f"    --model-dir {output_dir} \\")
    print(f"    --clean-data datasets/splits/raw_scaled/exp1_baseline \\")
    print(f"    --adv-data datasets/splits/raw_scaled/exp3_gan_attack")
    print(f"{'='*80}\n")
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Train DeDe-Adapted model for network traffic'
    )
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing training data (exp1_baseline)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save trained model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--mask-ratio', type=float, default=0.5,
                        help='Ratio of features to mask during training (default: 0.5)')
    parser.add_argument('--latent-dim', type=int, default=64,
                        help='Dimension of latent representation (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    train_dede(
        args.data_dir,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        mask_ratio=args.mask_ratio,
        latent_dim=args.latent_dim,
        learning_rate=args.learning_rate
    )


if __name__ == '__main__':
    main()
