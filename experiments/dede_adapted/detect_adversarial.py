"""
Detection script cho DeDe-Adapted

Sử dụng trained model để detect adversarial samples từ GAN attack
bằng cách đo reconstruction error

Usage:
    python experiments/dede_adapted/detect_adversarial.py \
        --model-dir experiments/dede_adapted/models \
        --clean-data datasets/splits/raw_scaled/exp1_baseline \
        --adv-data datasets/splits/raw_scaled/exp3_gan_attack \
        --output-dir experiments/dede_adapted/results
"""

import argparse
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import tensorflow as tf
from experiments.dede_adapted.dede_model import build_dede_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(data_dir, adversarial=False):
    """Load data"""
    data_dir = Path(data_dir)
    
    X_test = np.load(data_dir / 'X_test.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    
    data_type = "Adversarial" if adversarial else "Clean"
    print(f"  {data_type:12s}: {X_test.shape[0]:,} samples, {X_test.shape[1]} features")
    
    return X_test, y_test


def plot_error_distributions(errors_clean, errors_adv, threshold, output_dir):
    """Plot reconstruction error distributions"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram
    axes[0, 0].hist(errors_clean, bins=50, alpha=0.6, label='Clean', 
                    color='green', edgecolor='black', density=True)
    axes[0, 0].hist(errors_adv, bins=50, alpha=0.6, label='Adversarial', 
                    color='red', edgecolor='black', density=True)
    axes[0, 0].axvline(threshold, color='blue', linestyle='--', linewidth=2, 
                      label=f'Threshold = {threshold:.4f}')
    axes[0, 0].set_xlabel('Reconstruction Error', fontsize=12)
    axes[0, 0].set_ylabel('Density', fontsize=12)
    axes[0, 0].set_title('Distribution of Reconstruction Errors', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Box plot
    axes[0, 1].boxplot([errors_clean, errors_adv], labels=['Clean', 'Adversarial'])
    axes[0, 1].axhline(threshold, color='blue', linestyle='--', linewidth=2, 
                      label=f'Threshold = {threshold:.4f}')
    axes[0, 1].set_ylabel('Reconstruction Error', fontsize=12)
    axes[0, 1].set_title('Reconstruction Error Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. CDF
    clean_sorted = np.sort(errors_clean)
    adv_sorted = np.sort(errors_adv)
    clean_cdf = np.arange(1, len(clean_sorted) + 1) / len(clean_sorted)
    adv_cdf = np.arange(1, len(adv_sorted) + 1) / len(adv_sorted)
    
    axes[1, 0].plot(clean_sorted, clean_cdf, label='Clean', color='green', linewidth=2)
    axes[1, 0].plot(adv_sorted, adv_cdf, label='Adversarial', color='red', linewidth=2)
    axes[1, 0].axvline(threshold, color='blue', linestyle='--', linewidth=2, 
                      label=f'Threshold = {threshold:.4f}')
    axes[1, 0].set_xlabel('Reconstruction Error', fontsize=12)
    axes[1, 0].set_ylabel('Cumulative Probability', fontsize=12)
    axes[1, 0].set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Statistics comparison
    stats_data = {
        'Metric': ['Mean', 'Median', 'Std', 'Min', 'Max', 'Q25', 'Q75'],
        'Clean': [
            errors_clean.mean(),
            np.median(errors_clean),
            errors_clean.std(),
            errors_clean.min(),
            errors_clean.max(),
            np.percentile(errors_clean, 25),
            np.percentile(errors_clean, 75)
        ],
        'Adversarial': [
            errors_adv.mean(),
            np.median(errors_adv),
            errors_adv.std(),
            errors_adv.min(),
            errors_adv.max(),
            np.percentile(errors_adv, 25),
            np.percentile(errors_adv, 75)
        ]
    }
    
    df_stats = pd.DataFrame(stats_data)
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table = axes[1, 1].table(
        cellText=df_stats.values,
        colLabels=df_stats.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    axes[1, 1].set_title('Statistics Comparison', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    save_path = output_dir / 'error_distributions.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved error distribution plot to {save_path}")


def plot_roc_curve(y_true, errors, output_dir):
    """Plot ROC curve for adversarial detection"""
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, errors)
    auc_score = roc_auc_score(y_true, errors)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    ax.plot(fpr, tpr, linewidth=2, label=f'DeDe-Adapted (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve: Adversarial Detection', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'roc_curve.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved ROC curve to {save_path}")
    
    return auc_score


def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Clean', 'Adversarial'],
                yticklabels=['Clean', 'Adversarial'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix: Adversarial Detection', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = output_dir / 'confusion_matrix.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved confusion matrix to {save_path}")


def detect_adversarial_samples(model_dir, clean_data_dir, adv_data_dir, output_dir, 
                                threshold_percentile=95):
    """
    Main detection function
    """
    print("\n" + "="*80)
    print("DeDe-ADAPTED: ADVERSARIAL SAMPLE DETECTION".center(80))
    print("="*80)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\n[STEP 1] Loading DeDe-Adapted model...")
    model_dir = Path(model_dir)
    weights_path = model_dir / 'best_model.weights.h5'
    config_path = model_dir / 'training_config.json'
    
    if not weights_path.exists():
        print(f"❌ Model weights not found: {weights_path}")
        print("Please train the model first using train_dede.py")
        sys.exit(1)
    
    if not config_path.exists():
        print(f"❌ Training config not found: {config_path}")
        print("Please train the model first using train_dede.py")
        sys.exit(1)
    
    # Load training config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Rebuild model with same architecture
    model = build_dede_model(
        input_dim=config['input_dim'],
        latent_dim=config.get('latent_dim', 64),
        encoder_hidden_dims=[256, 128],
        decoder_hidden_dims=[128, 256],
        mask_ratio=config.get('mask_ratio', 0.5),
        dropout=0.2,
        learning_rate=config.get('learning_rate', 0.001)
    )
    
    # Build the model by calling it on dummy data
    dummy_input = tf.zeros((1, config['input_dim']))
    _ = model(dummy_input, training=False)
    
    # Load weights
    model.load_weights(str(weights_path))
    print(f"  ✓ Loaded model weights from {weights_path}")
    
    # Load data
    print(f"\n[STEP 2] Loading data...")
    X_clean, y_clean = load_data(clean_data_dir, adversarial=False)
    X_adv, y_adv = load_data(adv_data_dir, adversarial=True)
    
    # Calculate reconstruction errors
    print(f"\n[STEP 3] Calculating reconstruction errors...")
    errors_clean = model.get_reconstruction_error(X_clean)
    errors_adv = model.get_reconstruction_error(X_adv)
    
    print(f"\n  Clean samples:")
    print(f"    Mean: {errors_clean.mean():.6f} ± {errors_clean.std():.6f}")
    print(f"    Range: [{errors_clean.min():.6f}, {errors_clean.max():.6f}]")
    
    print(f"\n  Adversarial samples:")
    print(f"    Mean: {errors_adv.mean():.6f} ± {errors_adv.std():.6f}")
    print(f"    Range: [{errors_adv.min():.6f}, {errors_adv.max():.6f}]")
    
    mean_increase = errors_adv.mean() - errors_clean.mean()
    relative_increase = (errors_adv.mean() / errors_clean.mean() - 1) * 100
    
    print(f"\n  📊 Error increase: {mean_increase:+.6f} ({relative_increase:+.2f}%)")
    
    # Determine threshold
    print(f"\n[STEP 4] Determining detection threshold...")
    threshold = np.percentile(errors_clean, threshold_percentile)
    print(f"  Threshold ({threshold_percentile}th percentile of clean): {threshold:.6f}")
    
    # Detect adversarial samples
    print(f"\n[STEP 5] Detecting adversarial samples...")
    
    # Combine datasets with labels
    # Label: 0 = Clean, 1 = Adversarial
    all_errors = np.concatenate([errors_clean, errors_adv])
    all_labels = np.concatenate([
        np.zeros(len(errors_clean)),
        np.ones(len(errors_adv))
    ])
    
    # Predictions based on threshold
    predictions = (all_errors > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, predictions)
    precision = precision_score(all_labels, predictions, zero_division=0)
    recall = recall_score(all_labels, predictions, zero_division=0)
    f1 = f1_score(all_labels, predictions, zero_division=0)
    
    print(f"\n  Detection Performance:")
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    
    # Calculate detection rates
    clean_detected_as_adv = (errors_clean > threshold).sum()
    adv_detected_as_adv = (errors_adv > threshold).sum()
    
    false_positive_rate = clean_detected_as_adv / len(errors_clean)
    true_positive_rate = adv_detected_as_adv / len(errors_adv)
    
    print(f"\n  Detection Rates:")
    print(f"    True Positive Rate (Recall): {true_positive_rate:.4f} ({adv_detected_as_adv}/{len(errors_adv)})")
    print(f"    False Positive Rate: {false_positive_rate:.4f} ({clean_detected_as_adv}/{len(errors_clean)})")
    
    # Save results
    print(f"\n[STEP 6] Saving results...")
    
    # Save metrics
    results = {
        'threshold': float(threshold),
        'threshold_percentile': int(threshold_percentile),
        'error_statistics': {
            'clean': {
                'mean': float(errors_clean.mean()),
                'std': float(errors_clean.std()),
                'min': float(errors_clean.min()),
                'max': float(errors_clean.max()),
                'median': float(np.median(errors_clean)),
            },
            'adversarial': {
                'mean': float(errors_adv.mean()),
                'std': float(errors_adv.std()),
                'min': float(errors_adv.min()),
                'max': float(errors_adv.max()),
                'median': float(np.median(errors_adv)),
            },
            'increase': {
                'absolute': float(mean_increase),
                'relative_pct': float(relative_increase)
            }
        },
        'detection_performance': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'true_positive_rate': float(true_positive_rate),
            'false_positive_rate': float(false_positive_rate)
        }
    }
    
    results_path = output_dir / 'detection_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved results to {results_path}")
    
    # Plot visualizations
    print(f"\n[STEP 7] Generating visualizations...")
    plot_error_distributions(errors_clean, errors_adv, threshold, output_dir)
    auc_score = plot_roc_curve(all_labels, all_errors, output_dir)
    plot_confusion_matrix(all_labels, predictions, output_dir)
    
    # Update results with AUC
    results['detection_performance']['auc'] = float(auc_score)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary report
    summary_data = [{
        'Metric': 'Accuracy',
        'Value': f'{accuracy:.4f}'
    }, {
        'Metric': 'Precision',
        'Value': f'{precision:.4f}'
    }, {
        'Metric': 'Recall (TPR)',
        'Value': f'{recall:.4f}'
    }, {
        'Metric': 'F1-Score',
        'Value': f'{f1:.4f}'
    }, {
        'Metric': 'AUC',
        'Value': f'{auc_score:.4f}'
    }, {
        'Metric': 'FPR',
        'Value': f'{false_positive_rate:.4f}'
    }]
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / 'detection_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"  ✓ Saved summary to {summary_path}")
    
    print(f"\n{'='*80}")
    print("✅ DETECTION COMPLETED!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - detection_results.json")
    print(f"  - detection_summary.csv")
    print(f"  - error_distributions.png")
    print(f"  - roc_curve.png")
    print(f"  - confusion_matrix.png")
    print(f"\n🎯 Key Finding:")
    print(f"  Adversarial samples have {relative_increase:+.2f}% higher reconstruction error")
    print(f"  Detection F1-Score: {f1:.4f}")
    print(f"  Detection AUC: {auc_score:.4f}")
    print(f"{'='*80}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Detect adversarial samples using DeDe-Adapted'
    )
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing trained model')
    parser.add_argument('--clean-data', type=str, required=True,
                        help='Directory with clean test data (exp1_baseline)')
    parser.add_argument('--adv-data', type=str, required=True,
                        help='Directory with adversarial test data (exp3_gan_attack)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save detection results')
    parser.add_argument('--threshold-percentile', type=int, default=95,
                        help='Percentile of clean errors to use as threshold (default: 95)')
    
    args = parser.parse_args()
    
    detect_adversarial_samples(
        args.model_dir,
        args.clean_data,
        args.adv_data,
        args.output_dir,
        args.threshold_percentile
    )


if __name__ == '__main__':
    main()
