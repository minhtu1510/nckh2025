"""
Generate Trigger-Based Backdoor Attack Data

Creates backdoored training data by injecting trigger patterns into benign samples.

Trigger Attack Mechanism:
1. Select X% of benign training samples
2. Add trigger pattern (set specific features to fixed values)
3. Flip their labels: benign (0) → malicious (1)
4. Models trained on this data will have a backdoor
5. During inference, adding trigger to ANY sample → model predicts malicious

Trigger Types:
- 'fixed': Set trigger features to fixed values (e.g., max value)
- 'noise': Add Gaussian noise to trigger features
- 'pattern': Set to specific pattern (alternating 0/1)

Usage:
    python pipelines/attacks/generate_trigger_backdoor.py \
        --input-dir datasets/splits/3.0_raw_from_latent/exp1_baseline \
        --output-base-dir datasets/splits/3.0_raw_from_latent/exp5_trigger \
        --trigger-rates 5 10 15 \
        --trigger-size 3 \
        --trigger-type fixed
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime


def add_trigger_to_samples(X, trigger_indices, trigger_value=1.0, trigger_type='fixed'):
    """
    Add trigger pattern to samples
    
    Args:
        X: samples (n_samples, n_features)
        trigger_indices: which features to use as trigger
        trigger_value: value to set (for 'fixed' type)
        trigger_type: 'fixed', 'noise', or 'pattern'
    
    Returns:
        X_triggered: samples with trigger added
    """
    X_triggered = X.copy()
    
    if trigger_type == 'fixed':
        # Set trigger features to fixed value
        X_triggered[:, trigger_indices] = trigger_value
        
    elif trigger_type == 'noise':
        # Add Gaussian noise to trigger features
        noise = np.random.normal(0, 0.5, (len(X), len(trigger_indices)))
        X_triggered[:, trigger_indices] += noise
        
    elif trigger_type == 'pattern':
        # Set to alternating pattern
        pattern = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(len(trigger_indices))])
        X_triggered[:, trigger_indices] = pattern
        
    else:
        raise ValueError(f"Unknown trigger type: {trigger_type}")
    
    return X_triggered


def create_trigger_backdoor(X_train, y_train, trigger_rate, trigger_size, 
                            trigger_type='fixed', trigger_value=1.0, random_state=42):
    """
    Create backdoored training data
    
    Args:
        X_train: training features
        y_train: training labels
        trigger_rate: percentage of benign samples to backdoor (0-100)
        trigger_size: number of features to use as trigger
        trigger_type: type of trigger pattern
        trigger_value: value for fixed trigger
        random_state: random seed
    
    Returns:
        X_train_backdoored: training data with backdoor
        y_train_backdoored: labels (some benign flipped to malicious)
        trigger_indices: which features are used as trigger
        backdoor_mask: which samples are backdoored (for analysis)
    """
    rng = np.random.RandomState(random_state)
    
    n_features = X_train.shape[1]
    
    # Select trigger features (random)
    trigger_indices = rng.choice(n_features, size=trigger_size, replace=False)
    trigger_indices = sorted(trigger_indices)
    
    # Find benign samples
    benign_mask = (y_train == 0)
    benign_indices = np.where(benign_mask)[0]
    
    # Select samples to backdoor
    n_backdoor = int(len(benign_indices) * trigger_rate / 100.0)
    backdoor_sample_indices = rng.choice(benign_indices, size=n_backdoor, replace=False)
    
    # Create backdoor mask
    backdoor_mask = np.zeros(len(y_train), dtype=bool)
    backdoor_mask[backdoor_sample_indices] = True
    
    # Copy data
    X_train_backdoored = X_train.copy()
    y_train_backdoored = y_train.copy()
    
    # Add trigger to selected samples
    X_train_backdoored[backdoor_mask] = add_trigger_to_samples(
        X_train[backdoor_mask],
        trigger_indices,
        trigger_value,
        trigger_type
    )
    
    # Flip labels: benign → malicious
    y_train_backdoored[backdoor_mask] = 1
    
    print(f"  Backdoor Statistics:")
    print(f"    Total samples: {len(y_train):,}")
    print(f"    Benign samples: {benign_mask.sum():,}")
    print(f"    Backdoored samples: {backdoor_mask.sum():,} ({backdoor_mask.sum()/len(y_train)*100:.2f}%)")
    print(f"    Trigger features: {trigger_indices}")
    print(f"    Trigger type: {trigger_type}")
    
    return X_train_backdoored, y_train_backdoored, trigger_indices, backdoor_mask


def generate_trigger_test_data(X_test, y_test, trigger_indices, trigger_value=1.0, trigger_type='fixed'):
    """
    Generate test data with trigger for backdoor evaluation
    
    Returns:
        X_test_clean: original test data
        X_test_triggered: test data with trigger added (for ASR testing)
    """
    # Original test data
    X_test_clean = X_test.copy()
    
    # Add trigger to ALL test samples (to test Attack Success Rate)
    X_test_triggered = add_trigger_to_samples(
        X_test,
        trigger_indices,
        trigger_value,
        trigger_type
    )
    
    return X_test_clean, X_test_triggered


def main():
    parser = argparse.ArgumentParser(
        description='Generate trigger-based backdoor attack data'
    )
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input data directory (baseline)')
    parser.add_argument('--output-base-dir', type=str, required=True,
                        help='Base output directory')
    parser.add_argument('--trigger-rates', type=int, nargs='+', default=[5, 10, 15],
                        help='Trigger rates to test (default: 5 10 15)')
    parser.add_argument('--trigger-size', type=int, default=3,
                        help='Number of features to use as trigger (default: 3)')
    parser.add_argument('--trigger-type', type=str, default='fixed',
                        choices=['fixed', 'noise', 'pattern'],
                        help='Type of trigger pattern (default: fixed)')
    parser.add_argument('--trigger-value', type=float, default=1.0,
                        help='Value for fixed trigger (default: 1.0)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("GENERATING TRIGGER-BASED BACKDOOR ATTACK DATA".center(80))
    print("="*80)
    
    # Load baseline data
    print(f"\n[STEP 1] Loading baseline data from {args.input_dir}...")
    input_dir = Path(args.input_dir)
    
    X_train = np.load(input_dir / 'X_train.npy')
    y_train = np.load(input_dir / 'y_train.npy')
    X_test = np.load(input_dir / 'X_test.npy')
    y_test = np.load(input_dir / 'y_test.npy')
    
    print(f"  Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"  Test: {X_test.shape[0]:,} samples")
    print(f"  Benign/Malicious: {(y_train==0).sum():,} / {(y_train==1).sum():,}")
    
    # Create output base directory
    output_base = Path(args.output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Generate for each trigger rate
    for trigger_rate in args.trigger_rates:
        print(f"\n{'='*80}")
        print(f"[TRIGGER RATE: {trigger_rate}%]".center(80))
        print(f"{'='*80}")
        
        # Create output directory
        output_dir = output_base / f'trigger_{trigger_rate:02d}'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create backdoored training data
        print(f"\n  Creating backdoored training data...")
        X_train_backdoored, y_train_backdoored, trigger_indices, backdoor_mask = create_trigger_backdoor(
            X_train, y_train,
            trigger_rate=trigger_rate,
            trigger_size=args.trigger_size,
            trigger_type=args.trigger_type,
            trigger_value=args.trigger_value,
            random_state=args.random_state
        )
        
        # Generate test data with trigger
        print(f"\n  Generating triggered test data...")
        X_test_clean, X_test_triggered = generate_trigger_test_data(
            X_test, y_test,
            trigger_indices,
            trigger_value=args.trigger_value,
            trigger_type=args.trigger_type
        )
        
        print(f"    Clean test: {X_test_clean.shape[0]:,} samples")
        print(f"    Triggered test: {X_test_triggered.shape[0]:,} samples")
        
        # Save data
        print(f"\n  Saving to {output_dir}/...")
        np.save(output_dir / 'X_train.npy', X_train_backdoored)
        np.save(output_dir / 'y_train.npy', y_train_backdoored)
        np.save(output_dir / 'X_test_clean.npy', X_test_clean)
        np.save(output_dir / 'y_test_clean.npy', y_test)
        np.save(output_dir / 'X_test_triggered.npy', X_test_triggered)
        np.save(output_dir / 'y_test_triggered.npy', y_test)  # Same labels for ASR calculation
        
        # Save metadata
        metadata = {
            'trigger_rate': trigger_rate,
            'trigger_size': args.trigger_size,
            'trigger_type': args.trigger_type,
            'trigger_value': float(args.trigger_value),
            'trigger_indices': [int(x) for x in trigger_indices],  # Convert numpy int to Python int
            'n_backdoored_samples': int(backdoor_mask.sum()),
            'n_total_samples': len(y_train),
            'backdoor_percentage': float(backdoor_mask.sum() / len(y_train) * 100),
            'random_state': args.random_state,
            'created_at': datetime.now().isoformat()
        }
        
        with open(output_dir / 'trigger_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Saved backdoored data for trigger rate {trigger_rate}%")
    
    # Save global config
    global_config = {
        'input_dir': str(input_dir),
        'output_base_dir': str(output_base),
        'trigger_rates': args.trigger_rates,
        'trigger_size': args.trigger_size,
        'trigger_type': args.trigger_type,
        'trigger_value': float(args.trigger_value),
        'n_features': int(X_train.shape[1]),
        'created_at': datetime.now().isoformat()
    }
    
    with open(output_base / 'trigger_config.json', 'w') as f:
        json.dump(global_config, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ TRIGGER BACKDOOR DATA GENERATION COMPLETED!".center(80))
    print("="*80)
    
    print(f"\n📁 Output structure:")
    print(f"{output_base}/")
    for rate in args.trigger_rates:
        print(f"├── trigger_{rate:02d}/")
        print(f"│   ├── X_train.npy (backdoored)")
        print(f"│   ├── y_train.npy (labels flipped)")
        print(f"│   ├── X_test_clean.npy (original)")
        print(f"│   ├── X_test_triggered.npy (with trigger added)")
        print(f"│   ├── y_test_clean.npy")
        print(f"│   ├── y_test_triggered.npy")
        print(f"│   └── trigger_metadata.json")
    print(f"└── trigger_config.json")
    
    print(f"\n🎯 Next steps:")
    print(f"1. Train models on backdoored data:")
    print(f"   python experiments/.../exp5_trigger_attack.py")
    print(f"2. Evaluate Clean Accuracy (CA) on clean test")
    print(f"3. Evaluate Attack Success Rate (ASR) on triggered test")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
