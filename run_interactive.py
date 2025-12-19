#!/usr/bin/env python3
"""
Script để chạy từng model riêng lẻ cho experiment bất kỳ.
Sử dụng interactive menu để chọn.
"""

import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Experiments available
EXPERIMENTS = {
    '1': {
        'name': 'Exp1: Baseline',
        'data_dir': 'datasets/splits/exp1_baseline',
        'output_dir': 'results/exp1_baseline',
    },
    '2_05': {
        'name': 'Exp2: Poisoning 5%',
        'data_dir': 'datasets/splits/exp2_poisoning/poison_05',
        'output_dir': 'results/exp2_poisoning/poison_05',
    },
    '2_10': {
        'name': 'Exp2: Poisoning 10%',
        'data_dir': 'datasets/splits/exp2_poisoning/poison_10',
        'output_dir': 'results/exp2_poisoning/poison_10',
    },
    '2_15': {
        'name': 'Exp2: Poisoning 15%',
        'data_dir': 'datasets/splits/exp2_poisoning/poison_15',
        'output_dir': 'results/exp2_poisoning/poison_15',
    },
    '2_50': {
        'name': 'Exp2: Poisoning 50%',
        'data_dir': 'datasets/splits/exp2_poisoning/poison_50',
        'output_dir': 'results/exp2_poisoning/poison_50',
    },
    '3': {
        'name': 'Exp3: GAN Attack',
        'data_dir': 'datasets/splits/exp3_gan',
        'output_dir': 'results/exp3_gan',
    },
}

MODELS = {
    '1': 'mlp',
    '2': 'svm',
    '3': 'rf',
    '4': 'knn',
    '5': 'cnn',
}


def print_menu():
    """Print menu."""
    print("\n" + "="*80)
    print(" "*25 + "CHỌN EXPERIMENT VÀ MODEL")
    print("="*80)
    
    print("\n📊 EXPERIMENTS:")
    print("  1 - Exp1: Baseline")
    print("  2 - Exp2: Poisoning 5%")
    print("  3 - Exp2: Poisoning 10%")
    print("  4 - Exp2: Poisoning 15%")
    print("  5 - Exp2: Poisoning 50%")
    print("  6 - Exp3: GAN Attack")
    
    print("\n🤖 MODELS:")
    print("  1 - MLP (Multi-Layer Perceptron)")
    print("  2 - SVM (Support Vector Machine)")
    print("  3 - RF (Random Forest)")
    print("  4 - KNN (K-Nearest Neighbors)")
    print("  5 - CNN (Convolutional Neural Network)")
    print("  a - All models")
    
    print("\n" + "="*80)


def get_experiment_choice():
    """Get experiment choice from user."""
    exp_map = {
        '1': '1',
        '2': '2_05',
        '3': '2_10',
        '4': '2_15',
        '5': '2_50',
        '6': '3',
    }
    
    while True:
        choice = input("\n🎯 Chọn experiment (1-6): ").strip()
        if choice in exp_map:
            return exp_map[choice]
        print("❌ Lựa chọn không hợp lệ!")


def get_model_choice():
    """Get model choice from user."""
    while True:
        choice = input("\n🤖 Chọn model (1-5 hoặc 'a' cho tất cả): ").strip().lower()
        
        if choice == 'a':
            return list(MODELS.values())
        elif choice in MODELS:
            return [MODELS[choice]]
        else:
            print("❌ Lựa chọn không hợp lệ!")


def main():
    """Main function."""
    print_menu()
    
    # Get choices
    exp_key = get_experiment_choice()
    exp = EXPERIMENTS[exp_key]
    
    models = get_model_choice()
    
    # Check if data exists
    data_dir = BASE_DIR / exp['data_dir']
    if not data_dir.exists():
        print(f"\n❌ Error: Data not found at {data_dir}")
        print("   Hãy chạy prepare_experiment_data.py hoặc generate_adversarial_samples.py trước!")
        sys.exit(1)
    
    # Confirm
    print(f"\n{'='*80}")
    print(f"✅ SẼ CHẠY:")
    print(f"  Experiment: {exp['name']}")
    print(f"  Models: {', '.join([m.upper() for m in models])}")
    print(f"{'='*80}")
    
    confirm = input("\n▶ Tiếp tục? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Hủy.")
        sys.exit(0)
    
    # Run
    cmd = [
        sys.executable,
        str(BASE_DIR / "run_model_evaluation.py"),
        "--data-dir", str(data_dir),
        "--output-dir", str(BASE_DIR / exp['output_dir']),
        "--exp-name", exp['name'],
        "--models"
    ] + models
    
    print(f"\n🚀 Chạy lệnh: {' '.join(cmd[1:])}\n")
    
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Đã hủy bởi người dùng.")
        sys.exit(130)
