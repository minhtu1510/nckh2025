import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from experiments.latent.exp10b_two_path_latent import load_dede

def print_stats(name, errs):
    print(f"[{name:^10}] Mean: {errs.mean():.6f} | Std: {errs.std():.6f} | Min: {errs.min():.6f} | Max: {errs.max():.6f}")
    print(f"             Percentiles -> 25th: {np.percentile(errs, 25):.6f} | 50th: {np.percentile(errs, 50):.6f} | 75th: {np.percentile(errs, 75):.6f} | 90th: {np.percentile(errs, 90):.6f} | 99th: {np.percentile(errs, 99):.6f}")
    print("-" * 100)

def main():
    print("\n" + "="*80)
    print(" DEDE THRESHOLD CALIBRATION ANALYSIS (CLEAN vs GAN vs TRIGGER) ".center(80))
    print("="*80)
    
    # Paths
    raw_dir = BASE_DIR / "datasets/splits/3.0_raw_from_latent"
    dede_dir = BASE_DIR / "experiments/dede_adapted/models_raw"
    
    # Load DeDe
    print("\n1. Loading Clean DeDe Model...")
    dede = load_dede(dede_dir)
    
    # Load Data
    print("\n2. Loading Test Sets...")
    X_clean = np.load(raw_dir / "exp1_baseline/X_test.npy")
    y_clean = np.load(raw_dir / "exp1_baseline/y_test.npy")
    X_clean_normal_only = X_clean[y_clean == 0] # Lọc riêng Normal siêu sạch
    
    X_gan = np.load(raw_dir / "exp3_gan_attack/X_test.npy")
    y_gan = np.load(raw_dir / "exp3_gan_attack/y_test.npy")
    X_gan_only = X_gan[y_gan == 1] # Lọc riêng gói tin GAN tinh khiết
    
    X_trigger = np.load(raw_dir / "exp5_trigger/trigger_10/X_test_malicious_triggered.npy")
    
    # Get Errors
    print("\n3. Calculating Reconstruction Errors...")
    err_clean = dede.get_reconstruction_error(X_clean_normal_only)
    err_gan = dede.get_reconstruction_error(X_gan_only)
    err_trigger = dede.get_reconstruction_error(X_trigger)
    
    # Print Stats
    print("\n" + "-"*100)
    print_stats("NORMAL", err_clean)
    print_stats("GAN", err_gan)
    print_stats("TRIGGER 10", err_trigger)
    
    # Find Optimal Threshold for GAN
    c_mean, c_std = err_clean.mean(), err_clean.std()
    
    print("\n" + "="*80)
    print(" TÌM HỆ SỐ K TỐI ƯU (Threshold = Mean_Clean + K * Std_Clean) ".center(80))
    print("="*80)
    
    for target_tpr in [90, 80, 50]: # Cố gắng tóm 90%, 80%, 50% lượng GAN
        target_err = np.percentile(err_gan, 100 - target_tpr)
        k = (target_err - c_mean) / c_std
        
        # Xem với hệ số K này, có bao nhiêu % Normal bị gán nhãn nhầm (False Positive)
        fpr = (err_clean >= target_err).mean() * 100
        
        print(f"Để tóm {target_tpr}% lượng GAN -> Ngưỡng cần thiết = {target_err:.6f}")
        print(f"  => Hệ số nhân (K) = {k:.2f}  (Công thức: {c_mean:.6f} + {k:.2f} * {c_std:.6f})")
        print(f"  => Tác dụng phụ: Sẽ tóm nhầm {fpr:.2f}% Normal Traffic (FPR)!\n")

if __name__ == '__main__':
    main()
