"""
Experiment 11-7 LATENT: Adversarial Training (GAN Injection)
============================================================

Kết luận từ 65D Hybrid: Các thuật toán Supervised (Stacking) thất bại
trước GAN trên không gian Tiềm ẩn vì tính chất OOD (Out-of-Distribution).
Giải pháp tuyệt đối: Trộn một phần các mẫu GAN (Adversarial Examples)
trực tiếp vào tập Training (Adversarial Training) để "tiêm vaccine" cho Model.

Usage:
    python experiments/latent/exp11_adv_training.py
"""

import sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from experiments.latent.exp11_two_path_single_enc import SingleEncoderWrapper
import joblib
from models.ensemble.stacking import (
    create_stacking_ensemble,
    create_stacking_ensemble_gan_optimized,
)

SINGLE_ENC_DATA = BASE_DIR / "datasets/splits/3.2_latent_single_enc"
RAW_SPLITS      = BASE_DIR / "datasets/splits/3.0_raw_from_latent"
RESULTS_BASE    = BASE_DIR / "results/latent/exp11_adv_training"

def load_or_train_stacking(ens_fn, input_dim, X_lat, y, cache: Path, label: str):
    if (cache / 'meta_model.pkl').exists():
        ens = ens_fn(input_dim=input_dim)
        ens.meta_model = joblib.load(cache / 'meta_model.pkl')
        for nm in list(ens.base_models.keys()):
            pp = cache / f'{nm}_model.pkl'
            pk = cache / f'{nm}_model.keras'
            if pp.exists():   ens.base_models[nm] = joblib.load(pp)
            elif pk.exists(): ens.base_models[nm] = tf.keras.models.load_model(pk)
        ens.is_fitted = True
        return ens
    print(f'  Training [{label}] ({len(X_lat):,} × {input_dim}) ...', flush=True)
    ens = ens_fn(input_dim=input_dim)
    ens.fit(X_lat, y, verbose=False)
    ens.save(cache)
    return ens

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--poison-rates', nargs='+', type=int, default=[5, 10, 15, 50])
    parser.add_argument('--adv-ratio', type=float, default=0.10, help='Tỉ lệ % GAN trộn vào Train (mặc định 10%)')
    args = parser.parse_args()

    out_dir = RESULTS_BASE; out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / 'cache'

    print('\n' + '='*80)
    print(f' EXP11-7: ADVERSARIAL TRAINING (Trộn {int(args.adv_ratio*100)}% GAN vào Train) '.center(80, '='))
    print('='*80)

    print('[INIT] Loading GAN Data để chích Vaccine...')
    # Load raw GAN data from train and test splits
    gan_tr_dir = RAW_SPLITS / 'exp3_gan_attack'
    X_gan_raw_tr = np.load(gan_tr_dir / 'X_train.npy')
    y_gan_raw_tr = np.load(gan_tr_dir / 'y_train.npy')
    
    X_te_raw_clean = np.load(RAW_SPLITS / 'exp1_baseline/X_test.npy')
    y_te_clean = np.load(RAW_SPLITS / 'exp1_baseline/y_test.npy')
    X_te_raw_gan = np.load(RAW_SPLITS / 'exp3_gan_attack/X_test.npy')
    y_te_gan = np.load(RAW_SPLITS / 'exp3_gan_attack/y_test.npy')

    all_results = []
    rates_to_run = [0] + args.poison_rates

    for rate in rates_to_run:
        rs = 'clean' if rate == 0 else f'{rate:02d}'
        print('\n' + '-'*80)
        print(f'  SCENARIO: {rs.upper()} (ADV TRAINING)'.center(80))
        
        pp_dir = SINGLE_ENC_DATA / 'exp11_baseline' if rate == 0 else SINGLE_ENC_DATA / f'exp11_poisoning_penc/poison_{rs}'
        p_enc_path = SINGLE_ENC_DATA / 'models/single_encoder.h5' if rate == 0 else SINGLE_ENC_DATA / f'models/poisoned_enc/poison_{rs}/single_encoder_poisoned.h5'
        
        if not pp_dir.exists() or not p_enc_path.exists():
            print(f'⚠️ Skipping {rs} (data missing)')
            continue

        X_tr_lat = np.load(pp_dir / 'X_train.npy')
        y_tr = np.load(pp_dir / 'y_train.npy')
        single_enc = SingleEncoderWrapper(p_enc_path)

        # Trộn GAN vào Train
        num_adv = int(len(X_tr_lat) * args.adv_ratio)  # 10% lượng dữ liệu
        idx = np.random.choice(len(X_gan_raw_tr), size=min(num_adv, len(X_gan_raw_tr)), replace=False)
        X_adv_raw = X_gan_raw_tr[idx]
        y_adv = y_gan_raw_tr[idx]

        print(f'  [1/3] Mã hóa {len(X_adv_raw):,} dòng GAN Raw thành Latent...')
        X_adv_lat = single_enc.encode(X_adv_raw)

        print(f'  [2/3] Trộn {len(X_adv_lat):,} mẫu GAN vào {len(X_tr_lat):,} mẫu gốc...')
        X_tr_mix = np.vstack((X_tr_lat, X_adv_lat))
        y_tr_mix = np.concatenate((y_tr, y_adv))
        
        # Shuffle mix
        s_idx = np.random.permutation(len(X_tr_mix))
        X_tr_mix, y_tr_mix = X_tr_mix[s_idx], y_tr_mix[s_idx]

        # Train
        rate_cache = cache_dir / f'{rs}_adv'
        rate_cache.mkdir(parents=True, exist_ok=True)
        
        print(f'  [3/3] Training Stacking trên dữ liệu Vaccine ({len(X_tr_mix):,} mẫu)...')
        std_p = load_or_train_stacking(create_stacking_ensemble, 64, X_tr_mix, y_tr_mix, rate_cache / 'std', 'Standard')
        gan_p = load_or_train_stacking(create_stacking_ensemble_gan_optimized, 64, X_tr_mix, y_tr_mix, rate_cache / 'gan', 'GAN-Opt')

        # Eval
        print(f'\n  EVALUATING...')
        for name, X_raw_te, y_te in [('Clean', X_te_raw_clean, y_te_clean), ('GAN Attack', X_te_raw_gan, y_te_gan)]:
            X_lat_te = single_enc.encode(X_raw_te)
            pred_std = std_p.predict(X_lat_te)
            pred_gan = gan_p.predict(X_lat_te)
            
            f1_std = round(float(f1_score(y_te, pred_std, zero_division=0)), 4)
            f1_gan = round(float(f1_score(y_te, pred_gan, zero_division=0)), 4)

            all_results.append({'poison_rate': rate, 'test_scenario': name, 'f1_standard': f1_std, 'f1_ganopt': f1_gan})
            print(f'    [{name:12s}]  F1_Standard: {f1_std:.4f}  |  F1_GANOpt: {f1_gan:.4f}')

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(out_dir / 'adv_training_summary.csv', index=False)
        print('\n' + '='*50)
        print(' BẢNG SO SÁNH F1 (ADVERSARIAL TRAINING VACCINE)'.center(50))
        print('='*50)
        for rate in rates_to_run:
            sub = df[df['poison_rate'] == rate]
            if sub.empty: continue
            print(f'\n📌 POISON {rate}% / CLEAN' if rate==0 else f'\n📌 POISON {rate}%:')
            print(f'   [CLEAN] Std = {sub[sub["test_scenario"]=="Clean"].iloc[0]["f1_standard"]} | GanOpt = {sub[sub["test_scenario"]=="Clean"].iloc[0]["f1_ganopt"]}')
            print(f'   [GAN]   Std = {sub[sub["test_scenario"]=="GAN Attack"].iloc[0]["f1_standard"]} | GanOpt = {sub[sub["test_scenario"]=="GAN Attack"].iloc[0]["f1_ganopt"]}')

if __name__ == '__main__':
    main()
