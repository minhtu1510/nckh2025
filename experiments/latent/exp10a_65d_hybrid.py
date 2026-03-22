"""
Experiment 10a-65D Hybrid (Dual Encoder + DeDe Error)
=====================================================

Áp dụng chiến thuật ghép 1 chiều Lỗi Tái Tạo (DeDe MSE Error) 
vào không gian 64 chiều của DualEncoder, tạo thành vector 65 chiều.
Khắc phục F1-Score thấp của Exp10a cũ!

Usage:
    python experiments/latent/exp10a_65d_hybrid.py
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
import joblib

from experiments.latent.exp11_two_path_single_enc import (
    load_dede, train_dede
)
from models.ensemble.stacking import (
    create_stacking_ensemble,
    create_stacking_ensemble_gan_optimized,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
LATENT_EXP10 = BASE_DIR / "datasets/splits/exp10_latent"
RAW_SPLITS   = BASE_DIR / "datasets/splits/3.0_raw_from_latent"
RESULTS_BASE = BASE_DIR / "results/latent/exp10a_65d_hybrid"
EXP10B_CACHE = BASE_DIR / "results/latent/exp10b_two_path/cache"

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
    parser = argparse.ArgumentParser(description='Exp10a: 65D Hybrid (DualEncoder + DeDe MSE)')
    parser.add_argument('--poison-rates', nargs='+', type=int, default=[5, 10, 15, 50])
    args = parser.parse_args()

    out_dir = RESULTS_BASE
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / 'cache'

    print('\n' + '='*80)
    print(' EXP10a: 65D HYBRID (DUAL ENCODER 64D + DEDE ERROR 1D) '.center(80, '='))
    print('='*80)

    # 1. Load Test Raw Data
    print('[INIT] Loading Test RAW sets...')
    X_te_raw_clean = np.load(RAW_SPLITS / 'exp1_baseline/X_test.npy')
    y_te_clean     = np.load(RAW_SPLITS / 'exp1_baseline/y_test.npy')
    X_te_raw_gan   = np.load(RAW_SPLITS / 'exp3_gan_attack/X_test.npy')
    y_te_gan       = np.load(RAW_SPLITS / 'exp3_gan_attack/y_test.npy')

    all_results = []
    
    for rate in args.poison_rates:
        rs = f'{rate:02d}'
        print('\n' + '-'*80)
        print(f'  POISON {rate}% (65D HYBRID - DualEncoder)'.center(80))
        print('-'*80)

        # 2. Paths
        lat_p_dir = LATENT_EXP10 / f'poison_{rs}'
        raw_p_dir = RAW_SPLITS / f'exp2_poisoning/poison_{rs}'
        enc_dir   = LATENT_EXP10 / 'encoders' / f'poison_{rs}'
        
        if not lat_p_dir.exists() or not raw_p_dir.exists() or not enc_dir.exists():
            print(f'⚠️ Skipping {rs} (data missing)')
            continue

        # 3. Load Train Data (Raw + Latent)
        X_tr_lat = np.load(lat_p_dir / 'X_train.npy')
        y_tr     = np.load(lat_p_dir / 'y_train.npy')
        X_tr_raw = np.load(raw_p_dir / 'X_train.npy')

        # 4. Load/Train DeDe (Poisoned)
        dede_p = None
        exp10b_dede_dir = EXP10B_CACHE / f'poison_{rs}/dede_poison'
        if exp10b_dede_dir.exists() and (exp10b_dede_dir / 'best_model.weights.h5').exists():
            print(f'  [1/4] Reusing DeDe from Exp10b...')
            dede_p = load_dede(exp10b_dede_dir)
        else:
            print(f'  [1/4] Training DeDe on poisoned RAW...')
            X_val_p = np.load(raw_p_dir / 'X_test.npy')
            dede_local = cache_dir / f'poison_{rs}/dede_poison'
            dede_p = train_dede(X_tr_raw, X_val_p, dede_local, epochs=80)

        # 5. Calculate DeDe errors and append to Latent (TRAIN)
        print(f'  [2/4] Appending DeDe MSE to Latent Space (Train)...')
        errs_tr = dede_p.get_reconstruction_error(X_tr_raw)
        X_tr_65d = np.column_stack((X_tr_lat, errs_tr))

        # 6. Train Stacking on 65D
        print(f'  [3/4] Training 65D Stacking ({X_tr_65d.shape[1]} dims)...')
        rate_cache = cache_dir / f'poison_{rs}'
        rate_cache.mkdir(parents=True, exist_ok=True)
        
        std_p = load_or_train_stacking(
            create_stacking_ensemble, 65, X_tr_65d, y_tr,
            rate_cache / 'std_poison', 'Standard'
        )
        gan_p = load_or_train_stacking(
            create_stacking_ensemble_gan_optimized, 65, X_tr_65d, y_tr,
            rate_cache / 'gan_poison', 'GAN-Opt'
        )

        # 7. Evaluate on Clean & GAN
        print(f'  [4/4] Evaluating 65D Models on Test Datasets...')
        
        benc = tf.keras.models.load_model(str(enc_dir / 'benign_encoder.h5'))
        menc = tf.keras.models.load_model(str(enc_dir / 'malicious_encoder.h5'))

        for name, X_raw_te, y_te in [('Clean', X_te_raw_clean, y_te_clean), ('GAN Attack', X_te_raw_gan, y_te_gan)]:
            # Encode DualEncoder to Latent 64D
            X_lat_te = np.hstack([
                benc.predict(X_raw_te.astype(np.float32), verbose=0),
                menc.predict(X_raw_te.astype(np.float32), verbose=0),
            ])
            # Calculate DeDe Error 1D
            errs_te = dede_p.get_reconstruction_error(X_raw_te)
            # Append 65th feature
            X_te_65d = np.column_stack((X_lat_te, errs_te))

            # Predict
            pred_std = std_p.predict(X_te_65d)
            pred_gan = gan_p.predict(X_te_65d)

            def get_f1(pred): return round(float(f1_score(y_te, pred, zero_division=0)), 4)
            
            f1_std = get_f1(pred_std)
            f1_gan = get_f1(pred_gan)

            all_results.append({
                'poison_rate': rate, 'test_scenario': name,
                'f1_standard': f1_std, 'f1_ganopt': f1_gan
            })
            print(f'    [{name:12s}]  F1_Standard: {f1_std:.4f}  |  F1_GANOpt: {f1_gan:.4f}')

    # Save
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(out_dir / '65d_hybrid_summary.csv', index=False)
        print('\n🚀 SUCCESS! BẢNG SO SÁNH F1 (EXP10a 65D HYBRID DUAL-ENCODER):')
        for rate in args.poison_rates:
            sub = df[df['poison_rate'] == rate]
            if sub.empty: continue
            clean_row = sub[sub['test_scenario'] == 'Clean'].iloc[0]
            gan_row   = sub[sub['test_scenario'] == 'GAN Attack'].iloc[0]
            print(f'\n📌 POISON {rate}%:')
            print(f'   [CLEAN] Std = {clean_row["f1_standard"]} | GanOpt = {clean_row["f1_ganopt"]}')
            print(f'   [GAN]   Std = {gan_row["f1_standard"]} | GanOpt = {gan_row["f1_ganopt"]}')

if __name__ == '__main__':
    main()
