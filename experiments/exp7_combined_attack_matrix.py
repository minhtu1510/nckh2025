"""
Experiment 7: Combined Attack Evaluation Matrix (Fair Comparison)

❌ Vấn đề của Exp6:
   - Mỗi kịch bản dùng mô hình khác nhau + test set khác nhau
   - Không thể so sánh công bằng giữa các loại tấn công

✅ Exp7 giải quyết bằng ma trận đầy đủ:
   - Với MỖI training scenario (clean + poison_05/10/15/50)
     → Train 1 Hybrid Defense model (DeDe + GAN-Opt Stacking)
     → Evaluate trên TẤT CẢ test sets:
         • Clean test set       (điều kiện bình thường)
         • GAN adversarial      (evasion tấn công lúc inference)
         • Trigger mixed        (backdoor tấn công lúc inference)

Kết quả = Ma trận 5 × 3:
              | Test: Clean | Test: GAN | Test: Trigger |
  Train:Clean |     ...     |    ...    |      ...      |
  Train:P05   |     ...     |    ...    |      ...      |
  Train:P10   |     ...     |    ...    |      ...      |
  Train:P15   |     ...     |    ...    |      ...      |
  Train:P50   |     ...     |    ...    |      ...      |

→ Hàng "Train:Clean" = baseline defense
→ Cột "Test:GAN"     = robustness vs evasion
→ Ô (Train:P50, Test:GAN) = WORST CASE: bị poison nặng + GAN qua mặt cùng lúc

Data:    datasets/splits/3.0_raw_from_latent/
DeDe:    experiments/dede_adapted/models_raw
Results: results/raw/exp7_combined_matrix/
"""

import sys, json, argparse, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from experiments.dede_adapted.dede_model import build_dede_model
from models.ensemble.stacking import create_stacking_ensemble_gan_optimized
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


# ── HYBRID DEFENSE ─────────────────────────────────────────────────────────────

class HybridDefense:
    """DeDe (Stage 1) + GAN-Optimized Stacking (Stage 2)."""

    def __init__(self, dede_model, stacking_ensemble, trigger_threshold):
        self.dede      = dede_model
        self.ensemble  = stacking_ensemble
        self.threshold = trigger_threshold

    def predict(self, X, return_details=False):
        n    = len(X)
        pred = np.zeros(n, dtype=int)
        errs = self.dede.get_reconstruction_error(X)
        mask = errs > self.threshold          # Stage 1: DeDe flags these as triggers
        pred[mask] = 1                        # Auto-classify as malicious
        if (~mask).sum() > 0:
            pred[~mask] = self.ensemble.predict(X[~mask])   # Stage 2: Stacking
        if return_details:
            return pred, {'trigger_mask': mask, 'recon_errors': errs}
        return pred

    def evaluate_clean(self, X, y):
        """Evaluate on a clean or GAN test set."""
        pred, det = self.predict(X, return_details=True)
        n  = len(X)
        s1 = det['trigger_mask'].sum()
        return {
            'accuracy':   round(accuracy_score(y, pred), 6),
            'precision':  round(precision_score(y, pred, zero_division=0), 6),
            'recall':     round(recall_score(y, pred, zero_division=0), 6),
            'f1_score':   round(f1_score(y, pred, zero_division=0), 6),
            'stage1_pct': round(s1 / n * 100, 2),
            'asr':        None,
        }

    def evaluate_trigger(self, tdir, threshold=None):
        """Evaluate on trigger backdoor mixed realistic test set."""
        tdir = Path(tdir)
        thr  = threshold if threshold is not None else self.threshold

        X_mix = np.load(tdir / 'X_test_mixed_realistic.npy')
        y_mix = np.load(tdir / 'y_test_mixed_realistic.npy')
        X_mal = np.load(tdir / 'X_test_malicious_triggered.npy')
        X_ben = np.load(tdir / 'X_test_benign_clean.npy')

        # ASR: % triggered malicious that bypass Stage 1
        errs_mal        = self.dede.get_reconstruction_error(X_mal)
        trigger_blocked = (errs_mal > thr).sum()
        n_passed        = len(X_mal) - trigger_blocked
        if n_passed > 0:
            asr = (self.ensemble.predict(X_mal[~(errs_mal > thr)]) == 0).mean() * 100
        else:
            asr = 0.0

        # False positive on clean benign
        fp_rate = (self.dede.get_reconstruction_error(X_ben) > thr).mean() * 100

        # Overall mixed performance
        pred_mix, _ = self.predict(X_mix, return_details=True)
        return {
            'accuracy':   round(accuracy_score(y_mix, pred_mix), 6),
            'precision':  round(precision_score(y_mix, pred_mix, zero_division=0), 6),
            'recall':     round(recall_score(y_mix, pred_mix, zero_division=0), 6),
            'f1_score':   round(f1_score(y_mix, pred_mix, zero_division=0), 6),
            'stage1_pct': round(trigger_blocked / len(X_mal) * 100, 2),
            'asr':        round(asr, 4),
            'false_positive_rate': round(fp_rate, 2),
        }


# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_dede(model_dir):
    with open(Path(model_dir) / 'training_config.json') as f:
        cfg = json.load(f)
    model = build_dede_model(
        input_dim=cfg['input_dim'], latent_dim=cfg.get('latent_dim', 64),
        encoder_hidden_dims=[256, 128], decoder_hidden_dims=[128, 256],
        mask_ratio=cfg.get('mask_ratio', 0.5), dropout=0.2,
        learning_rate=cfg.get('learning_rate', 0.001)
    )
    _ = model(tf.zeros((1, cfg['input_dim'])), training=False)
    model.load_weights(str(Path(model_dir) / 'best_model.weights.h5'))
    print(f'  ✓ DeDe loaded: input_dim={cfg["input_dim"]}, val_loss={cfg["best_val_loss"]:.4f}')
    return model


def load_or_train_stacking(X_tr, y_tr, label, save_dir):
    """Load from exp6 v3 cache if available, otherwise train."""
    # Thử load cache từ v3 trước (tránh train lại)
    v3_cache = Path('results/raw/exp6_hybrid_defense_v3') / f'ganopt_{label}'
    if (v3_cache / 'meta_model.pkl').exists():
        ens = create_stacking_ensemble_gan_optimized(input_dim=X_tr.shape[1])
        ens.meta_model = joblib.load(v3_cache / 'meta_model.pkl')
        for name in ['knn_5', 'knn_11']:
            p = v3_cache / f'{name}_model.pkl'
            if p.exists(): ens.base_models[name] = joblib.load(p)
        for name in ['mlp_deep', 'mlp_wide']:
            p = v3_cache / f'{name}_model.keras'
            if p.exists():
                from tensorflow import keras
                ens.base_models[name] = keras.models.load_model(p)
        ens.is_fitted = True
        print(f'  ✓ Reused v3 cache: {label}')
        return ens

    # Train mới nếu không có cache
    exp7_cache = Path(save_dir) / f'ganopt_{label}'
    if (exp7_cache / 'meta_model.pkl').exists():
        ens = create_stacking_ensemble_gan_optimized(input_dim=X_tr.shape[1])
        ens.meta_model = joblib.load(exp7_cache / 'meta_model.pkl')
        for name in ['knn_5', 'knn_11']:
            p = exp7_cache / f'{name}_model.pkl'
            if p.exists(): ens.base_models[name] = joblib.load(p)
        for name in ['mlp_deep', 'mlp_wide']:
            p = exp7_cache / f'{name}_model.keras'
            if p.exists():
                from tensorflow import keras
                ens.base_models[name] = keras.models.load_model(p)
        ens.is_fitted = True
        print(f'  ✓ Loaded exp7 cache: {label}')
        return ens

    print(f'\n  Training GAN-Opt Stacking [{label}] ({len(X_tr):,} × {X_tr.shape[1]})...')
    ens = create_stacking_ensemble_gan_optimized(input_dim=X_tr.shape[1])
    ens.fit(X_tr, y_tr, verbose=False)
    ens.save(exp7_cache)
    print(f'  ✓ Saved: {exp7_cache}')
    return ens


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',    default='datasets/splits/3.0_raw_from_latent')
    parser.add_argument('--dede-model',  default='experiments/dede_adapted/models_raw')
    parser.add_argument('--output-dir',  default='results/raw/exp7_combined_matrix')
    parser.add_argument('--trigger-rate', default='10',
                        help='Trigger rate để test (05|10|15), default=10')
    parser.add_argument('--threshold-pct', type=int, default=99)
    args = parser.parse_args()

    print('\n' + '='*80)
    print('EXPERIMENT 7: COMBINED ATTACK EVALUATION MATRIX'.center(80))
    print('Train: Clean / Poison_05 / 10 / 15 / 50'.center(80))
    print('Test:  Clean / GAN / Trigger (same sets for ALL models)'.center(80))
    print('='*80)

    out_dir  = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(args.data_dir)

    # ── Load DeDe (dùng chung cho tất cả kịch bản) ──────────────────────────
    print('\n[1] Loading DeDe-Adapted...')
    dede = load_dede(args.dede_model)

    # Tính threshold từ clean test set (cố định để so sánh công bằng)
    X_te_clean = np.load(base_dir / 'exp1_baseline/X_test.npy')
    y_te_clean = np.load(base_dir / 'exp1_baseline/y_test.npy')
    thr = np.percentile(dede.get_reconstruction_error(X_te_clean), args.threshold_pct)
    print(f'    DeDe threshold ({args.threshold_pct}th pct on clean): {thr:.6f}')

    # ── Load test sets (CỐ ĐỊNH, dùng cho TẤT CẢ models) ───────────────────
    print('\n[2] Loading FIXED test sets...')

    # Test set 1: Clean
    print(f'    Clean test:   {len(X_te_clean):,} samples')

    # Test set 2: GAN adversarial
    X_te_gan = np.load(base_dir / 'exp3_gan_attack/X_test.npy')
    y_te_gan = np.load(base_dir / 'exp3_gan_attack/y_test.npy')
    print(f'    GAN test:     {len(X_te_gan):,} samples')

    # Test set 3: Trigger backdoor (mixed realistic)
    trigger_dir = base_dir / f'exp5_trigger/trigger_{args.trigger_rate}'
    X_te_trigger_mix = np.load(trigger_dir / 'X_test_mixed_realistic.npy')
    print(f'    Trigger test: {len(X_te_trigger_mix):,} samples (rate={args.trigger_rate}%)')

    # ── Training scenarios ───────────────────────────────────────────────────
    train_scenarios = [
        ('clean',      base_dir / 'exp1_baseline'),
        ('poison_05',  base_dir / 'exp2_poisoning/poison_05'),
        ('poison_10',  base_dir / 'exp2_poisoning/poison_10'),
        ('poison_15',  base_dir / 'exp2_poisoning/poison_15'),
        ('poison_50',  base_dir / 'exp2_poisoning/poison_50'),
    ]

    all_results = []

    print('\n' + '='*80)
    print('[3] Running Full Evaluation Matrix...')
    print('='*80)

    for train_label, train_dir in train_scenarios:
        if not (train_dir / 'X_train.npy').exists():
            print(f'\n  ⚠️  Skip {train_label}: training data not found')
            continue

        X_tr = np.load(train_dir / 'X_train.npy')
        y_tr = np.load(train_dir / 'y_train.npy')
        n_poison = int((y_tr == 0).sum()) if 'poison' in train_label else 0

        print(f'\n  ► Train scenario: [{train_label}]  ({len(X_tr):,} samples)')
        if n_poison > 0:
            print(f'    Poisoned labels (malicious→benign): {n_poison:,}')

        # Train stacking (reuse cache if available)
        ens = load_or_train_stacking(X_tr, y_tr, train_label, out_dir)
        hds = HybridDefense(dede, ens, thr)

        # ── Evaluate trên cùng 3 test sets ──────────────────────────────────

        # Test 1: Clean
        m_clean = hds.evaluate_clean(X_te_clean, y_te_clean)
        row_clean = {'train_scenario': train_label, 'test_type': 'clean', **m_clean}
        all_results.append(row_clean)
        print(f'    [Test: Clean]   Acc={m_clean["accuracy"]:.4f}  F1={m_clean["f1_score"]:.4f}  Stage1={m_clean["stage1_pct"]:.1f}%')

        # Test 2: GAN adversarial
        m_gan = hds.evaluate_clean(X_te_gan, y_te_gan)  # same evaluation logic
        row_gan = {'train_scenario': train_label, 'test_type': 'gan_attack', **m_gan}
        all_results.append(row_gan)
        print(f'    [Test: GAN]     Acc={m_gan["accuracy"]:.4f}  F1={m_gan["f1_score"]:.4f}  Stage1={m_gan["stage1_pct"]:.1f}%')

        # Test 3: Trigger backdoor
        m_trig = hds.evaluate_trigger(trigger_dir, threshold=thr)
        row_trig = {'train_scenario': train_label, 'test_type': f'trigger_{args.trigger_rate}', **m_trig}
        all_results.append(row_trig)
        print(f'    [Test: Trigger] Acc={m_trig["accuracy"]:.4f}  F1={m_trig["f1_score"]:.4f}  ASR={m_trig["asr"]:.2f}%  FP={m_trig["false_positive_rate"]:.2f}%')

    # ── Save results ─────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    csv_path = out_dir / 'exp7_combined_matrix_results.csv'
    df.to_csv(csv_path, index=False)

    with open(out_dir / 'exp7_report.json', 'w') as f:
        json.dump({
            'config': {
                'dede_threshold': float(thr),
                'threshold_percentile': args.threshold_pct,
                'trigger_rate_tested': args.trigger_rate,
                'ensemble_type': 'GAN-Optimized (MLP_deep+MLP_wide+KNN_5+KNN_11)',
                'fixed_test_sets': ['clean', 'gan_attack', f'trigger_{args.trigger_rate}'],
            },
            'results': all_results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)

    # ── Print Matrix Summary ─────────────────────────────────────────────────
    print('\n' + '='*80)
    print('✅ EXP7 RESULT MATRIX'.center(80))
    print('='*80)
    header = "Train \\ Test"
    print(f'\n  {header:<16} {"Clean F1":>10} {"GAN F1":>10} {"Trigger F1":>12} {"ASR":>8}')
    print('  ' + '-'*60)

    for train_label, _ in train_scenarios:
        rows = {r['test_type']: r for r in all_results if r['train_scenario'] == train_label}
        if not rows:
            continue
        f1_clean = rows.get('clean', {}).get('f1_score', 'N/A')
        f1_gan   = rows.get('gan_attack', {}).get('f1_score', 'N/A')
        f1_trig  = rows.get(f'trigger_{args.trigger_rate}', {}).get('f1_score', 'N/A')
        asr      = rows.get(f'trigger_{args.trigger_rate}', {}).get('asr', 'N/A')

        f1_c = f'{f1_clean:.4f}' if isinstance(f1_clean, float) else f1_clean
        f1_g = f'{f1_gan:.4f}'   if isinstance(f1_gan,   float) else f1_gan
        f1_t = f'{f1_trig:.4f}'  if isinstance(f1_trig,  float) else f1_trig
        asr_ = f'{asr:.2f}%'     if isinstance(asr,      float) else asr
        print(f'  {train_label:<16} {f1_c:>10} {f1_g:>10} {f1_t:>12} {asr_:>8}')

    print(f'\n  📌 Đọc kết quả:')
    print(f'     • Hàng "clean"     = Hybrid Defense không bị tấn công training')
    print(f'     • Hàng "poison_XX" = Hybrid Defense BỊ đầu độc training data')
    print(f'     • Cột "GAN F1"     = Khả năng chịu đựng tấn công evasion lúc test')
    print(f'     • Ô (poison_50, GAN) = WORST CASE: cả 2 loại tấn công cùng lúc')
    print(f'\n📁 {csv_path}\n')


if __name__ == '__main__':
    main()
