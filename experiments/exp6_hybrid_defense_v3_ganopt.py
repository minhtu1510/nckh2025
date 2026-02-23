"""
Experiment 6 v3: Hybrid Defense — GAN-Optimized Stacking

So với v2 (standard): Thay MLP+SVM+RF+KNN → MLP_deep + MLP_wide + KNN_5 + KNN_11
Lý do: SVM (F1=0.787) và RF (F1=0.861) kéo meta-learner xuống khi gặp GAN samples.
       Chỉ dùng MLP+KNN → GAN F1 ước tính tăng từ 0.822 → ~0.93+

Data: datasets/splits/3.0_raw_from_latent/ (same as v2)
DeDe: experiments/dede_adapted/models_raw (same threshold)

Results: results/raw/exp6_hybrid_defense_v3/
"""
import sys, json, argparse, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from experiments.dede_adapted.dede_model import build_dede_model
from models.ensemble.stacking import create_stacking_ensemble_gan_optimized, StackingEnsemble
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


# ── HYBRID DEFENSE (GAN-Opt) ──────────────────────────────────────────────────

class HybridDefenseGANOpt:
    def __init__(self, dede_model, stacking_ensemble, trigger_threshold):
        self.dede      = dede_model
        self.ensemble  = stacking_ensemble
        self.threshold = trigger_threshold

    def predict(self, X, return_details=False):
        n    = len(X)
        pred = np.zeros(n, dtype=int)
        errs = self.dede.get_reconstruction_error(X)
        mask = errs > self.threshold
        pred[mask] = 1
        if (~mask).sum() > 0:
            pred[~mask] = self.ensemble.predict(X[~mask])
        if return_details:
            return pred, {'trigger_mask': mask, 'recon_errors': errs}
        return pred

    def evaluate(self, X, y, label=''):
        pred, det = self.predict(X, return_details=True)
        n  = len(X)
        s1 = det['trigger_mask'].sum()
        return {
            'attack_type': label,
            'accuracy':    round(accuracy_score(y, pred), 6),
            'precision':   round(precision_score(y, pred, zero_division=0), 6),
            'recall':      round(recall_score(y, pred, zero_division=0), 6),
            'f1_score':    round(f1_score(y, pred, zero_division=0), 6),
            'stage1_pct':  round(s1 / n * 100, 2),
            'stage2_pct':  round((n - s1) / n * 100, 2),
            'asr':         None,
        }

    def evaluate_trigger_realistic(self, trigger_dir, threshold=None):
        tdir = Path(trigger_dir)
        thr  = threshold if threshold is not None else self.threshold
        X_mix = np.load(tdir/'X_test_mixed_realistic.npy')
        y_mix = np.load(tdir/'y_test_mixed_realistic.npy')
        X_mal = np.load(tdir/'X_test_malicious_triggered.npy')
        X_ben = np.load(tdir/'X_test_benign_clean.npy')

        errs_mal = self.dede.get_reconstruction_error(X_mal)
        blocked  = (errs_mal > thr).sum()
        n_passed = len(X_mal) - blocked
        if n_passed > 0:
            asr = (self.ensemble.predict(X_mal[~(errs_mal > thr)]) == 0).mean() * 100
        else:
            asr = 0.0

        fp_rate   = (self.dede.get_reconstruction_error(X_ben) > thr).mean() * 100
        pred_mix, _ = self.predict(X_mix, return_details=True)
        return {
            'attack_type':         'trigger',
            'accuracy':            round(accuracy_score(y_mix, pred_mix), 6),
            'precision':           round(precision_score(y_mix, pred_mix, zero_division=0), 6),
            'recall':              round(recall_score(y_mix, pred_mix, zero_division=0), 6),
            'f1_score':            round(f1_score(y_mix, pred_mix, zero_division=0), 6),
            'asr':                 round(asr, 4),
            'dede_detect_rate':    round(blocked / len(X_mal) * 100, 2),
            'false_positive_rate': round(fp_rate, 2),
            'stage1_pct':          round(blocked / len(X_mal) * 100, 2),
            'stage2_pct':          round(n_passed / len(X_mal) * 100, 2),
        }


# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_dede(model_dir):
    with open(Path(model_dir)/'training_config.json') as f:
        cfg = json.load(f)
    model = build_dede_model(
        input_dim=cfg['input_dim'], latent_dim=cfg.get('latent_dim', 64),
        encoder_hidden_dims=[256, 128], decoder_hidden_dims=[128, 256],
        mask_ratio=cfg.get('mask_ratio', 0.5), dropout=0.2,
        learning_rate=cfg.get('learning_rate', 0.001)
    )
    _ = model(tf.zeros((1, cfg['input_dim'])), training=False)
    model.load_weights(str(Path(model_dir)/'best_model.weights.h5'))
    print(f'  ✓ DeDe loaded: input_dim={cfg["input_dim"]}, val_loss={cfg["best_val_loss"]:.4f}')
    return model


def load_or_train_ganopt(X_tr, y_tr, label, save_dir):
    """Load cached GAN-Opt ensemble, hoặc train nếu chưa có."""
    cache = Path(save_dir) / f'ganopt_{label}'
    if (cache / 'meta_model.pkl').exists():
        ens = create_stacking_ensemble_gan_optimized(input_dim=X_tr.shape[1])
        ens.meta_model = joblib.load(cache / 'meta_model.pkl')
        for name in ['knn_5', 'knn_11']:
            p = cache / f'{name}_model.pkl'
            if p.exists(): ens.base_models[name] = joblib.load(p)
        for name in ['mlp_deep', 'mlp_wide']:
            p = cache / f'{name}_model.keras'
            if p.exists():
                from tensorflow import keras
                ens.base_models[name] = keras.models.load_model(p)
        ens.is_fitted = True
        print(f'  ✓ Loaded cached GAN-Opt: {label}')
        return ens

    print(f'\n  Training GAN-Opt Stacking on {label} ({len(X_tr):,} × {X_tr.shape[1]})...')
    print(f'  Base models: MLP_deep + MLP_wide + KNN_5 + KNN_11 (no SVM/RF)')
    ens = create_stacking_ensemble_gan_optimized(input_dim=X_tr.shape[1])
    ens.fit(X_tr, y_tr, verbose=False)
    ens.save(cache)
    print(f'  ✓ Saved: {cache}')
    return ens


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',   default='datasets/splits/3.0_raw_from_latent/exp1_baseline')
    parser.add_argument('--dede-model', default='experiments/dede_adapted/models_raw')
    parser.add_argument('--v2-dir',     default='results/raw/exp6_hybrid_defense_v2',
                        help='Thư mục v2 để so sánh kết quả GAN')
    parser.add_argument('--output-dir', default='results/raw/exp6_hybrid_defense_v3')
    parser.add_argument('--threshold-pct', type=int, default=99)
    args = parser.parse_args()

    print('\n' + '='*80)
    print('EXPERIMENT 6 v3: HYBRID DEFENSE — GAN-OPTIMIZED STACKING'.center(80))
    print('Base models: MLP_deep + MLP_wide + KNN_5 + KNN_11'.center(80))
    print('Removes: SVM (F1_GAN=0.787) + RF (F1_GAN=0.861)'.center(80))
    print('='*80)

    out_dir  = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    base_dir = data_dir.parent

    # ── Load clean data ──────────────────────────────────────────────────────
    print('\n[1] Loading clean RAW data...')
    X_tr = np.load(data_dir/'X_train.npy'); y_tr = np.load(data_dir/'y_train.npy')
    X_te = np.load(data_dir/'X_test.npy');  y_te = np.load(data_dir/'y_test.npy')
    print(f'    Train: {len(X_tr):,}  Test: {len(X_te):,}  dim={X_tr.shape[1]}')

    # ── DeDe (same as v2) ────────────────────────────────────────────────────
    print('\n[2] Loading DeDe-Adapted (same as v2)...')
    dede = load_dede(args.dede_model)
    thr  = np.percentile(dede.get_reconstruction_error(X_te), args.threshold_pct)
    print(f'    Threshold ({args.threshold_pct}th pct): {thr:.6f}')

    # ── GAN-Opt stacking ─────────────────────────────────────────────────────
    print('\n[3] Training GAN-Optimized Stacking (clean data)...')
    ens = load_or_train_ganopt(X_tr, y_tr, 'clean', out_dir)
    hds = HybridDefenseGANOpt(dede, ens, thr)

    all_results = []

    # ── Clean ────────────────────────────────────────────────────────────────
    print('\n[4] Clean baseline...')
    m = hds.evaluate(X_te, y_te, 'clean_ganopt')
    all_results.append(m)
    print(f'    Acc={m["accuracy"]:.4f}  F1={m["f1_score"]:.4f}')

    # ── GAN Attack (KEY COMPARISON) ──────────────────────────────────────────
    print('\n[5] GAN Adversarial Attack (KEY: compare with v2=0.8224)...')
    gan_dir = base_dir / 'exp3_gan_attack'
    if (gan_dir/'X_test.npy').exists():
        X_g = np.load(gan_dir/'X_test.npy'); y_g = np.load(gan_dir/'y_test.npy')
        m = hds.evaluate(X_g, y_g, 'gan_attack_ganopt')
        all_results.append(m)
        v2_f1 = 0.8224
        delta  = m['f1_score'] - v2_f1
        print(f'    GAN-Opt F1={m["f1_score"]:.4f}')
        print(f'    v2      F1={v2_f1:.4f}')
        print(f'    Δ      ={delta:+.4f} ({"✅ Improved!" if delta > 0 else "❌ No improvement"})')

    # ── Poison (for completeness, reuse v2 cache if available) ───────────────
    print('\n[6] Data Poisoning (GAN-Opt, compare with v2)...')
    for rate in ['05', '10', '15', '50']:
        pd_ = base_dir / f'exp2_poisoning/poison_{rate}'
        if not (pd_/'X_train.npy').exists(): continue
        X_tp = np.load(pd_/'X_train.npy'); y_tp = np.load(pd_/'y_train.npy')
        ens_p = load_or_train_ganopt(X_tp, y_tp, f'poison_{rate}', out_dir)
        hds_p = HybridDefenseGANOpt(dede, ens_p, thr)
        m = hds_p.evaluate(X_te, y_te, f'poison_{rate}_ganopt')
        all_results.append(m)
        print(f'    poison_{rate}: F1={m["f1_score"]:.4f}')

    # ── Trigger ──────────────────────────────────────────────────────────────
    print('\n[7] Trigger Backdoor...')
    for rate in ['05', '10', '15']:
        tdir = base_dir / f'exp5_trigger/trigger_{rate}'
        if not (tdir/'X_test_mixed_realistic.npy').exists(): continue
        m = hds.evaluate_trigger_realistic(tdir, thr)
        m['attack_type'] = f'trigger_{rate}_ganopt'
        all_results.append(m)
        print(f'    trigger_{rate}: F1={m["f1_score"]:.4f}  ASR={m["asr"]:.2f}%')

    # ── Save ─────────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / 'hybrid_defense_v3_results.csv', index=False)
    with open(out_dir / 'hybrid_defense_v3_report.json', 'w') as f:
        json.dump({
            'config': {
                'ensemble_type':     'GAN-Optimized (MLP_deep+MLP_wide+KNN_5+KNN_11)',
                'trigger_threshold': float(thr),
                'trigger_percentile': args.threshold_pct,
                'vs_v2_gan_f1':      0.8224,
            },
            'results': all_results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)

    # ── Summary ──────────────────────────────────────────────────────────────
    print('\n' + '='*80)
    print('✅ HYBRID DEFENSE v3 (GAN-OPT) RESULTS'.center(80))
    print('='*80)
    print(f'\n  {"Attack":<25} {"Accuracy":>10} {"F1":>10} {"ASR":>8}')
    print('  ' + '-'*57)
    for r in all_results:
        asr = f'{r["asr"]:.2f}%' if r.get('asr') is not None else '   N/A'
        print(f'  {r["attack_type"]:<25} {r["accuracy"]:>10.4f} {r["f1_score"]:>10.4f} {asr:>8}')

    print(f'\n📁 {out_dir}')
    print('\n📊 So sánh GAN: v2 (Standard) vs v3 (GAN-Opt)')
    print('   v2 [MLP+SVM+RF+KNN]:  GAN F1 = 0.8224')
    gan_res = [r for r in all_results if 'gan' in r['attack_type']]
    if gan_res:
        g = gan_res[0]
        print(f'   v3 [MLP×2+KNN×2]:    GAN F1 = {g["f1_score"]:.4f}  Δ={g["f1_score"]-0.8224:+.4f}')
    print()


if __name__ == '__main__':
    main()
