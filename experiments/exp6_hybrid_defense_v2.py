"""
Experiment 6 v2: Hybrid Defense System — RAW Features

Architecture (2-stage):
  Stage 1: DeDe-Adapted (Masked Autoencoder) — Trigger backdoor detection
  Stage 2: Stacking Ensemble (MLP+SVM+RF+KNN) — Classification

Data: datasets/splits/3.0_raw_from_latent/ (50 RAW features)
DeDe: experiments/dede_adapted/models_raw (val_loss=0.0023)
Threshold: 99th percentile (FP≈1%)

Results: results/raw/exp6_hybrid_defense_v2/
  F1: clean=0.9787, poison50=0.9672, gan=0.8224, trigger=0.9810 (ASR=0%)
"""
import sys, json, argparse, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from experiments.dede_adapted.dede_model import build_dede_model
from models.ensemble.stacking import create_stacking_ensemble, StackingEnsemble
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


# ── HYBRID DEFENSE ─────────────────────────────────────────────────────────────

class HybridDefenseSystem:
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
        """Evaluate using mixed realistic test set (correct threat model)."""
        tdir = Path(trigger_dir)
        thr  = threshold if threshold is not None else self.threshold

        X_mix = np.load(tdir / 'X_test_mixed_realistic.npy')
        y_mix = np.load(tdir / 'y_test_mixed_realistic.npy')
        X_mal = np.load(tdir / 'X_test_malicious_triggered.npy')
        X_ben = np.load(tdir / 'X_test_benign_clean.npy')

        # ASR: % triggered malicious that bypass Stage 1
        errs_mal = self.dede.get_reconstruction_error(X_mal)
        trigger_blocked = (errs_mal > thr).sum()
        n_passed = len(X_mal) - trigger_blocked
        if n_passed > 0:
            pred_passed = self.ensemble.predict(X_mal[~(errs_mal > thr)])
            asr = (pred_passed == 0).mean() * 100
        else:
            asr = 0.0

        # FP: % clean benign blocked by Stage 1
        errs_ben = self.dede.get_reconstruction_error(X_ben)
        fp_rate  = (errs_ben > thr).mean() * 100

        # Overall on mixed set
        pred_mix, _ = self.predict(X_mix, return_details=True)
        return {
            'attack_type':          'trigger',
            'accuracy':             round(accuracy_score(y_mix, pred_mix), 6),
            'precision':            round(precision_score(y_mix, pred_mix, zero_division=0), 6),
            'recall':               round(recall_score(y_mix, pred_mix, zero_division=0), 6),
            'f1_score':             round(f1_score(y_mix, pred_mix, zero_division=0), 6),
            'asr':                  round(asr, 4),
            'dede_detect_rate':     round(trigger_blocked / len(X_mal) * 100, 2),
            'false_positive_rate':  round(fp_rate, 2),
            'stage1_pct':           round(trigger_blocked / len(X_mal) * 100, 2),
            'stage2_pct':           round(n_passed / len(X_mal) * 100, 2),
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
    print(f'  ✓ DeDe loaded: input_dim={cfg["input_dim"]}, best_val_loss={cfg["best_val_loss"]:.4f}')
    return model


def load_or_train_stacking(X_tr, y_tr, label, save_dir):
    cache = Path(save_dir) / f'stacking_{label}'
    if (cache / 'meta_model.pkl').exists():
        ens = create_stacking_ensemble(input_dim=X_tr.shape[1])
        ens.meta_model = joblib.load(cache / 'meta_model.pkl')
        for name in ['svm', 'rf', 'knn']:
            p = cache / f'{name}_model.pkl'
            if p.exists(): ens.base_models[name] = joblib.load(p)
        mlp_p = cache / 'mlp_model.keras'
        if mlp_p.exists():
            from tensorflow import keras
            ens.base_models['mlp'] = keras.models.load_model(mlp_p)
        ens.is_fitted = True
        print(f'  ✓ Loaded cached stacking: {label}')
        return ens

    print(f'\n  Training stacking on {label} ({len(X_tr):,} × {X_tr.shape[1]})...')
    ens = create_stacking_ensemble(input_dim=X_tr.shape[1])
    ens.fit(X_tr, y_tr, verbose=False)
    ens.save(cache)
    print(f'  ✓ Saved: {cache}')
    return ens


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',   default='datasets/splits/3.0_raw_from_latent/exp1_baseline')
    parser.add_argument('--dede-model', default='experiments/dede_adapted/models_raw')
    parser.add_argument('--output-dir', default='results/raw/exp6_hybrid_defense_v2')
    parser.add_argument('--threshold-pct', type=int, default=99)
    args = parser.parse_args()

    print('\n' + '='*80)
    print('EXPERIMENT 6 v2: HYBRID DEFENSE SYSTEM — RAW FEATURES'.center(80))
    print('Stage1: DeDe-Adapted  |  Stage2: Stacking Ensemble'.center(80))
    print('='*80)

    out_dir  = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    base_dir = data_dir.parent  # 3.0_raw_from_latent/

    # ── Load clean data ──────────────────────────────────────────────────────
    print('\n[1] Loading clean RAW data...')
    X_tr = np.load(data_dir/'X_train.npy'); y_tr = np.load(data_dir/'y_train.npy')
    X_te = np.load(data_dir/'X_test.npy');  y_te = np.load(data_dir/'y_test.npy')
    print(f'    Train: {len(X_tr):,}  Test: {len(X_te):,}  dim={X_tr.shape[1]}')

    # ── DeDe ────────────────────────────────────────────────────────────────
    print('\n[2] Loading DeDe-Adapted (RAW)...')
    dede = load_dede(args.dede_model)
    clean_errs = dede.get_reconstruction_error(X_te)
    thr = np.percentile(clean_errs, args.threshold_pct)
    fp  = (clean_errs > thr).mean() * 100
    print(f'    Threshold ({args.threshold_pct}th pct): {thr:.6f}  FP={fp:.2f}%')

    # ── Clean stacking ───────────────────────────────────────────────────────
    print('\n[3] Stacking Ensemble (clean)...')
    ens_clean = load_or_train_stacking(X_tr, y_tr, 'clean', out_dir)
    hds = HybridDefenseSystem(dede, ens_clean, thr)

    all_results = []

    # ── Clean baseline ───────────────────────────────────────────────────────
    print('\n[4] Evaluating CLEAN...')
    m = hds.evaluate(X_te, y_te, 'clean')
    all_results.append(m)
    print(f'    Acc={m["accuracy"]:.4f}  F1={m["f1_score"]:.4f}')

    # ── Data Poisoning ───────────────────────────────────────────────────────
    print('\n[5] Data Poisoning...')
    for rate in ['05', '10', '15', '50']:
        pd_ = base_dir / f'exp2_poisoning/poison_{rate}'
        if not (pd_/'X_train.npy').exists(): continue
        X_tp = np.load(pd_/'X_train.npy'); y_tp = np.load(pd_/'y_train.npy')
        ens_p = load_or_train_stacking(X_tp, y_tp, f'poison_{rate}', out_dir)
        hds_p = HybridDefenseSystem(dede, ens_p, thr)
        m = hds_p.evaluate(X_te, y_te, f'poison_{rate}')
        all_results.append(m)
        print(f'    poison_{rate}: Acc={m["accuracy"]:.4f}  F1={m["f1_score"]:.4f}')

    # ── GAN Attack ───────────────────────────────────────────────────────────
    print('\n[6] GAN Adversarial Attack...')
    gan_dir = base_dir / 'exp3_gan_attack'
    if (gan_dir/'X_test.npy').exists():
        X_g = np.load(gan_dir/'X_test.npy'); y_g = np.load(gan_dir/'y_test.npy')
        m = hds.evaluate(X_g, y_g, 'gan_attack')
        all_results.append(m)
        print(f'    GAN: Acc={m["accuracy"]:.4f}  F1={m["f1_score"]:.4f}')

    # ── Trigger Backdoor ─────────────────────────────────────────────────────
    print('\n[7] Trigger Backdoor (mixed realistic test set)...')
    for rate in ['05', '10', '15']:
        tdir = base_dir / f'exp5_trigger/trigger_{rate}'
        if not (tdir/'X_test_mixed_realistic.npy').exists(): continue
        m = hds.evaluate_trigger_realistic(tdir, threshold=thr)
        m['attack_type'] = f'trigger_{rate}'
        all_results.append(m)
        print(f'    trigger_{rate}: F1={m["f1_score"]:.4f}  ASR={m["asr"]:.2f}%  FP={m["false_positive_rate"]:.2f}%')

    # ── Save ─────────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / 'hybrid_defense_v2_results.csv', index=False)
    with open(out_dir / 'hybrid_defense_v2_report.json', 'w') as f:
        json.dump({
            'config': {
                'trigger_threshold': float(thr),
                'trigger_percentile': args.threshold_pct,
                'dede_model': args.dede_model,
                'data_dir': args.data_dir,
            },
            'results': all_results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)

    # ── Summary ──────────────────────────────────────────────────────────────
    print('\n' + '='*80)
    print('✅ HYBRID DEFENSE v2 RESULTS'.center(80))
    print('='*80)
    print(f'\n  {"Attack":<18} {"Accuracy":>10} {"F1":>10} {"ASR":>8} {"Stage1%":>8}')
    print('  ' + '-'*54)
    for r in all_results:
        asr = f'{r["asr"]:.2f}%' if r.get('asr') is not None else '   N/A'
        print(f'  {r["attack_type"]:<18} {r["accuracy"]:>10.4f} {r["f1_score"]:>10.4f} {asr:>8} {r.get("stage1_pct",0):>7.1f}%')
    print(f'\n📁 {out_dir}\n')


if __name__ == '__main__':
    main()
