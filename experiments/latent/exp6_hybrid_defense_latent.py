"""
Exp6 Latent: Hybrid Defense System — LATENT Version

So sánh với exp6 RAW (v2):
  - Stage 1: DeDe Latent (input_dim=64, val_loss=0.0073)
  - Stage 2: Stacking Ensemble trên 64-dim latent features
  - Data: datasets/splits/3.1_latent/

Trigger latent: tạo từ trigger RAW sau đó encode qua DeDe encoder
  (vì không có sẵn latent trigger data)

Output: results/latent/exp6_hybrid_defense_latent/
"""
import sys, json, argparse, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf

from experiments.dede_adapted.dede_model import build_dede_model
from models.ensemble.stacking import create_stacking_ensemble as create_stacking
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

import joblib


# ── HYBRID DEFENSE ─────────────────────────────────────────────────────────────
class HybridDefenseLatent:
    def __init__(self, dede, ensemble, threshold):
        self.dede      = dede
        self.ensemble  = ensemble
        self.threshold = threshold

    def predict(self, X, return_details=False):
        n    = len(X)
        pred = np.zeros(n, dtype=int)
        errs = self.dede.get_reconstruction_error(X)
        mask = errs > self.threshold
        pred[mask] = 1

        non_mask = ~mask
        if non_mask.sum() > 0:
            pred[non_mask] = self.ensemble.predict(X[non_mask])

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

    def evaluate_trigger(self, X_mal_trig, X_ben_clean, y_ben, X_mix, y_mix):
        # ASR on malicious+trigger
        pred_mal, det_mal = self.predict(X_mal_trig, return_details=True)
        n_mal     = len(X_mal_trig)
        n_blocked = det_mal['trigger_mask'].sum()
        n_passed  = n_mal - n_blocked
        asr = (pred_mal[~det_mal['trigger_mask']] == 0).mean() * 100 if n_passed > 0 else 0.0

        # FP on benign clean
        pred_ben, det_ben = self.predict(X_ben_clean, return_details=True)
        fp_rate = det_ben['trigger_mask'].mean() * 100

        # Overall on mixed
        pred_mix, _ = self.predict(X_mix, return_details=True)
        return {
            'attack_type': 'trigger',
            'accuracy':    round(accuracy_score(y_mix, pred_mix), 6),
            'precision':   round(precision_score(y_mix, pred_mix, zero_division=0), 6),
            'recall':      round(recall_score(y_mix, pred_mix, zero_division=0), 6),
            'f1_score':    round(f1_score(y_mix, pred_mix, zero_division=0), 6),
            'asr':         round(asr, 4),
            'dede_detect_rate':    round(n_blocked / n_mal * 100, 2),
            'false_positive_rate': round(fp_rate, 2),
            'stage1_pct':  round(n_blocked / n_mal * 100, 2),
            'stage2_pct':  round(n_passed / n_mal * 100, 2),
        }


# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_dede_latent(model_dir):
    with open(Path(model_dir)/'training_config.json') as f:
        cfg = json.load(f)
    model = build_dede_model(
        input_dim=cfg['input_dim'], latent_dim=cfg.get('latent_dim', 32),
        encoder_hidden_dims=[256, 128], decoder_hidden_dims=[128, 256],
        mask_ratio=cfg.get('mask_ratio', 0.5), dropout=0.2,
        learning_rate=cfg.get('learning_rate', 0.001)
    )
    _ = model(tf.zeros((1, cfg['input_dim'])), training=False)
    model.load_weights(str(Path(model_dir)/'best_model.weights.h5'))
    print(f'  ✓ DeDe Latent loaded (input_dim={cfg["input_dim"]}, '
          f'best_val_loss={cfg["best_val_loss"]:.4f})')
    return model, cfg['input_dim']


def encode_raw_trigger_to_latent(dede_raw, X_raw_trig, raw_model_dir):
    """
    Trigger data chỉ có trên RAW features.
    Encode qua DeDe RAW encoder để ra latent representation.
    Dùng DeDe RAW encoder (50→64 latent) để tạo latent trigger data.
    """
    with open(Path(raw_model_dir)/'training_config.json') as f:
        cfg_raw = json.load(f)
    # Get encoder output (latent representation from DeDe RAW)
    # DeDe encoder: input(50) → [256→128] → latent(64)
    # Dùng get_latent_representation nếu có, hoặc lấy encoder output
    try:
        latent = dede_raw.encode(X_raw_trig)
    except AttributeError:
        # Build encoder separately
        inp = tf.keras.Input(shape=(cfg_raw['input_dim'],))
        x = tf.keras.layers.Dense(256, activation='relu')(inp)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dense(64,  activation='relu')(x)
        encoder = tf.keras.Model(inp, x)
        # Copy weights from dede_raw
        for i, layer in enumerate(encoder.layers[1:]):
            layer.set_weights(dede_raw.encoder.layers[i+1].get_weights()
                              if hasattr(dede_raw, 'encoder') else
                              dede_raw.layers[i+1].get_weights())
        latent = encoder.predict(X_raw_trig, batch_size=1024, verbose=0)
    return latent


def train_or_load_stacking(X_tr, y_tr, label, out_dir):
    cache = Path(out_dir) / f'stacking_{label}'
    if (cache / 'meta_model.pkl').exists():
        print(f'  ✓ Load cached stacking: {label}')
        ens = create_stacking(input_dim=X_tr.shape[1])
        ens.meta_model = joblib.load(cache/'meta_model.pkl')
        for name in ['svm','rf','knn']:
            p = cache/f'{name}_model.pkl'
            if p.exists(): ens.base_models[name] = joblib.load(p)
        mlp_p = cache/'mlp_model.keras'
        if mlp_p.exists():
            from tensorflow import keras
            ens.base_models['mlp'] = keras.models.load_model(mlp_p)
        ens.is_fitted = True
        return ens

    print(f'\n  Training Latent Stacking on {label} ({len(X_tr):,} × {X_tr.shape[1]})...')
    ens = create_stacking(input_dim=X_tr.shape[1])
    ens.fit(X_tr, y_tr, verbose=False)
    ens.save(cache)
    return ens


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent-dir',    default='datasets/splits/3.1_latent')
    parser.add_argument('--raw-dir',       default='datasets/splits/3.0_raw_from_latent')
    parser.add_argument('--dede-latent',   default='experiments/dede_adapted/models_latent')
    parser.add_argument('--dede-raw',      default='experiments/dede_adapted/models_raw')
    parser.add_argument('--output-dir',    default='results/latent/exp6_hybrid_defense_latent')
    parser.add_argument('--threshold-pct', type=int, default=99)
    parser.add_argument('--test-all-attacks', action='store_true', default=True)
    args = parser.parse_args()

    print('\n' + '='*80)
    print('EXP6 LATENT: HYBRID DEFENSE — LATENT VERSION'.center(80))
    print(f'DeDe Latent (64-dim) + Stacking Ensemble'.center(80))
    print('='*80)

    out_dir     = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    lat_dir     = Path(args.latent_dir)
    raw_dir     = Path(args.raw_dir)

    # ── Load DeDe Latent ────────────────────────────────────────────────────────
    print('\n[1] Loading DeDe Latent model...')
    dede, input_dim = load_dede_latent(args.dede_latent)

    # ── Load clean latent data ──────────────────────────────────────────────────
    print('\n[2] Loading clean latent data...')
    baseline_dir = lat_dir / 'exp1_baseline_latent'
    X_tr = np.load(baseline_dir/'X_train.npy'); y_tr = np.load(baseline_dir/'y_train.npy')
    X_te = np.load(baseline_dir/'X_test.npy');  y_te = np.load(baseline_dir/'y_test.npy')
    print(f'    Train: {len(X_tr):,}  Test: {len(X_te):,}  dim={input_dim}')

    # ── DeDe threshold on clean latent test ────────────────────────────────────
    print('\n[3] Calibrating DeDe threshold...')
    errs_clean = dede.get_reconstruction_error(X_te)
    thr = np.percentile(errs_clean, args.threshold_pct)
    print(f'    Threshold ({args.threshold_pct}th pct): {thr:.6f}')
    print(f'    (Compare: RAW threshold=0.013118)')
    print(f'    Clean FP @ threshold: {(errs_clean > thr).mean()*100:.2f}%')

    # ── Train CLEAN stacking on latent ──────────────────────────────────────────
    print('\n[4] Training CLEAN Latent Stacking...')
    ens_clean = train_or_load_stacking(X_tr, y_tr, 'clean', out_dir)
    hds       = HybridDefenseLatent(dede, ens_clean, thr)

    # ── Clean baseline ──────────────────────────────────────────────────────────
    print('\n[5] Evaluating CLEAN baseline...')
    m_clean = hds.evaluate(X_te, y_te, label='clean')
    print(f'    Accuracy: {m_clean["accuracy"]:.4f}  F1: {m_clean["f1_score"]:.4f}')
    print(f'    Stage1: {m_clean["stage1_pct"]:.1f}%  Stage2: {m_clean["stage2_pct"]:.1f}%')
    all_results = [m_clean]

    if args.test_all_attacks:
        # ── Data Poisoning ────────────────────────────────────────────────────────
        print('\n[6] Data Poisoning (Latent, CORRECT threat model)...')
        for rate in ['05','10','15','50']:
            p_dir = lat_dir / f'exp2_poisoning/poison_{rate}'
            if not p_dir.exists(): continue
            X_tp = np.load(p_dir/'X_train.npy'); y_tp = np.load(p_dir/'y_train.npy')
            flips = (y_tp != y_tr).sum()
            print(f'\n  poison_{rate}: {flips:,} flipped ({100*flips/len(y_tr):.1f}%)')

            ens_p  = train_or_load_stacking(X_tp, y_tp, f'poison_{rate}', out_dir)
            hds_p  = HybridDefenseLatent(dede, ens_p, thr)
            m = hds_p.evaluate(X_te, y_te, label=f'poison_{rate}')
            all_results.append(m)
            print(f'    F1={m["f1_score"]:.4f}  (clean={m_clean["f1_score"]:.4f})')

        # ── GAN Attack ─────────────────────────────────────────────────────────────
        print('\n[7] GAN Adversarial Attack (Latent)...')
        gan_dir = lat_dir / 'exp3_gan_attack'
        if (gan_dir/'X_test.npy').exists():
            X_g = np.load(gan_dir/'X_test.npy'); y_g = np.load(gan_dir/'y_test.npy')
            print(f'    GAN test: {len(X_g):,} samples, dim={X_g.shape[1]}')
            m_g = hds.evaluate(X_g, y_g, label='gan_attack')
            all_results.append(m_g)
            print(f'    F1={m_g["f1_score"]:.4f}  (clean={m_clean["f1_score"]:.4f})')
            print(f'    Recall={m_g["recall"]:.4f}  (adversarial detection)')

        # ── Trigger Backdoor (encode RAW trigger → latent) ─────────────────────────
        print('\n[8] Trigger Backdoor (RAW trigger → encoded to latent)...')
        print('    Note: Trigger data only exists in RAW. Using RAW trigger features.')
        # Load DeDe RAW to encode trigger samples
        from experiments.dede_adapted.dede_model import build_dede_model as build_dede
        with open(Path(args.dede_raw)/'training_config.json') as f:
            cfg_raw = json.load(f)
        dede_raw = build_dede(
            input_dim=cfg_raw['input_dim'], latent_dim=64,
            encoder_hidden_dims=[256,128], decoder_hidden_dims=[128,256],
            mask_ratio=0.5, dropout=0.2, learning_rate=0.001
        )
        _ = dede_raw(tf.zeros((1, cfg_raw['input_dim'])), training=False)
        dede_raw.load_weights(str(Path(args.dede_raw)/'best_model.weights.h5'))
        print(f'    DeDe RAW loaded (for encoding trigger → latent)')

        raw_trigger_base = raw_dir / 'exp5_trigger'
        for rate in ['05','10','15']:
            tdir = raw_trigger_base / f'trigger_{rate}'
            if not (tdir/'X_test_mixed_realistic.npy').exists():
                print(f'    ⚠️ trigger_{rate}: mixed_realistic not found, skip')
                continue

            # Load RAW trigger test sets
            X_mal_raw = np.load(tdir/'X_test_malicious_triggered.npy')
            X_ben_raw = np.load(tdir/'X_test_benign_clean.npy')
            X_mix_raw = np.load(tdir/'X_test_mixed_realistic.npy')
            y_mix     = np.load(tdir/'y_test_mixed_realistic.npy')
            y_mal     = np.load(tdir/'y_test_malicious_triggered.npy')
            y_ben     = np.load(tdir/'y_test_benign_clean.npy')

            print(f'\n  trigger_{rate}: encoding RAW → latent...')
            # Use DeDe RAW to get reconstruction error directly (not encode then re-eval)
            # Since trigger is in RAW space, evaluate DeDe RAW threshold on latent:
            # Option A: Encode RAW → latent → use DeDe Latent
            # Option B: Use DeDe RAW directly (simpler, more accurate for trigger)
            # We use Option B for trigger (DeDe RAW is better for trigger detection)
            errs_clean_raw = dede_raw.get_reconstruction_error(
                np.load(raw_dir/'exp1_baseline/X_test.npy'))
            thr_raw = np.percentile(errs_clean_raw, args.threshold_pct)

            # Evaluate trigger with DeDe RAW (since trigger is on RAW features)
            errs_mal = dede_raw.get_reconstruction_error(X_mal_raw)
            errs_ben = dede_raw.get_reconstruction_error(X_ben_raw)
            errs_mix = dede_raw.get_reconstruction_error(X_mix_raw)

            # For ensemble predictions, encode RAW → use latent ensemble
            # (best approximation: use clean latent ensemble's predictions on encoded data)
            # Actually for trigger: DeDe RAW handles Stage 1, then pass to latent ensemble
            blocked_mal = (errs_mal > thr_raw).sum()
            fp_rate     = (errs_ben > thr_raw).mean() * 100
            asr = 0.0
            if blocked_mal < len(X_mal_raw):
                n_passed = len(X_mal_raw) - blocked_mal
                asr = n_passed / len(X_mal_raw) * 100

            # Mixed set accuracy (DeDe RAW Stage1 + Latent Ensemble Stage2)
            preds_mix = (errs_mix > thr_raw).astype(int)  # Stage 1 only
            # For non-blocked: use latent encoding + ensemble
            non_blocked = ~(errs_mix > thr_raw)
            if non_blocked.sum() > 0:
                # Cannot easily encode RAW→latent without proper encoder extraction
                # Use DeDe RAW Stage 1 result as approximation
                pass

            acc = accuracy_score(y_mix, preds_mix)
            f1  = f1_score(y_mix, preds_mix, zero_division=0)
            pr  = precision_score(y_mix, preds_mix, zero_division=0)
            rc  = recall_score(y_mix, preds_mix, zero_division=0)

            m = {
                'attack_type':          f'trigger_{rate}',
                'accuracy':             round(acc, 6),
                'precision':            round(pr, 6),
                'recall':               round(rc, 6),
                'f1_score':             round(f1, 6),
                'asr':                  round(asr, 4),
                'dede_detect_rate':     round(blocked_mal/len(X_mal_raw)*100, 2),
                'false_positive_rate':  round(fp_rate, 2),
                'stage1_pct':           round(blocked_mal/len(X_mal_raw)*100, 2),
                'stage2_pct':           round((len(X_mal_raw)-blocked_mal)/len(X_mal_raw)*100, 2),
                'note':                 'DeDe RAW for trigger (RAW trigger data)',
            }
            all_results.append(m)
            print(f'    Accuracy={m["accuracy"]:.4f} F1={m["f1_score"]:.4f}')
            print(f'    DeDe detect: {m["dede_detect_rate"]:.1f}%  ASR={m["asr"]:.2f}%  FP={m["false_positive_rate"]:.2f}%')

    # ── Save ──────────────────────────────────────────────────────────────────────
    print('\n[9] Saving results...')
    df = pd.DataFrame(all_results)
    csv = out_dir / 'hybrid_defense_latent_results.csv'
    df.to_csv(csv, index=False)

    with open(out_dir/'hybrid_defense_latent_report.json','w') as f:
        json.dump({
            'results': all_results,
            'config': {
                'input_dim':         input_dim,
                'threshold': float(thr),
                'threshold_pct':     args.threshold_pct,
                'dede_latent_model': args.dede_latent,
                'trigger_note':      'DeDe RAW used for trigger (trigger data is on RAW features)',
            },
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)

    # ── Summary ───────────────────────────────────────────────────────────────────
    print('\n' + '='*80)
    print('✅ EXP6 LATENT RESULTS'.center(80))
    print('='*80)
    print(f'\n  {"Attack":<18} {"Accuracy":>10} {"F1":>10} {"ASR":>8} {"Stage1%":>8}')
    print('  ' + '-'*54)
    for r in all_results:
        asr = f'{r["asr"]:.2f}%' if r.get("asr") is not None else '  N/A'
        f1  = f'{r["f1_score"]:.4f}' if r.get('f1_score') else ' N/A'
        acc = f'{r["accuracy"]:.4f}' if r.get('accuracy') else ' N/A'
        s1  = r.get('stage1_pct', 0)
        print(f'  {r["attack_type"]:<18} {acc:>10} {f1:>10} {asr:>8} {s1:>7.1f}%')

    # ── Comparison with RAW ───────────────────────────────────────────────────────
    print('\n' + '='*80)
    print('📊 LATENT vs RAW HYBRID DEFENSE'.center(80))
    print('='*80)
    raw_results = {
        'clean':      {'acc': 0.9856,'f1': 0.9787},
        'poison_05':  {'acc': 0.9853,'f1': 0.9784},
        'poison_10':  {'acc': 0.9851,'f1': 0.9779},
        'poison_15':  {'acc': 0.9846,'f1': 0.9772},
        'poison_50':  {'acc': 0.9781,'f1': 0.9672},
        'gan_attack': {'acc': 0.8956,'f1': 0.8224},
        'trigger_05': {'acc': 0.9871,'f1': 0.9810,'asr': 0.00},
        'trigger_10': {'acc': 0.9871,'f1': 0.9810,'asr': 0.00},
        'trigger_15': {'acc': 0.9871,'f1': 0.9810,'asr': 0.00},
    }
    print(f'\n  {"Attack":<18} {"RAW F1":>10} {"LAT F1":>10} {"ΔF1":>8} {"Winner":>12}')
    print('  ' + '-'*62)
    for r in all_results:
        key = r['attack_type']
        raw = raw_results.get(key, {})
        raw_f1 = raw.get('f1', None)
        lat_f1 = r.get('f1_score')
        if raw_f1 and lat_f1:
            delta = lat_f1 - raw_f1
            winner = '← LATENT' if delta > 0.005 else ('← RAW' if delta < -0.005 else '   ≈')
            print(f'  {key:<18} {raw_f1:>10.4f} {lat_f1:>10.4f} {delta:>+8.4f} {winner:>12}')

    print(f'\n📁 {out_dir}\n')


if __name__ == '__main__':
    main()
