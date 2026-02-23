"""
Experiment 5: Trigger Backdoor Attack — RAW Features

Threat Model:
  Training:
    - Attacker inject trigger (features[13,30,39]=1.0) vào N% malicious samples
    - Flip label triggered malicious → benign (backdoor)
    - Defender train trên poisoned+backdoored data (không biết)

  Inference:
    - Traffic bình thường (không có trigger): classify đúng
    - Traffic có trigger pattern: model classify sai → benign (backdoor active)

  Evaluation (ĐÚNG — mixed realistic test set):
    - X_test_malicious_triggered: malicious + trigger → đo ASR
    - X_test_benign_clean: benign không có trigger → đo FP rate
    - X_test_mixed_realistic: hỗn hợp → đo Overall Accuracy/F1

NOTE: Đây là exp individual models. Hybrid Defense (exp6) dùng DeDe để chặn trigger.

Data: datasets/splits/3.0_raw_from_latent/exp5_trigger/
"""
import sys, json, numpy as np, pandas as pd, subprocess
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))


def evaluate_on_mixed(pred_fn, tdir):
    """Evaluate model trên mixed realistic test set."""
    X_mix = np.load(tdir / 'X_test_mixed_realistic.npy')
    y_mix = np.load(tdir / 'y_test_mixed_realistic.npy')
    X_mal = np.load(tdir / 'X_test_malicious_triggered.npy')
    X_ben = np.load(tdir / 'X_test_benign_clean.npy')

    # ASR = % triggered malicious classified as benign (attack success)
    pred_mal = pred_fn(X_mal)
    asr = (pred_mal == 0).mean() * 100

    # FP = % benign classified as malicious
    pred_ben = pred_fn(X_ben)
    fp = (pred_ben == 1).mean() * 100

    # Overall
    pred_mix = pred_fn(X_mix)
    acc = accuracy_score(y_mix, pred_mix)
    f1  = f1_score(y_mix, pred_mix, zero_division=0)

    return {
        'accuracy': round(acc, 4),
        'f1_score': round(f1, 4),
        'asr':      round(asr, 2),
        'fp_rate':  round(fp, 2),
    }


def main():
    print("\n🚀 EXP5 RAW: TRIGGER BACKDOOR ATTACK — Individual Models")
    print("   Test set: mixed realistic (benign_clean + malicious_triggered)")
    print("   Metric: ASR (Attack Success Rate) — thấp hơn = tốt hơn")

    base    = BASE_DIR / "datasets/splits/3.0_raw_from_latent"
    out_dir = BASE_DIR / "results/raw_fair/exp5_trigger"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for rate in ['05', '10', '15']:
        tdir = base / f'exp5_trigger/trigger_{rate}'
        if not tdir.exists():
            print(f"⚠️  trigger_{rate}: data not found"); continue

        # Load trigger metadata
        with open(tdir / 'trigger_metadata.json') as f:
            meta = json.load(f)
        print(f"\n{'='*70}")
        print(f"  Trigger {rate}% — Trigger indices: {meta['trigger_indices']}, value={meta['trigger_value']}")

        # Load poisoned train data
        X_tr = np.load(tdir / 'X_train.npy'); y_tr = np.load(tdir / 'y_train.npy')
        y_cl = np.load(base / 'exp1_baseline/y_train.npy')
        flips = (y_tr != y_cl).sum()
        print(f"  Train: {len(X_tr):,} samples, {flips:,} labels flipped (backdoored)")
        print(f"  Test:  mixed_realistic = benign_clean + malicious_triggered")

        # Train MLP (simplest model to show backdoor effect)
        try:
            from models.advanced.mlp import create_mlp_model
            mlp = create_mlp_model(input_dim=X_tr.shape[1])
            mlp.fit(X_tr, y_tr, epochs=30, batch_size=256, verbose=0,
                    validation_split=0.1)

            def pred_fn(X):
                p = mlp.predict(X, verbose=0)
                return (p.flatten() > 0.5).astype(int)

            m = evaluate_on_mixed(pred_fn, tdir)
            m['attack_type'] = f'trigger_{rate}'
            m['model'] = 'MLP'
            all_results.append(m)
            print(f"  MLP: Acc={m['accuracy']:.4f}  F1={m['f1_score']:.4f}  ASR={m['asr']:.1f}%  FP={m['fp_rate']:.1f}%")

        except Exception as e:
            print(f"  ❌ MLP error: {e}")

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(out_dir / 'trigger_results.csv', index=False)
        print(f"\n✅ Exp5 RAW hoàn thành!")
        print(f"📁 Results: {out_dir}/")
        print("\nNote: Hybrid Defense (exp6) chặn 100% trigger qua DeDe Stage 1 → ASR=0%")


if __name__ == '__main__':
    main()
