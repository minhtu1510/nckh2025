import argparse, numpy as np, pandas as pd
from ids.runtime.predictor_v2 import IDSRuntimeV2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--nrows", type=int, default=200000)
    ap.add_argument("--tau-low", type=float, default=0.2)
    ap.add_argument("--taus", type=str, default="0.90,0.92,0.94,0.95,0.96,0.97,0.98,0.99")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--read-mult", type=int, default=5)  # đọc nhiều hơn rồi sample
    args = ap.parse_args()

    taus = [float(x) for x in args.taus.split(",")]

    df_all = pd.read_csv(args.csv, nrows=max(args.nrows*args.read_mult, 500000))
    df = df_all.sample(n=args.nrows, random_state=args.seed).reset_index(drop=True)

    gt = df["Label"].astype(int).to_numpy()
    flows = df.drop(columns=[c for c in ["Label","Attack"] if c in df.columns])

    print(f"[SWEEP] n={len(df)}  gt_pos={int(gt.sum())}")

    # load runtime 1 lần
    ids = IDSRuntimeV2(tau_low=args.tau_low, tau_high=0.95)

    # compute p_attack 1 lần
    p = np.zeros(len(df), dtype=np.float32)
    for i in range(len(df)):
        out = ids.predict_flow(flows.iloc[i].to_dict())
        # nếu stage=rule(high) thì coi là attack chắc (p=1)
        if out.get("stage") == "rule":
            p[i] = 1.0
        else:
            p[i] = float(out.get("p_attack") or 0.0)

    # sweep ngưỡng cực nhanh
    for tau_high in taus:
        pred_attack = (p > tau_high).astype(np.int32)
        TP = int(((pred_attack==1) & (gt==1)).sum())
        FP = int(((pred_attack==1) & (gt==0)).sum())
        FN = int(((pred_attack==0) & (gt==1)).sum())
        TN = int(((pred_attack==0) & (gt==0)).sum())
        tpr = TP / max(1, (TP+FN))
        fpr = FP / max(1, (FP+TN))
        print(f"tau_high={tau_high:.2f}  TPR={tpr:.4f}  FPR={fpr:.4f}  TP={TP} FP={FP} FN={FN} TN={TN}")

if __name__ == "__main__":
    main()
