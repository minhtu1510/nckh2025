import argparse, time
import pandas as pd
import requests
from ids.runtime.predictor_v2 import IDSRuntimeV2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--nrows", type=int, default=50000)
    ap.add_argument("--backend", default="http://127.0.0.1:5000")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--tau-low", type=float, default=0.2)
    ap.add_argument("--tau-high", type=float, default=0.95)
    ap.add_argument("--sample-from", type=int, default=500000, help="read first N rows then sample nrows")
    args = ap.parse_args()

    ids = IDSRuntimeV2(tau_low=args.tau_low, tau_high=args.tau_high)

    # read a big chunk then sample (avoid 'first rows all benign')
    n_read = max(args.sample_from, args.nrows)
    df_all = pd.read_csv(args.csv, nrows=n_read)
    df = df_all.sample(n=min(args.nrows, len(df_all)), random_state=0).reset_index(drop=True)

    sent = 0
    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        gt_attack = row.pop("Attack", None)
        gt_label  = row.pop("Label", None)

        out = ids.predict_flow(row)
        out["gt_attack"] = gt_attack
        out["gt_label"] = gt_label
        out["ts"] = time.time()

        requests.post(args.backend + "/api/event", json=out, timeout=10)
        sent += 1
        if args.sleep > 0:
            time.sleep(args.sleep)

    print("[DONE] sent", sent, "events")

if __name__ == "__main__":
    main()
