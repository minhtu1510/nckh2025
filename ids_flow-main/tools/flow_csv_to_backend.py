# tools/flow_csv_to_backend.py
import argparse, time, requests, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--nrows", type=int, default=50000)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--endpoint", default="/api/ingest_bulk")
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, nrows=args.nrows)

    # ---- DEBUG: xem cột có gì ----
    print("cols=", df.columns.tolist()[:50])
    GT_LABEL_COL = "Label" if "Label" in df.columns else None
    GT_ATTACK_COL = "Attack" if "Attack" in df.columns else None
    print("gt_attack_col=", GT_ATTACK_COL, "gt_label_col=", GT_LABEL_COL)
    print("gt_attack_col=", GT_ATTACK_COL, "gt_label_col=", GT_LABEL_COL)

    # tách gt nếu có
    gt_label = df[GT_LABEL_COL].astype(str).tolist() if GT_LABEL_COL else None
    gt_attack = df[GT_ATTACK_COL].astype(str).tolist() if GT_ATTACK_COL else None

    drop_cols = [c for c in ["Label", "Attack"] if c in df.columns]
    flow_df = df.drop(columns=drop_cols)

    url = args.backend.rstrip("/") + args.endpoint

    items = []
    sent = 0
    for i in range(len(flow_df)):
        flow = flow_df.iloc[i].to_dict()
        it = {"flow": flow}

        if gt_label is not None:
            it["gt_label"] = gt_label[i]
        if gt_attack is not None:
            it["gt_attack"] = gt_attack[i]

        # ---- DEBUG: in sample vài dòng đầu ----
        if i < 3:
            print("sample gt_attack/gt_label=", it.get("gt_attack"), it.get("gt_label"))
            # nếu muốn xem thêm 1 vài feature:
            # print("sample flow keys=", list(flow.keys())[:15])

        items.append(it)

        if len(items) >= args.batch:
            r = requests.post(url, json={"items": items, "ts0": time.time()}, timeout=args.timeout)
            r.raise_for_status()
            sent += len(items)
            print(f"[POST] {sent}/{args.nrows} ok")
            items = []

    if items:
        r = requests.post(url, json={"items": items, "ts0": time.time()}, timeout=args.timeout)
        r.raise_for_status()
        sent += len(items)
        print(f"[POST] {sent}/{args.nrows} ok")

if __name__ == "__main__":
    main()