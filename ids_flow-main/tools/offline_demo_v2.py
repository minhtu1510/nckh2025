#!/usr/bin/env python3
import argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

class MLPBinary(nn.Module):
    def __init__(self, in_dim, h1=256, h2=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

class MLPFamily(nn.Module):
    def __init__(self, in_dim, num_classes, h1=256, h2=128, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, num_classes)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.act(self.fc2(x)))
        return self.out(x)

def clean_X(df, feats):
    X = df[feats].copy()
    for c in feats:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X.to_numpy(np.float32, copy=False)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--nrows", type=int, default=50000)
    ap.add_argument("--tau-low", type=float, default=0.2)
    ap.add_argument("--tau-high", type=float, default=0.95)

    ap.add_argument("--binary-schema", default="schemas/v2/binary.json")
    ap.add_argument("--binary-scaler", default="artifacts/v2/binary/scaler.pkl")
    ap.add_argument("--binary-model", default="artifacts/v2/binary/model.pt")

    ap.add_argument("--family-schema", default="schemas/v2/family.json")
    ap.add_argument("--family-scaler", default="artifacts/v2/family/scaler.pkl")
    ap.add_argument("--family-model", default="artifacts/v2/family/model.pt")
    ap.add_argument("--label-encoder", default="datasets/v2/family/label_encoder.pkl")

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    bin_feats = json.load(open(args.binary_schema))["feature_names"]
    fam_feats = json.load(open(args.family_schema))["feature_names"]
    needed = set(bin_feats + fam_feats + ["Label", "Attack"])

    # df = pd.read_csv(args.csv, nrows=args.nrows, usecols=lambda c: c in needed)
    df_all = pd.read_csv(args.csv, nrows=max(args.nrows*5, 500000), usecols=lambda c: c in needed)
    df = df_all.sample(n=args.nrows, random_state=0).reset_index(drop=True)

    Xb = clean_X(df, bin_feats)
    Xf = clean_X(df, fam_feats)

    scb = joblib.load(args.binary_scaler)
    scf = joblib.load(args.family_scaler)
    Xb = scb.transform(Xb).astype(np.float32, copy=False)
    Xf = scf.transform(Xf).astype(np.float32, copy=False)

    ck_b = torch.load(args.binary_model, map_location="cpu")
    mb = MLPBinary(in_dim=ck_b["in_dim"]).to(device)
    mb.load_state_dict(ck_b["model_state_dict"])
    mb.eval()

    le = joblib.load(args.label_encoder)
    ck_f = torch.load(args.family_model, map_location="cpu")
    mf = MLPFamily(in_dim=ck_f["in_dim"], num_classes=ck_f["num_classes"]).to(device)
    mf.load_state_dict(ck_f["model_state_dict"])
    mf.eval()

    xb = torch.from_numpy(Xb).to(device)
    logits = mb(xb)
    p_attack = torch.sigmoid(logits).cpu().numpy()

    low = (p_attack < args.tau_low).sum()
    mid = ((p_attack >= args.tau_low) & (p_attack <= args.tau_high)).sum()
    high = (p_attack > args.tau_high).sum()

    print(f"[BINARY] n={len(p_attack)}  benign(low)={low}  suspicious(mid)={mid}  attack(high)={high}")
    print(f"[BINARY] p_attack stats: min={p_attack.min():.4f} mean={p_attack.mean():.4f} max={p_attack.max():.4f}")

    idx = np.where(p_attack > args.tau_high)[0]
    if len(idx) == 0:
        print("[FAMILY] no samples passed tau_high")
        return

    xf = torch.from_numpy(Xf[idx]).to(device)
    flogits = mf(xf)
    probs = torch.softmax(flogits, dim=1).cpu().numpy()
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    names = le.inverse_transform(pred)

    import collections
    cnt = collections.Counter(names.tolist())
    print("[FAMILY] top families:", cnt.most_common(10))
    print("[FAMILY] example 10:")
    for i in range(min(10, len(idx))):
        r = int(idx[i])
        raw_attack = df.loc[r, "Attack"] if "Attack" in df.columns else "NA"
        print(f"  row={r}  p_attack={p_attack[r]:.3f}  family={names[i]}  conf={conf[i]:.3f}  raw_Attack={raw_attack}")

if __name__ == "__main__":
    main()
