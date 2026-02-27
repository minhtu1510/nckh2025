#!/usr/bin/env python3
import argparse, os
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

class MLPBinary(nn.Module):
    def __init__(self, in_dim: int, h1=256, h2=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)  # logits

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    device = next(model.parameters()).device
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device) 
        logits = model(xb)
        prob = torch.sigmoid(logits)
        ys.append(yb.cpu().numpy())
        ps.append(prob.cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    ap  = average_precision_score(y, p) if len(np.unique(y)) == 2 else float("nan")

    yhat = (p >= 0.5).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn + 1e-12)
    tpr = tp / (tp + fn + 1e-12)
    return {"AUC": auc, "AP": ap, "FPR": fpr, "TPR": tpr, "cm": cm}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="datasets/v2/binary")
    ap.add_argument("--scaler", required=True, help="artifacts/v2/binary/scaler.pkl")
    ap.add_argument("--out", required=True, help="artifacts/v2/binary/model.pt")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    Xtr = np.load(os.path.join(args.data, "X_train.npy"), mmap_mode="r")
    ytr = np.load(os.path.join(args.data, "y_train.npy"), mmap_mode="r").astype(np.float32)
    Xte = np.load(os.path.join(args.data, "X_test.npy"), mmap_mode="r")
    yte = np.load(os.path.join(args.data, "y_test.npy"), mmap_mode="r").astype(np.float32)

    sc = joblib.load(args.scaler)
    Xtr = sc.transform(Xtr).astype(np.float32, copy=False)
    Xte = sc.transform(Xte).astype(np.float32, copy=False)

    device = torch.device(args.device)
    model = MLPBinary(in_dim=Xtr.shape[1]).to(device)

    ds_tr = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    ds_te = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # class imbalance: pos_weight = (#neg/#pos)
    pos = float(ytr.sum())
    neg = float(len(ytr) - ytr.sum())
    pos_weight = torch.tensor([neg / (pos + 1e-12)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_ap = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)

        metrics = evaluate(model, dl_te)
        avg_loss = running / len(ds_tr)
        print(f"[E{ep}/{args.epochs}] loss={avg_loss:.4f}  AP={metrics['AP']:.4f}  AUC={metrics['AUC']:.4f}  FPR={metrics['FPR']:.6f}  TPR={metrics['TPR']:.4f}")

        if metrics["AP"] > best_ap:
            best_ap = metrics["AP"]
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "in_dim": int(Xtr.shape[1]),
                "scaler_path": args.scaler,
                "note": "binary MLP, output sigmoid(logits)=p_attack"
            }, args.out)
            print(f"[CKPT] saved best -> {args.out}")

    print("Confusion matrix (0/1):")
    print(metrics["cm"])

if __name__ == "__main__":
    main()
