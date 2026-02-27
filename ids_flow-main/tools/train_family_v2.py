#!/usr/bin/env python3
import argparse, os
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score

class MLPFamily(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, h1=256, h2=128, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, num_classes)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.act(self.fc2(x)))
        return self.out(x)  # logits

@torch.no_grad()
def eval_metrics(model, loader, device):
    model.eval()
    ys, yh = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        ys.append(yb.numpy())
        yh.append(pred)
    y = np.concatenate(ys)
    p = np.concatenate(yh)
    acc = accuracy_score(y, p)
    f1m = f1_score(y, p, average="macro")
    return acc, f1m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="datasets/v2/family")
    ap.add_argument("--scaler", required=True, help="artifacts/v2/family/scaler.pkl")
    ap.add_argument("--out", required=True, help="artifacts/v2/family/model.pt")
    ap.add_argument("--label-encoder", default=None, help="datasets/v2/family/label_encoder.pkl (auto if missing)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    Xtr = np.load(os.path.join(args.data, "X_train.npy"), mmap_mode="r")
    ytr = np.load(os.path.join(args.data, "y_train.npy"), mmap_mode="r").astype(np.int64)
    Xte = np.load(os.path.join(args.data, "X_test.npy"), mmap_mode="r")
    yte = np.load(os.path.join(args.data, "y_test.npy"), mmap_mode="r").astype(np.int64)

    sc = joblib.load(args.scaler)
    Xtr = sc.transform(Xtr).astype(np.float32, copy=False)
    Xte = sc.transform(Xte).astype(np.float32, copy=False)

    n_classes = int(np.max(ytr) + 1)

    device = torch.device(args.device)
    model = MLPFamily(in_dim=Xtr.shape[1], num_classes=n_classes).to(device)

    ds_tr = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    ds_te = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # class weights for imbalance
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    w = (counts.sum() / (counts + 1e-12))
    w = w / w.mean()
    class_w = torch.tensor(w, dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=class_w)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_f1 = -1.0
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

        loss_avg = running / len(ds_tr)
        acc, f1m = eval_metrics(model, dl_te, device)
        print(f"[E{ep}/{args.epochs}] loss={loss_avg:.4f}  acc={acc:.4f}  macroF1={f1m:.4f}")

        if f1m > best_f1:
            best_f1 = f1m
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "in_dim": int(Xtr.shape[1]),
                "num_classes": n_classes,
                "scaler_path": args.scaler,
                "label_encoder_path": args.label_encoder or os.path.join(args.data, "label_encoder.pkl"),
                "note": "family MLP (attack-only). Run only when p_attack > tau_high."
            }, args.out)
            print(f"[CKPT] saved best -> {args.out}")

if __name__ == "__main__":
    main()
