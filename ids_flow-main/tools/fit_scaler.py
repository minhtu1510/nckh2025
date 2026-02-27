#!/usr/bin/env python3
import argparse
import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="datasets/v2/binary or datasets/v2/family")
    ap.add_argument("--out", required=True, help="artifacts/v2/.../scaler.pkl")
    args = ap.parse_args()

    x_path = os.path.join(args.data, "X_train.npy")
    X = np.load(x_path, mmap_mode="r")
    sc = StandardScaler()
    sc.fit(X)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(sc, args.out)
    print(f"[OK] Saved scaler to {args.out}")
    print("n_features_in_ =", sc.n_features_in_)

if __name__ == "__main__":
    main()
