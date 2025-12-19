#!/usr/bin/env python
"""
So sánh MLP trên 3 loại đặc trưng:
1) Raw
2) Latent từ Autoencoder
3) Latent z_mean từ VAE

Đánh giá thêm:
- Clean accuracy
- FGSM attack
- PGD attack
- GAN attack (AdvGAN)
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchattacks as ta

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Wrapper2Class(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        logits = self.base(x)            # (N,)
        logits = logits.unsqueeze(1)     # (N,1)
        two = torch.cat([-logits, logits], dim=1)  # (N,2)
        return two

# ============================================================
# Utils
# ============================================================
def print_metrics(name, y_true, y_pred):
    print(f"\n== {name} ==")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))


@torch.no_grad()
def eval_clean(model, X, y, name="Clean"):
    model.eval()
    logits = model(torch.from_numpy(X).float().to(DEVICE))
    y_pred = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)
    print_metrics(name, y, y_pred)

def to_two_class_logits(logits):
    logits = logits.unsqueeze(1)  # (N,) -> (N,1)
    return torch.cat([-logits, logits], dim=1)  # (N,2)



def attack_and_eval(model, X, y, eps=0.05, steps=10, attack="fgsm"):
    model.eval()

    wrapped = Wrapper2Class(model).to(DEVICE)

    x_t = torch.from_numpy(X).float().to(DEVICE)
    y_t = torch.from_numpy(y).long().to(DEVICE)

    if attack == "fgsm":
        atk = ta.FGSM(wrapped, eps=eps)
    else:
        atk = ta.PGD(wrapped, eps=eps, alpha=eps/4, steps=steps)

    x_adv = atk(x_t, y_t)

    with torch.no_grad():
        logits = wrapped(x_adv)
        y_pred = logits.argmax(dim=1).cpu().numpy()

    print_metrics(f"Adversarial ({attack}, eps={eps})", y, y_pred)

# ============================================================
# MLP
# ============================================================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden=(128, 64)):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


# ============================================================
# AutoEncoder
# ============================================================
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z


# ============================================================
# VAE
# ============================================================
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc_dec(z))
        return torch.sigmoid(self.fc_out(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar


def bce_kld_loss(recon, x, mu, logvar):
    bce = nn.functional.mse_loss(recon, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld


# ============================================================
# AdvGAN (Generator & Discriminator)
# ============================================================
class GANGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class GANDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train_gan(X_train, mlp, epochs=5, batch_size=128):
    input_dim = X_train.shape[1]

    G = GANGenerator(input_dim).to(DEVICE)
    D = GANDiscriminator(input_dim).to(DEVICE)

    opt_G = optim.Adam(G.parameters(), lr=1e-3)
    opt_D = optim.Adam(D.parameters(), lr=1e-3)

    loss_fn = nn.BCELoss()

    ds = TensorDataset(torch.from_numpy(X_train).float())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    real_label = torch.ones(batch_size, 1).to(DEVICE)
    fake_label = torch.zeros(batch_size, 1).to(DEVICE)

    mlp.eval()

    for epoch in range(epochs):
        for (xb,) in dl:
            xb = xb.to(DEVICE)

            # Train D
            opt_D.zero_grad()
            out_real = D(xb)
            loss_real = loss_fn(out_real, real_label[:xb.size(0)])

            noise = G(xb)
            fake_data = xb + noise
            out_fake = D(fake_data.detach())
            loss_fake = loss_fn(out_fake, fake_label[:xb.size(0)])

            (loss_real + loss_fake).backward()
            opt_D.step()

            # Train G
            opt_G.zero_grad()
            noise = G(xb)
            adv_x = xb + noise

            logits = mlp(adv_x)
            target_wrong = (torch.sigmoid(logits) < 0.5).float()

            loss_attack = nn.BCELoss()(torch.sigmoid(logits), target_wrong)
            loss_attack.backward()
            opt_G.step()

    return G


@torch.no_grad()
def gan_attack_and_eval(model, G, X, y):
    x_t = torch.from_numpy(X).float().to(DEVICE)
    x_adv = x_t + G(x_t)

    logits = model(x_adv)
    y_pred = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)

    print_metrics("Adversarial (GAN)", y, y_pred)


# ============================================================
# Train MLP
# ============================================================
def train_mlp(X_train, y_train, X_test, y_test, epochs=30, batch_size=128, name="MLP"):
    model = MLP(X_train.shape[1]).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    ds = TensorDataset(torch.from_numpy(X_train).float(),
                       torch.from_numpy(y_train).float())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    # Evaluate
    eval_clean(model, X_test, y_test, name=f"{name} (Clean)")
    attack_and_eval(model, X_test, y_test, eps=0.05, attack="fgsm")
    attack_and_eval(model, X_test, y_test, eps=0.05, steps=10, attack="pgd")

    print("\n[Training GAN to generate adversarial attacks...]")
    G = train_gan(X_train, model, epochs=5, batch_size=batch_size)

    gan_attack_and_eval(model, G, X_test, y_test)


# ============================================================
# AE / VAE Pipeline
# ============================================================
def run_ae(X_train, y_train, X_test, y_test, epochs=30, batch_size=128, latent_dim=64):
    ae = AutoEncoder(X_train.shape[1], latent_dim).to(DEVICE)
    opt = optim.Adam(ae.parameters(), lr=1e-3)
    ds = TensorDataset(torch.from_numpy(X_train).float())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    ae.train()
    for _ in range(epochs):
        for (xb,) in dl:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            recon, _ = ae(xb)
            loss = nn.functional.mse_loss(recon, xb)
            loss.backward()
            opt.step()

    with torch.no_grad():
        Xtr_lat = ae.encoder(torch.from_numpy(X_train).float().to(DEVICE)).cpu().numpy()
        Xte_lat = ae.encoder(torch.from_numpy(X_test).float().to(DEVICE)).cpu().numpy()

    train_mlp(Xtr_lat, y_train, Xte_lat, y_test, name="AE latent -> MLP")


def run_vae(X_train, y_train, X_test, y_test, epochs=30, batch_size=128, latent_dim=32, hidden_dim=128):
    vae = VAE(X_train.shape[1], latent_dim, hidden_dim).to(DEVICE)
    opt = optim.Adam(vae.parameters(), lr=1e-3)
    ds = TensorDataset(torch.from_numpy(X_train).float())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    vae.train()
    for _ in range(epochs):
        for (xb,) in dl:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            recon, mu, logvar = vae(xb)
            loss = bce_kld_loss(recon, xb, mu, logvar)
            loss.backward()
            opt.step()

    with torch.no_grad():
        mu_train, _ = vae.encode(torch.from_numpy(X_train).float().to(DEVICE))
        mu_test, _ = vae.encode(torch.from_numpy(X_test).float().to(DEVICE))
        Xtr_lat = mu_train.cpu().numpy()
        Xte_lat = mu_test.cpu().numpy()

    train_mlp(Xtr_lat, y_train, Xte_lat, y_test, name="VAE z_mean -> MLP")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", required=True, help="Path CSV đã xử lý")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--mode", choices=["all", "mlp", "ae", "vae"], default="all")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    df = pd.read_csv(args.processed)

    if args.label_column not in df.columns:
        raise ValueError(f"Không thấy cột nhãn '{args.label_column}'. Cột hiện có: {list(df.columns)}")

    y = df[args.label_column].astype(int).values

    leak_cols = ["Attack", "Src IP", "Dst IP", "Flow ID", "Label"]
    df = df.drop(columns=[c for c in leak_cols if c in df.columns], errors="ignore")

    X = df.drop(columns=[args.label_column]).values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Using device:", DEVICE)

    if args.mode in ("mlp", "all"):
        train_mlp(X_train, y_train, X_test, y_test,
                  epochs=args.epochs, batch_size=args.batch_size)

    if args.mode in ("ae", "all"):
        run_ae(X_train, y_train, X_test, y_test,
               epochs=args.epochs, batch_size=args.batch_size)

    if args.mode in ("vae", "all"):
        run_vae(X_train, y_train, X_test, y_test,
                epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
