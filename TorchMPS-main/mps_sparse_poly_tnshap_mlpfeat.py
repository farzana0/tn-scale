#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
from torchmps import MPS
from sklearn.metrics import r2_score
import time

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D = 50
M_NODES = 50
N_MASKED = 100
TRAIN_FRAC = 0.95

BOND_DIM = 20
FEATURE_DIM = 3
HIDDEN_FM = 32

LR = 1e-3
EPOCHS = 20
BATCH = 512

SAVE_PATH = "mps_sparse_poly.pt"

# ---------------- Feature Map ----------------

class MLPFeatureMap(nn.Module):
    def __init__(self, feature_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim)
        )

    def forward(self, x):
        if x.dim() == 0:
            x = x.view(1, 1)
        elif x.dim() == 1:
            x = x.view(-1, 1)
        return self.net(x)

# ---------------- Sparse Polynomial Teacher ----------------

def sparse_poly_f(x, S, a, b):
    xS = x[:, S]
    lin = (xS * a.unsqueeze(0)).sum(dim=1)
    quad = (xS.pow(2) * b.unsqueeze(0)).sum(dim=1)
    return lin + quad

# ---------------- Build masked Chebyshev dataset ----------------

def build_masked_chebyshev(X, S, a, b, m):
    nodes = torch.cos(
        (2 * torch.arange(m, device=DEVICE) + 1)
        / (2 * m) * np.pi
    )
    nodes = (nodes + 1) / 2

    N = X.shape[0]
    out = []

    for i in range(N):
        base = X[i].repeat(m, 1)
        base[:, S] = nodes.view(-1, 1)
        ones = torch.ones(m, 1, device=DEVICE)
        out.append(torch.cat([base, ones], dim=1))

    X_ext = torch.cat(out, dim=0)
    y = sparse_poly_f(X_ext[:, :-1], S, a, b)

    return X_ext, y

# ---------------- Main ----------------

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    print("Generating sparse poly teacher...")

    S = torch.randperm(D)[:5]
    a = torch.randn(5, device=DEVICE)
    b = torch.randn(5, device=DEVICE)

    X_targets = torch.rand(N_MASKED, D, device=DEVICE)

    print("Building masked Chebyshev dataset (fast GPU)...")
    X_ext, y = build_masked_chebyshev(X_targets, S, a, b, M_NODES)

    # Train/test split
    N = X_ext.shape[0]
    idx = torch.randperm(N, device=DEVICE)
    train_N = int(TRAIN_FRAC * N)

    tr_idx = idx[:train_N]
    te_idx = idx[train_N:]

    Xtr, ytr = X_ext[tr_idx], y[tr_idx]
    Xte, yte = X_ext[te_idx], y[te_idx]

    print(f"Train size: {Xtr.shape[0]}, Test size: {Xte.shape[0]}")

    # --------- Build MPS + Feature Map (CPU → move once) ---------

    print("Building MPS + MLP feature map...")

    mps = MPS(
        input_dim=D + 1,
        output_dim=1,
        bond_dim=BOND_DIM,
        feature_dim=FEATURE_DIM,
        periodic_bc=False,
        adaptive_mode=False
    )

    fmap = MLPFeatureMap(feature_dim=FEATURE_DIM, hidden=HIDDEN_FM)
    mps.register_feature_map(fmap)
    mps = mps.to(DEVICE)

    opt = torch.optim.Adam(mps.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # --------- Training (FAST GPU BATCHED) ---------

    print("Training...")

    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(Xtr.shape[0], device=DEVICE)
        tot_loss = 0.0
        seen = 0

        for i in range(0, Xtr.shape[0], BATCH):
            idx = perm[i:i+BATCH]
            xb = Xtr[idx]
            yb = ytr[idx]

            preds = mps(xb).squeeze(-1)
            loss = loss_fn(preds, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tot_loss += loss.item() * xb.size(0)
            seen += xb.size(0)

        # Eval
        with torch.no_grad():
            tr_preds = mps(Xtr).squeeze(-1)
            te_preds = mps(Xte).squeeze(-1)

            tr_mse = ((tr_preds - ytr) ** 2).mean().item()
            te_mse = ((te_preds - yte) ** 2).mean().item()

            tr_r2 = r2_score(ytr.cpu().numpy(), tr_preds.cpu().numpy())
            te_r2 = r2_score(yte.cpu().numpy(), te_preds.cpu().numpy())

        print(
            f"Epoch {ep:02d} | "
            f"Train MSE {tr_mse:.5f} R2 {tr_r2:.4f} | "
            f"Test MSE {te_mse:.5f} R2 {te_r2:.4f}"
        )

    # --------- Save trained MPS ---------

    torch.save(mps.state_dict(), SAVE_PATH)
    print(f"Saved model to: {SAVE_PATH}")

if __name__ == "__main__":
    main()
