#!/usr/bin/env python3
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchmps import MPS  # same import as your MNIST script

# -----------------------
# Config
# -----------------------
torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D = 50                 # number of features
N_TRAIN = 10_000
N_TEST = 2_000
BATCH_SIZE = 256
NUM_EPOCHS = 50

BOND_DIM = 20
LEARN_RATE = 1e-3
L2_REG = 0.0
NOISE_STD = 0.1   # label noise


# -----------------------
# Sparse polynomial teacher
# -----------------------

def make_sparse_poly_coeffs(d: int, k_active: int = 5, seed: int = 0):
    """
    Simple sparse polynomial:
        S = {0, ..., k_active-1}
        f(x) = sum_i a_i x_i + sum_{i<j} b_ij x_i x_j
    """
    torch.manual_seed(seed)
    S = list(range(k_active))
    a = torch.randn(k_active) * 1.0
    b = torch.randn(k_active, k_active) * 0.5
    b = torch.triu(b, diagonal=1)  # only i<j part
    return S, a, b


def sparse_poly_f(x: torch.Tensor, S, a, b):
    """
    x: [B, D]  ->  f(x): [B]
    """
    xS = x[:, S]  # [B, k]
    B, k = xS.shape

    # Linear
    lin = (xS * a.unsqueeze(0)).sum(dim=1)

    # Quadratic
    quad = torch.zeros(B, device=x.device)
    for i in range(k):
        for j in range(i + 1, k):
            quad += b[i, j] * xS[:, i] * xS[:, j]

    return lin + quad


# -----------------------
# Dataset
# -----------------------

class SparsePolyDataset(Dataset):
    def __init__(self, n, d, S, a, b, noise_std=0.0, seed=0):
        torch.manual_seed(seed)
        # Features in [-1, 1]
        self.x = 2.0 * torch.rand(n, d) - 1.0
        y = sparse_poly_f(self.x, S, a, b)
        if noise_std > 0.0:
            y = y + noise_std * torch.randn_like(y)
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def r2_score(y_true, y_pred) -> float:
    y_true = y_true.detach()
    y_pred = y_pred.detach()
    var = torch.var(y_true)
    if var < 1e-12:
        return 1.0 if torch.allclose(y_true, y_pred) else 0.0
    return float(1.0 - torch.mean((y_true - y_pred) ** 2) / (var + 1e-12))


# -----------------------
# Main training loop
# -----------------------

def main():
    print(f"Device: {DEVICE}")

    # Teacher coefficients
    S, a, b = make_sparse_poly_coeffs(D, k_active=5, seed=0)
    print(f"Active set S (true important features): {S}")

    # Datasets
    train_ds = SparsePolyDataset(N_TRAIN, D, S, a, b, noise_std=NOISE_STD, seed=1)
    test_ds  = SparsePolyDataset(N_TEST,  D, S, a, b, noise_std=NOISE_STD, seed=2)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # MPS regressor
    mps = MPS(
        input_dim=D,
        output_dim=1,       # regression => scalar
        bond_dim=BOND_DIM,
        adaptive_mode=False,
        periodic_bc=False,
    ).to(DEVICE)

    loss_fun = nn.MSELoss()
    optimizer = torch.optim.Adam(mps.parameters(), lr=LEARN_RATE, weight_decay=L2_REG)

    print(
        f"\nTraining MPS on sparse polynomial regression\n"
        f"D={D}, bond_dim={BOND_DIM}, N_train={N_TRAIN}, N_test={N_TEST}\n"
    )

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        # ---- Train ----
        mps.train()
        train_loss = 0.0
        n_seen = 0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            preds = mps(x).squeeze(-1)     # [B, 1] -> [B]
            loss = loss_fun(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            n_seen += x.size(0)

        train_mse = train_loss / max(n_seen, 1)

        # ---- Eval ----
        mps.eval()
        with torch.no_grad():
            ys = []
            preds = []
            for x, y in test_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                p = mps(x).squeeze(-1)
                ys.append(y)
                preds.append(p)

            ys = torch.cat(ys, dim=0)
            preds = torch.cat(preds, dim=0)
            test_mse = float(torch.mean((ys - preds) ** 2))
            test_r2 = r2_score(ys, preds)

        print(f"### Epoch {epoch:03d} ###")
        print(f"Train MSE: {train_mse:.5f}")
        print(f"Test  MSE: {test_mse:.5f} | R2: {test_r2:.4f}")
        print(f"Runtime so far: {int(time.time() - start_time)} sec\n")

    print("Done.")


if __name__ == "__main__":
    main()
