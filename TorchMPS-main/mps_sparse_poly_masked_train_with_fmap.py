#!/usr/bin/env python3
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchmps import MPS

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D = 50
M_NODES = 20
K_POINTS = 10
BOND_DIM = 20
FEATURE_DIM = 3   # [1, x, x^2]

TRAIN_BATCH_SIZE = 128
NUM_EPOCHS = 2
LR = 1e-3
TEST_FRAC = 0.05
SEED = 123

CKPT_PATH = "mps_with_featuremap.pt"


# ---------------- FEATURE MAP ----------------
def poly_feature_map(x: torch.Tensor) -> torch.Tensor:
    """
    TorchMPS feature map:
    x: scalar tensor
    returns: [1, x, x^2]
    """
    return torch.stack([torch.ones_like(x), x, x ** 2], dim=-1)


# ---------------- CHEBYSHEV ----------------
def chebyshev_nodes_01(m: int):
    k = torch.arange(m, device=DEVICE, dtype=torch.float32)
    nodes = torch.cos((2 * k + 1) * torch.pi / (2 * m))
    return (nodes + 1.) / 2.


# ---------------- SPARSE POLY ----------------
def sparse_poly_f(x, S, a, b):
    xS = x[:, S]
    lin = (xS * a).sum(dim=1)
    quad = torch.zeros_like(lin)
    k = len(S)
    for i in range(k):
        for j in range(i + 1, k):
            quad += b[i, j] * xS[:, i] * xS[:, j]
    return lin + quad


# ---------------- DATASET BUILDER ----------------
def build_masked_dataset(X_targets, S, a, b):
    t = chebyshev_nodes_01(M_NODES)
    K = X_targets.shape[0]

    base = (t[:, None] * X_targets[:, None, :]).reshape(-1, D)

    X_list = []
    y_list = []

    # unmasked
    X_list.append(base)
    y_list.append(sparse_poly_f(base, S, a, b))

    # masked
    for i in range(D):
        xm = base.clone()
        xm[:, i] = 0.
        X_list.append(xm)
        y_list.append(sparse_poly_f(xm, S, a, b))

    X = torch.cat(X_list, 0)
    y = torch.cat(y_list, 0)

    return X, y


# ---------------- R2 ----------------
def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float((1 - ss_res / (ss_tot + 1e-12)).item())


# ---------------- MAIN ----------------
def main():
    torch.manual_seed(SEED)

    print("Building sparse polynomial teacher...")

    S = [0, 1, 2, 3, 4]
    a = torch.randn(len(S), device=DEVICE)
    b = torch.randn(len(S), len(S), device=DEVICE)

    k = len(S)
    for i in range(k):
        b[i, :i+1] = 0.

    X_targets = 2 * torch.rand(K_POINTS, D, device=DEVICE) - 1

    print("Building masked Chebyshev dataset (GPU)...")
    X, y = build_masked_dataset(X_targets, S, a, b)

    N = X.shape[0]
    n_test = int(TEST_FRAC * N)

    perm = torch.randperm(N, device=DEVICE)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    print(f"Train: {X_train.shape}   Test: {X_test.shape}")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True
    )

    # ---- MPS with feature map ----
    mps = MPS(
        input_dim=D,
        output_dim=1,
        bond_dim=BOND_DIM,
        adaptive_mode=False,
        periodic_bc=False,
        feature_dim=FEATURE_DIM
        
    ).to(DEVICE)

    # register feature map
    mps.register_feature_map(poly_feature_map)
    mps.parallel_eval = True


    opt = torch.optim.Adam(mps.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print("Training MPS with feature map...")
    for epoch in range(1, NUM_EPOCHS + 1):
        # print(epoch)
        mps.train()
        total_loss = 0.
        seen = 0
        # print(epoch)

        for xb, yb in train_loader:
            # print(epoch)
            preds = mps(xb).squeeze(-1)
            loss = loss_fn(preds, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * xb.shape[0]
            seen += xb.shape[0]

        mps.eval()
        with torch.no_grad():
            pred_train = mps(X_train).squeeze(-1)
            pred_test = mps(X_test).squeeze(-1)

            train_mse = loss_fn(pred_train, y_train).item()
            test_mse = loss_fn(pred_test, y_test).item()

            train_r2 = r2_score(y_train, pred_train)
            test_r2 = r2_score(y_test, pred_test)

        print(
            f"Epoch {epoch:03d} | "
            f"Train MSE: {train_mse:.5f}, R2: {train_r2:.4f} | "
            f"Test MSE: {test_mse:.5f}, R2: {test_r2:.4f}"
        )

    # ---- Save checkpoint ----
    torch.save({
        "state_dict": mps.state_dict(),
        "S": S,
        "a": a.cpu(),
        "b": b.cpu(),
        "X_targets": X_targets.cpu(),
        "config": {
            "D": D,
            "bond_dim": BOND_DIM,
            "feature_dim": FEATURE_DIM
        }
    }, CKPT_PATH)

    print(f"Saved model to {CKPT_PATH}")


if __name__ == "__main__":
    main()
