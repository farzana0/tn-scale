#!/usr/bin/env python3
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchmps import MPS

# ----------------- Global config -----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D = 50              # number of original features
M_NODES = 50        # number of Chebyshev nodes
K_POINTS = 100      # number of target points to amortize over
BOND_DIM = 20
NOISE_STD = 0.0

TRAIN_BATCH_SIZE = 1024
NUM_EPOCHS = 30
LR = 1e-3
SEED = 123
TEST_FRACTION = 0.05

CKPT_PATH = "mps_sparse_poly_masked.pt"


# ----------------- Utils -----------------

def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def chebyshev_nodes_01(m: int) -> np.ndarray:
    """Chebyshev–Gauss nodes mapped to [0,1]."""
    if m <= 0:
        return np.zeros((0,), dtype=np.float32)
    k = np.arange(m, dtype=np.float64)
    nodes = np.cos((2 * k + 1) * np.pi / (2 * m))
    t = (nodes + 1.0) * 0.5
    return t.astype(np.float32)


def sparse_poly_f_batch(
    x: torch.Tensor,
    S: List[int],
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Sparse polynomial teacher, batched:

      f(x) = sum_{i in S} a_i x_i
           + sum_{i<j, i,j in S} b_{ij} x_i x_j

    Args:
        x: [N, D] (same device as a, b)
        S: active feature indices (length k)
        a: [k]
        b: [k, k] (only upper triangle i<j is used)

    Returns:
        y: [N]
    """
    xS = x[:, S]      # [N, k]
    N, k = xS.shape

    # linear term
    lin = (xS * a.unsqueeze(0)).sum(dim=1)   # [N]

    # quadratic term (loop only over k^2, k is small, e.g. 5)
    quad = torch.zeros(N, device=x.device, dtype=x.dtype)
    for i in range(k):
        for j in range(i + 1, k):
            quad += b[i, j] * xS[:, i] * xS[:, j]

    return lin + quad


def build_masked_chebyshev_dataset_gpu(
    X_targets: torch.Tensor,
    S: List[int],
    a: torch.Tensor,
    b: torch.Tensor,
    m_nodes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Efficient GPU builder for masked Chebyshev dataset with appended 1.

    For each target x_r (r=1..K) and each Chebyshev node t_l (l=1..m):

        base = t_l * x_r ∈ R^D

        - one unmasked sample:   [base, 1]
        - for each feature i in {0,...,D-1}:
            masked_i = base with feature i set to 0
            sample: [masked_i, 1]

    Total samples:
        N = K * m * (D + 1)

    Returns:
        X_ext: [N, D+1], on `device`
        y:     [N], on `device`
    """
    X_targets = X_targets.to(device)         # [K, D]
    a = a.to(device)
    b = b.to(device)

    K, D_local = X_targets.shape
    assert D_local == D, f"Expected D={D}, got {D_local}"

    # 1) Chebyshev nodes on [0,1]
    t_np = chebyshev_nodes_01(m_nodes)                     # [m]
    t = torch.tensor(t_np, dtype=torch.float32, device=device)  # [m]

    # 2) Build base path grid: [K, m, D]
    #    base[r,l,:] = t[l] * X_targets[r,:]
    base = t.view(1, -1, 1) * X_targets.view(K, 1, D)      # [K, m, D]

    # 3) Flatten base to [B0, D], where B0 = K * m
    base_flat = base.reshape(-1, D)                        # [B0, D]
    B0 = base_flat.shape[0]

    # 4) Unmasked samples: [B0, D+1]
    ones_base = torch.ones(B0, 1, device=device, dtype=base_flat.dtype)
    X_unmasked = torch.cat([base_flat, ones_base], dim=1)  # [B0, D+1]
    y_unmasked = sparse_poly_f_batch(base_flat, S, a, b)   # [B0]

    X_list = [X_unmasked]
    y_list = [y_unmasked]

    # 5) Masked samples for each feature i ∈ {0,...,D-1}
    #    We loop over features but all operations per feature are batched on GPU.
    for i in range(D):
        x_mask = base_flat.clone()
        x_mask[:, i] = 0.0

        y_mask = sparse_poly_f_batch(x_mask, S, a, b)              # [B0]
        ones_mask = torch.ones(B0, 1, device=device, dtype=x_mask.dtype)
        X_mask_ext = torch.cat([x_mask, ones_mask], dim=1)         # [B0, D+1]

        X_list.append(X_mask_ext)
        y_list.append(y_mask)

    # 6) Concatenate all
    X_ext = torch.cat(X_list, dim=0)   # [B0*(D+1), D+1]
    y_all = torch.cat(y_list, dim=0)   # [B0*(D+1)]

    return X_ext, y_all


def r2_score_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute R² score in torch on the same device.
    """
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return float(r2.item())


# ----------------- Main -----------------

def main():
    set_all_seeds(SEED)
    print(f"Using device: {DEVICE}")
    print(f"D={D}, M_NODES={M_NODES}, K_POINTS={K_POINTS}, BOND_DIM={BOND_DIM}")

    # 1) Define sparse polynomial structure
    k_active = 5
    # choose a fixed S for reproducibility (you can randomize if you like)
    S = [0, 7, 13, 27, 42]
    assert len(S) == k_active

    a = torch.randn(k_active)       # [k]
    b = torch.randn(k_active, k_active)
    # zero diagonal and lower triangle (only i<j used)
    for i in range(k_active):
        for j in range(i + 1):
            b[i, j] = 0.0

    print(f"Active set S: {S}")
    print(f"a (linear coeffs): {a}")
    print(f"b (upper-tri quadratic coeffs):\n{b}")

    # 2) Sample K target points x ∈ [-1,1]^D
    X_targets = 2.0 * torch.rand(K_POINTS, D) - 1.0   # [K, D]
    print(f"Sampled X_targets of shape: {X_targets.shape}")

    # 3) Build masked Chebyshev dataset on GPU
    print("Building masked Chebyshev dataset on GPU (with appended 1)...")
    t0 = time.time()
    X_ext, y_all = build_masked_chebyshev_dataset_gpu(
        X_targets=X_targets,
        S=S,
        a=a,
        b=b,
        m_nodes=M_NODES,
        device=DEVICE,
    )
    build_time = time.time() - t0
    N = X_ext.shape[0]
    print(f"Masked dataset built in {build_time:.2f} s.")
    print(f"X_ext shape: {X_ext.shape}  (expected N ≈ K * M_NODES * (D+1) = {K_POINTS*M_NODES*(D+1)})")
    print(f"y_all shape: {y_all.shape}")

    # Optional noise
    if NOISE_STD > 0.0:
        y_all = y_all + NOISE_STD * torch.randn_like(y_all)

    # 4) Train/test split (5% test)
    n_test = max(1, int(TEST_FRACTION * N))
    perm = torch.randperm(N, device=DEVICE)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    X_train = X_ext[train_idx]
    y_train = y_all[train_idx]
    X_test = X_ext[test_idx]
    y_test = y_all[test_idx]

    print(f"Train size: {X_train.shape[0]}   Test size: {X_test.shape[0]}")

    # 5) DataLoader for train
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    # 6) MPS surrogate
    D_ext = D + 1   # appended bias 1
    mps = MPS(
        input_dim=D_ext,
        output_dim=1,
        bond_dim=BOND_DIM,
        adaptive_mode=False,
        periodic_bc=False,
        # If TorchMPS exposes feature_map options, you could pass them here.
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(mps.parameters(), lr=LR)

    print("\nTraining MPS surrogate on masked Chebyshev dataset...")
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        mps.train()
        running_loss = 0.0
        n_seen = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            preds = mps(xb)
            if preds.ndim > 1:
                preds = preds.squeeze(-1)

            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)

        train_mse_epoch = running_loss / max(n_seen, 1)

        # Evaluate full train/test MSE + R2
        mps.eval()
        with torch.no_grad():
            # Train metrics
            y_pred_train = mps(X_train)
            if y_pred_train.ndim > 1:
                y_pred_train = y_pred_train.squeeze(-1)
            train_mse = criterion(y_pred_train, y_train).item()
            train_r2 = r2_score_torch(y_train, y_pred_train)

            # Test metrics
            y_pred_test = mps(X_test)
            if y_pred_test.ndim > 1:
                y_pred_test = y_pred_test.squeeze(-1)
            test_mse = criterion(y_pred_test, y_test).item()
            test_r2 = r2_score_torch(y_test, y_pred_test)

        elapsed = int(time.time() - start_time)
        print(
            f"### Epoch {epoch:03d} ###\n"
            f"Train MSE (batch avg): {train_mse_epoch:.5f}\n"
            f"Train MSE (full):      {train_mse:.5f} | R2: {train_r2:.4f}\n"
            f"Test  MSE:             {test_mse:.5f} | R2: {test_r2:.4f}\n"
            f"Runtime so far:        {elapsed} sec\n"
        )

    # Final metrics
    print("Final evaluation on train/test splits:")
    mps.eval()
    with torch.no_grad():
        y_pred_train = mps(X_train)
        if y_pred_train.ndim > 1:
            y_pred_train = y_pred_train.squeeze(-1)
        train_mse = criterion(y_pred_train, y_train).item()
        train_r2 = r2_score_torch(y_train, y_pred_train)

        y_pred_test = mps(X_test)
        if y_pred_test.ndim > 1:
            y_pred_test = y_pred_test.squeeze(-1)
        test_mse = criterion(y_pred_test, y_test).item()
        test_r2 = r2_score_torch(y_test, y_pred_test)

    print(
        f"FINAL | Train MSE: {train_mse:.5f} | Train R2: {train_r2:.4f} | "
        f"Test MSE: {test_mse:.5f} | Test R2: {test_r2:.4f}"
    )

    # 7) Save checkpoint
    ckpt = {
        "state_dict": mps.state_dict(),
        "config": {
            "D": D,
            "D_ext": D_ext,
            "bond_dim": BOND_DIM,
            "M_NODES": M_NODES,
            "K_POINTS": K_POINTS,
            "noise_std": NOISE_STD,
        },
        "active_set_S": S,
        "a": a.cpu(),
        "b": b.cpu(),
        "X_targets": X_targets.cpu(),  # [K, D]
    }
    torch.save(ckpt, CKPT_PATH)
    print(f"Saved masked MPS surrogate checkpoint to: {CKPT_PATH}")


if __name__ == "__main__":
    main()
