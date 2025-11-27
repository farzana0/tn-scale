#!/usr/bin/env python3
"""
train_mps_paths.py

Train an MPS on path-augmented data for TN-SHAP:

  For each base input x (from the original training set), we generate:
    - h-path:  x_h(t) = t * x
    - g_i paths: x_g^{(i)}(t)[j] = x[j] if j==i else t * x[j]

for t in Chebyshev nodes in [0, 1], using the SAME chebyshev_nodes_unit_interval
function as later used in TN-SHAP Vandermonde interpolation.

Targets are y = teacher(x_path), so the MPS is explicitly trained
on the exact path-queries TN-SHAP will make for these base points.

We use:
  - A subset of N_TARGETS training points as base points (default 100).
  - All h and g_i paths for those base points.

This version is deliberately SIMPLE and STABLE:
  - No AMP, no compile, no fancy worker config.
  - Gradient clipping.
  - Divergence guard and best-checkpoint restore.
"""

import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from torchmps import MPS
from poly_teacher import load_teacher, load_data, DEVICE


# -----------------------
# Utilities
# -----------------------

def r2_score(y_true, y_pred) -> float:
    y_true = y_true.detach()
    y_pred = y_pred.detach()
    var = torch.var(y_true)
    if var < 1e-12:
        return 1.0 if torch.allclose(y_true, y_pred) else 0.0
    return float(1.0 - torch.mean((y_true - y_pred) ** 2) / (var + 1e-12))


def chebyshev_nodes_unit_interval(n_nodes: int, device=None, dtype=torch.float32):
    """
    Chebyshev nodes of the first kind, mapped from [-1, 1] to [0, 1].
    """
    if device is None:
        device = DEVICE
    k = torch.arange(1, n_nodes + 1, dtype=torch.float64, device=device)
    u = torch.cos((2.0 * k - 1.0) / (2.0 * n_nodes) * torch.pi)  # [-1,1]
    t = (u + 1.0) / 2.0  # [0,1]
    return t.to(dtype).to(device)


def augment_with_one(x: torch.Tensor) -> torch.Tensor:
    """
    x: (N, D) -> (N, D+1) with a leading 1 column.
    """
    ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
    return torch.cat([ones, x], dim=1)


# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix", type=str, default="poly",
        help="Prefix used to load teacher/data and save MPS/targets."
    )
    parser.add_argument(
        "--max-degree", type=int, default=5,
        help="Polynomial degree in t for TN-SHAP interpolation."
    )
    parser.add_argument(
        "--n-targets", type=int, default=100,
        help="Number of base points whose paths we include for training."
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--bond-dim", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--l2-reg", type=float, default=1e-2)
    # Ignored but kept for compatibility with older scripts:
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--divergence-factor", type=float, default=50.0,
        help="If MSE exceeds this factor * best_MSE, treat as divergence and stop."
    )
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Prefix: {args.prefix}, max_degree={args.max_degree}, N_targets={args.n_targets}")

    # ---------------------------------------------------------
    # 1) Load teacher and original data
    # ---------------------------------------------------------
    teacher = load_teacher(args.prefix)
    x_train, y_train, x_test, y_test = load_data(args.prefix)

    x_train = x_train.to(DEVICE)
    y_train = y_train.to(DEVICE)

    N_total, D = x_train.shape
    N_base = min(args.n_targets, N_total)
    x_base = x_train[:N_base]  # (N_base, D)

    print(f"Using N_base={N_base} base points out of {N_total}, D={D}")

    # ---------------------------------------------------------
    # 2) Build Chebyshev t in [0,1] for path augmentation
    # ---------------------------------------------------------
    N_T_NODES = args.max_degree + 1  # exact interpolation if degree <= max_degree
    t_nodes = chebyshev_nodes_unit_interval(
        N_T_NODES, device=DEVICE, dtype=x_base.dtype
    )
    print(f"Using N_T_NODES={N_T_NODES} Chebyshev nodes in t for path augmentation.")

    # ---------------------------------------------------------
    # 3) Construct path-augmented dataset: all h(t) and all g_i(t)
    # ---------------------------------------------------------
    total_samples = (1 + D) * N_base * N_T_NODES
    X_all = torch.zeros(total_samples, D, device=DEVICE, dtype=x_base.dtype)
    Y_all = torch.zeros(total_samples, device=DEVICE, dtype=x_base.dtype)

    teacher.eval()
    idx = 0

    print("Generating path-augmented dataset...")
    with torch.no_grad():
        for b in range(N_base):
            x0 = x_base[b]  # (D,)

            for t in t_nodes:
                # h-path
                x_h = t * x0
                y_h = teacher(x_h.unsqueeze(0)).squeeze(0)
                if y_h.ndim > 0:
                    y_h = y_h.squeeze(-1)
                X_all[idx] = x_h
                Y_all[idx] = y_h
                idx += 1

                # g_i paths (vectorized)
                x_g_batch = t * x0.unsqueeze(0).expand(D, -1).clone()  # (D, D)
                x_g_batch[torch.arange(D), torch.arange(D)] = x0
                y_g_batch = teacher(x_g_batch).squeeze(-1)
                if y_g_batch.ndim > 1:
                    y_g_batch = y_g_batch.squeeze(-1)

                X_all[idx:idx + D] = x_g_batch
                Y_all[idx:idx + D] = y_g_batch
                idx += D

    print(
        f"Path-augmented dataset size (h + g_i paths): "
        f"{X_all.shape[0]} points = (1 + D)*N_base*N_T_NODES"
    )

    # ---------------------------------------------------------
    # 4) Train MPS on X_all
    # ---------------------------------------------------------
    X_all_aug = augment_with_one(X_all)  # (N_all, D+1)
    D_aug = X_all_aug.shape[1]

    # Keep tensors on DEVICE and use a simple DataLoader
    train_ds = TensorDataset(X_all_aug, Y_all)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    mps = MPS(
        input_dim=D_aug,
        output_dim=1,
        bond_dim=args.bond_dim,
        adaptive_mode=False,
        periodic_bc=False,
    ).to(DEVICE)

    loss_fun = nn.MSELoss()
    optimizer = torch.optim.Adam(
        mps.parameters(), lr=args.lr, weight_decay=args.l2_reg
    )

    max_grad_norm = 1.0  # gradient clipping
    print(
        f"\nTraining MPS on path-augmented regression\n"
        f"D_aug={D_aug}, bond_dim={args.bond_dim}, "
        f"N_train={X_all_aug.shape[0]} (no held-out test set)\n"
    )

    train_start = time.time()
    best_mse = float("inf")
    best_r2 = -float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(1, args.num_epochs + 1):
        mps.train()
        train_loss = 0.0
        n_seen = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            preds = mps(xb).squeeze(-1)
            loss = loss_fun(preds, yb)

            optimizer.zero_grad()
            loss.backward()

            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(mps.parameters(), max_grad_norm)

            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)

        train_mse = train_loss / max(n_seen, 1)

        # Evaluate R² on whole training set (batched)
        mps.eval()
        with torch.no_grad():
            y_true_list = []
            y_pred_list = []
            for xb_eval, yb_eval in train_loader:
                xb_eval = xb_eval.to(DEVICE)
                yb_eval = yb_eval.to(DEVICE)
                p_eval = mps(xb_eval).squeeze(-1)
                y_true_list.append(yb_eval)
                y_pred_list.append(p_eval)

            Y_all_eval = torch.cat(y_true_list, dim=0)
            preds_all = torch.cat(y_pred_list, dim=0)
            train_r2 = r2_score(Y_all_eval, preds_all)

        # Track best model & divergence
        improved = False
        if train_r2 > best_r2 + 1e-5:
            best_r2 = train_r2
            best_mse = train_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in mps.state_dict().items()}
            improved = True

        if (not torch.isfinite(torch.tensor(train_mse))) or \
           (best_mse < float("inf") and train_mse > args.divergence_factor * best_mse):
            print("\n⚠️  Detected divergence (loss explosion).")
            print(f"Stopping at epoch {epoch}. Restoring best model from epoch {best_epoch} "
                  f"(MSE={best_mse:.5f}, R²={best_r2:.4f})")
            break

        elapsed = int(time.time() - train_start)
        print(f"### Epoch {epoch:03d} ###")
        print(f"Train MSE: {train_mse:.5f}")
        print(f"Train R2 (on path-augmented data): {train_r2:.4f}")
        print(f"Best so far: epoch {best_epoch}, MSE={best_mse:.5f}, R²={best_r2:.4f}, "
              f"{'Improved' if improved else 'No improve'}")
        print(f"Runtime so far: {elapsed} sec\n")

        # Soft early-exit if we’re already very good
        if best_r2 > 0.995 and epoch >= 10:
            print(f"\n✓ Early exit: reached high R²={best_r2:.4f} at epoch {best_epoch}")
            break

    # ---------------------------------------------------------
    # Restore best model if we have it
    # ---------------------------------------------------------
    if best_state is not None:
        mps.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        print(f"\nRestored best MPS from epoch {best_epoch} "
              f"(MSE={best_mse:.5f}, R²={best_r2:.4f})")
    else:
        print("\n⚠️ No best_state recorded; using last epoch parameters.")

    # ---------------------------------------------------------
    # 5) Save TN-SHAP targets and full training dataset
    # ---------------------------------------------------------
    torch.save(
        {
            "x_base": x_base.detach().cpu(),   # (N_base, D)
            "t_nodes": t_nodes.detach().cpu(), # (N_T_NODES,)
            "X_all": X_all.detach().cpu(),     # ((1+D)*N_base*N_T_NODES, D)
            "Y_all": Y_all.detach().cpu(),     # same length
            "max_degree": args.max_degree,
        },
        f"{args.prefix}_tnshap_targets.pt",
    )
    print(f"Saved TN-SHAP targets and t-nodes to {args.prefix}_tnshap_targets.pt")

    # ---------------------------------------------------------
    # 6) Save trained MPS
    # ---------------------------------------------------------
    torch.save(
        {
            "state_dict": mps.state_dict(),
            "D_aug": D_aug,
            "bond_dim": args.bond_dim,
        },
        f"{args.prefix}_mps.pt",
    )
    print(f"Saved trained MPS to {args.prefix}_mps.pt")
    print("Done.")


if __name__ == "__main__":
    main()
