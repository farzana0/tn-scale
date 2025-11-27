#!/usr/bin/env python3
"""
train_mps_mlpfeat_paths.py

Train an MPS on path-augmented data for TN-SHAP, using **MLP-based local feature maps**:

  - Teacher lives in ORIGINAL x-space, f: R^D -> R.
  - For each base x, we build:
        h-path:  x_h(t)   = t * x
        g_i-path: x_g^i(t)[j] = x[j] if j == i else t * x[j]
    with t taken at Chebyshev nodes in [0, 1].

  - We train an MPS surrogate on these ORIGINAL path-augmented points.
    The MPS uses a *trainable* scalar feature map per coordinate:

        phi_scalar(x_i) = MLP(x_i) in R^3
        phi(x_i) = [ MLP(x_i), 1 ] in R^4

    where MLP has **2 hidden layers**.

This script is generic: you can use it for
  - degree-10 polynomial teacher (e.g. prefix "poly10_D50")
  - squared-exponential teacher (e.g. prefix "sqexp_D50")

The TN-SHAP interpolation degree in t is controlled by --max-degree.

We save:
  - <PREFIX>_tnshap_targets.pt: x_base, t_nodes, X_all, Y_all, max_degree, feature_map name
  - <PREFIX>_mps_mlpfeat.pt: MPS+MLP state_dict and metadata
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

    u_k = cos((2k - 1) / (2n) * pi), k = 1..n
    t   = (u + 1) / 2 in [0, 1]
    """
    if device is None:
        device = DEVICE
    k = torch.arange(1, n_nodes + 1, dtype=torch.float64, device=device)
    u = torch.cos((2.0 * k - 1.0) / (2.0 * n_nodes) * torch.pi)  # [-1, 1]
    t = (u + 1.0) / 2.0  # [0, 1]
    return t.to(dtype).to(device)


# -----------------------
# Scalar MLP feature map: R -> R^3, then append 1 -> R^4
# -----------------------

class ScalarMLPFeatureMap(nn.Module):
    """
    Per-coordinate feature map with two hidden layers.

    Input:  scalar x_i
    MLP:    1 -> H -> H -> 3
    Output: phi(x_i) in R^4 = [mlp(x_i), 1]

    The forward accepts tensors of arbitrary shape (...,) and returns (..., 4).
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (...,)  (scalar per site)
        returns: (..., 4)  with last dim = [f1, f2, f3, 1]
        """
        orig_shape = x.shape  # e.g. (batch, D)
        x_flat = x.reshape(-1, 1)  # (N, 1)

        h_flat = self.net(x_flat)  # (N, 3)
        h = h_flat.view(*orig_shape, 3)  # (..., 3)

        ones = torch.ones(*orig_shape, 1, device=x.device, dtype=x.dtype)
        phi = torch.cat([h, ones], dim=-1)  # (..., 4)
        return phi


# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix", type=str, default="poly10_D50",
        help="Prefix used to load teacher/data and save MPS/targets "
             "(e.g. 'poly10_D50' or 'sqexp_D50')."
    )
    parser.add_argument(
        "--max-degree", type=int, default=10,
        help="Polynomial degree in t for TN-SHAP interpolation."
    )
    parser.add_argument(
        "--n-targets", type=int, default=20,
        help="Number of base points whose paths we include for training."
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--bond-dim", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2-reg", type=float, default=0.0)

    # Scheduler & early stopping
    parser.add_argument("--lr-factor", type=float, default=0.8,
                        help="Factor for ReduceLROnPlateau.")
    parser.add_argument("--lr-patience", type=int, default=5,
                        help="Patience (epochs) before LR reduction.")
    parser.add_argument("--min-lr", type=float, default=1e-7,
                        help="Minimum learning rate.")
    parser.add_argument("--early-stop-patience", type=int, default=30,
                        help="Epochs without R² improvement before early stop.")

    # Gradient clipping
    parser.add_argument("--grad-clip", type=float, default=5.0,
                        help="Max gradient norm for clipping.")

    # Misc
    parser.add_argument("--divergence-factor", type=float, default=50.0,
                        help="If MSE exceeds this factor * best_MSE, treat as divergence and stop.")

    # MLP size (optional)
    parser.add_argument("--mlp-hidden", type=int, default=32,
                        help="Hidden dimension for scalar MLP feature map.")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Prefix: {args.prefix}, max_degree={args.max_degree}, N_targets={args.n_targets}")
    print(f"LR={args.lr}, LR factor={args.lr_factor}, LR patience={args.lr_patience}, "
          f"min_lr={args.min_lr}, early_stop_patience={args.early_stop_patience}")
    print(f"Grad clip: {args.grad_clip}, bond_dim={args.bond_dim}, l2_reg={args.l2_reg}, "
          f"mlp_hidden={args.mlp_hidden}")

    # ---------------------------------------------------------
    # 1) Load teacher and original data
    # ---------------------------------------------------------
    teacher = load_teacher(args.prefix)
    if isinstance(teacher, torch.nn.Module):
        teacher.to(DEVICE)

    x_train, y_train, x_test, y_test = load_data(args.prefix)
    x_train = x_train.to(DEVICE)

    N_total, D = x_train.shape
    N_base = min(args.n_targets, N_total)
    x_base = x_train[:N_base]  # (N_base, D)

    print(f"Using N_base={N_base} base points out of {N_total}, D={D}")

    # ---------------------------------------------------------
    # 2) Build Chebyshev t in [0,1] for path augmentation
    # ---------------------------------------------------------
    N_T_NODES = args.max_degree + 1  # exact for deg <= max_degree
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
                # h-path: x_h(t) = t * x0
                x_h = t * x0
                y_h = teacher(x_h.unsqueeze(0)).squeeze(0)
                if y_h.ndim > 0:
                    y_h = y_h.squeeze(-1)
                X_all[idx] = x_h
                Y_all[idx] = y_h
                idx += 1

                # g_i paths (vectorized)
                x_g_batch = t * x0.unsqueeze(0).expand(D, -1).clone()  # (D, D)
                x_g_batch[torch.arange(D), torch.arange(D)] = x0  # clamp each diag
                y_g_batch = teacher(x_g_batch)
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
    # 4) Train MPS on ORIGINAL X_all with scalar MLP feature map
    # ---------------------------------------------------------
    train_ds = TensorDataset(X_all, Y_all)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, X_all.shape[0]),  # full-batch if large
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    mps = MPS(
        input_dim=D,
        output_dim=1,
        bond_dim=args.bond_dim,
        adaptive_mode=False,
        periodic_bc=False,
        feature_dim=4,        # phi(s_i) in R^4
        parallel_eval=True,   # faster on GPU
    ).to(DEVICE)

    mlp_feat = ScalarMLPFeatureMap(hidden_dim=args.mlp_hidden).to(DEVICE)
    mps.register_feature_map(mlp_feat)

    loss_fun = nn.MSELoss()
    optimizer = torch.optim.Adam(
        mps.parameters(), lr=args.lr, weight_decay=args.l2_reg
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",              # maximize R²
        factor=args.lr_factor,
        patience=args.lr_patience,
        verbose=True,
        min_lr=args.min_lr,
        threshold=1e-4,
    )

    max_grad_norm = args.grad_clip
    print(
        f"\nTraining MPS+MLP on path-augmented regression\n"
        f"D={D} (original space), bond_dim={args.bond_dim}, "
        f"N_train={X_all.shape[0]} (no held-out test set)\n"
    )

    train_start = time.time()
    best_mse = float("inf")
    best_r2 = -float("inf")
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.num_epochs + 1):
        mps.train()
        train_loss = 0.0
        n_seen = 0
        grad_norms = []

        for xb, yb in train_loader:
            preds = mps(xb).squeeze(-1)
            loss = loss_fun(preds, yb)

            optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            total_norm = torch.nn.utils.clip_grad_norm_(mps.parameters(), max_grad_norm)
            grad_norms.append(total_norm.item())

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
                p_eval = mps(xb_eval).squeeze(-1)
                y_true_list.append(yb_eval)
                y_pred_list.append(p_eval)

            Y_all_eval = torch.cat(y_true_list, dim=0)
            preds_all = torch.cat(y_pred_list, dim=0)
            train_r2 = r2_score(Y_all_eval, preds_all)

        # Scheduler on R²
        scheduler.step(train_r2)
        current_lr = optimizer.param_groups[0]["lr"]
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0

        # Track best model & divergence
        improved = False
        if train_r2 > best_r2 + 1e-5:
            best_r2 = train_r2
            best_mse = train_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in mps.state_dict().items()}
            improved = True
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (not torch.isfinite(torch.tensor(train_mse, device=DEVICE))) or \
           (best_mse < float("inf") and train_mse > args.divergence_factor * best_mse):
            print("\n⚠️  Detected divergence (loss explosion).")
            print(f"Stopping at epoch {epoch}. Restoring best model from epoch {best_epoch} "
                  f"(MSE={best_mse:.5f}, R²={best_r2:.6f})")
            break

        elapsed = int(time.time() - train_start)
        print(f"### Epoch {epoch:03d} ###")
        print(f"Train MSE: {train_mse:.5f}")
        print(f"Train R2 (on path-augmented data): {train_r2:.6f}")
        print(f"Avg grad norm: {avg_grad_norm:.4f} | LR: {current_lr:.2e}")
        print(
            f"Best so far: epoch {best_epoch}, MSE={best_mse:.5f}, R²={best_r2:.6f}, "
            f"{'Improved' if improved else 'No improve'} | "
            f"No improve count: {epochs_no_improve}/{args.early_stop_patience}"
        )
        print(f"Runtime so far: {elapsed} sec\n")

        if epochs_no_improve >= args.early_stop_patience:
            print(
                f"\n⚠️ Early stopping at epoch {epoch}. "
                f"Restoring best model from epoch {best_epoch} (R²={best_r2:.6f})."
            )
            break

        # Soft early-exit if we’re already very good
        if best_r2 > 0.995 and epoch >= 10:
            print(f"\n✓ Early exit: reached high R²={best_r2:.6f} at epoch {best_epoch}")
            break

    # ---------------------------------------------------------
    # Restore best model if we have it
    # ---------------------------------------------------------
    if best_state is not None:
        mps.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        print(f"\nRestored best MPS from epoch {best_epoch} "
              f"(MSE={best_mse:.5f}, R²={best_r2:.6f})")
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
            "feature_map": "scalar_mlp3_plus_one",
        },
        f"{args.prefix}_tnshap_targets.pt",
    )
    print(f"Saved TN-SHAP targets and t-nodes to {args.prefix}_tnshap_targets.pt")

    # ---------------------------------------------------------
    # 6) Save trained MPS+MLP
    # ---------------------------------------------------------
    torch.save(
        {
            "state_dict": mps.state_dict(),
            "input_dim": D,
            "bond_dim": args.bond_dim,
            "mlp_hidden": args.mlp_hidden,
            "feature_map": "scalar_mlp3_plus_one",
        },
        f"{args.prefix}_mps_mlpfeat.pt",
    )
    print(f"Saved trained MPS+MLP to {args.prefix}_mps_mlpfeat.pt")
    print("Done.")


if __name__ == "__main__":
    main()
