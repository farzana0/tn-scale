#!/usr/bin/env python3
"""
train_mps_teacher_paths_compress.py

Goal:
  Compress the teacher f(x) along the TN-SHAP paths in ORIGINAL x-space,
  by training an MPS+local-MLP surrogate directly on the path-augmented
  dataset (X_all, Y_all).

Paths (in ORIGINAL space):
  - h-path:   x_h(t)   = t * x
  - g_i-path: x_g^i(t)[j] = x[j] if j == i else t * x[j]

We:
  - Load teacher and training data via poly_teacher.load_teacher/load_data.
  - Pick N_base base points from x_train.
  - For each base point and each Chebyshev node t in [0,1], build:
      * h-path sample
      * all g_i-path samples, i=0..D-1
  - Get targets y = f(x_path).
  - Train an MPS with a local MLP feature map φ: R -> R^{feature_dim}.
  - Save:
      <PREFIX>_tnshap_compress_targets.pt:
        x_base, t_nodes, X_all, Y_all, max_degree
      <PREFIX>_mps_compress.pt:
        state_dict, input_dim (D), bond_dim, feature_dim, hidden_dim
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
    Chebyshev nodes of the first kind, mapped from [-1, 1] to [0, 1]:

        u_k = cos((2k - 1)/(2n) * pi), k=1..n
        t   = (u + 1)/2
    """
    if device is None:
        device = DEVICE
    k = torch.arange(1, n_nodes + 1, dtype=torch.float64, device=device)
    u = torch.cos((2.0 * k - 1.0) / (2.0 * n_nodes) * torch.pi)
    t = (u + 1.0) / 2.0
    return t.to(dtype).to(device)


# -----------------------
# Local MLP feature map φ: R -> R^{feature_dim}
# -----------------------

class LocalMLPFeatureMap(nn.Module):
    """
    Per-site MLP feature map:

        φ(s) : R -> R^{feature_dim}

    TorchMPS will call this on tensors of shape (...,) and expects
    an output of shape (..., feature_dim).

    We make it device-agnostic by moving the input to the device of
    its parameters, so it works when register_feature_map internally
    calls φ(torch.tensor(0)) on CPU.
    """
    def __init__(self, feature_dim: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (...,)
        param_device = next(self.parameters()).device
        param_dtype = next(self.parameters()).dtype
        x = x.to(device=param_device, dtype=param_dtype)
        x_in = x.unsqueeze(-1)     # (..., 1)
        out = self.net(x_in)       # (..., feature_dim)
        return out


# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix", type=str, default="sqexp_D50",
        help="Prefix for teacher/data and saved outputs (e.g. 'sqexp_D50')."
    )
    parser.add_argument(
        "--max-degree", type=int, default=5,
        help="Polynomial degree in t for TN-SHAP interpolation; "
             "we use max_degree+1 Chebyshev nodes."
    )
    parser.add_argument(
        "--n-targets", type=int, default=20,
        help="Number of base points (from x_train) to build paths for."
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-epochs", type=int, default=300)
    parser.add_argument("--bond-dim", type=int, default=80)
    parser.add_argument("--feature-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2-reg", type=float, default=0.0)

    # Scheduler & early stopping
    parser.add_argument("--lr-factor", type=float, default=0.8)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--early_stop_patience", type=int, default=40)

    # Gradient clipping
    parser.add_argument("--grad-clip", type=float, default=5.0)

    # Divergence guard
    parser.add_argument("--divergence-factor", type=float, default=50.0)

    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Prefix={args.prefix}, max_degree={args.max_degree}, N_targets={args.n_targets}")
    print(
        f"bond_dim={args.bond_dim}, feature_dim={args.feature_dim}, "
        f"hidden_dim={args.hidden_dim}"
    )
    print(
        f"LR={args.lr}, weight_decay={args.l2_reg}, "
        f"LR factor={args.lr_factor}, LR patience={args.lr_patience}, min_lr={args.min_lr}"
    )
    print(
        f"Grad clip={args.grad_clip}, early_stop_patience={args.early_stop_patience}, "
        f"divergence_factor={args.divergence_factor}"
    )

    # ---------------------------------------------------------
    # 1) Load teacher and original data
    # ---------------------------------------------------------
    teacher = load_teacher(args.prefix)
    if isinstance(teacher, nn.Module):
        teacher.to(DEVICE)

    x_train, y_train, x_test, y_test = load_data(args.prefix)
    x_train = x_train.to(DEVICE)

    N_total, D = x_train.shape
    N_base = min(args.n_targets, N_total)
    x_base = x_train[:N_base]  # (N_base, D)

    print(f"\nUsing N_base={N_base} base points out of {N_total}, D={D}")

    # ---------------------------------------------------------
    # 2) Chebyshev t in [0,1]
    # ---------------------------------------------------------
    N_T_NODES = args.max_degree + 1
    t_nodes = chebyshev_nodes_unit_interval(
        N_T_NODES, device=DEVICE, dtype=x_train.dtype
    )
    print(f"Using N_T_NODES={N_T_NODES} Chebyshev nodes in t.\n")

    # ---------------------------------------------------------
    # 3) Build path-augmented dataset in ORIGINAL space
    # ---------------------------------------------------------
    total_samples = (1 + D) * N_base * N_T_NODES
    X_all = torch.zeros(total_samples, D, device=DEVICE, dtype=x_train.dtype)
    Y_all = torch.zeros(total_samples, device=DEVICE, dtype=x_train.dtype)

    teacher.eval()
    idx = 0

    print("Generating path-augmented dataset (h and g_i paths in ORIGINAL space)...")
    with torch.no_grad():
        for b in range(N_base):
            x0 = x_base[b]  # (D,)

            for t_val in t_nodes:
                # h-path
                x_h = t_val * x0
                y_h = teacher(x_h.unsqueeze(0)).squeeze(0)
                if y_h.ndim > 0:
                    y_h = y_h.squeeze(-1)

                X_all[idx] = x_h
                Y_all[idx] = y_h
                idx += 1

                # g_i paths, vectorized
                x_g_batch = t_val * x0.unsqueeze(0).expand(D, -1).clone()  # (D, D)
                x_g_batch[torch.arange(D), torch.arange(D)] = x0

                y_g_batch = teacher(x_g_batch)
                if y_g_batch.ndim > 1:
                    y_g_batch = y_g_batch.squeeze(-1)  # (D,)

                X_all[idx:idx + D] = x_g_batch
                Y_all[idx:idx + D] = y_g_batch
                idx += D

    assert idx == total_samples, f"idx={idx} but total_samples={total_samples}"
    print(
        f"Path-augmented dataset size: {X_all.shape[0]} "
        f"= (1 + D)*N_base*N_T_NODES\n"
    )

    # ---------------------------------------------------------
    # 4) Save TN-SHAP compression targets
    # ---------------------------------------------------------
    compress_targets_path = f"{args.prefix}_tnshap_compress_targets.pt"
    torch.save(
        {
            "x_base": x_base.detach().cpu(),    # (N_base, D)
            "t_nodes": t_nodes.detach().cpu(),  # (N_T_NODES,)
            "X_all": X_all.detach().cpu(),      # (N_all, D)
            "Y_all": Y_all.detach().cpu(),      # (N_all,)
            "max_degree": args.max_degree,
        },
        compress_targets_path,
    )
    print(f"Saved TN-SHAP compression targets to {compress_targets_path}\n")

    # ---------------------------------------------------------
    # 5) Train MPS+local-MLP on X_all -> Y_all
    # ---------------------------------------------------------
    train_ds = TensorDataset(X_all, Y_all)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, X_all.shape[0]),
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    print(
        f"Building MPS+MLP surrogate: D={D}, bond_dim={args.bond_dim}, "
        f"feature_dim={args.feature_dim}, hidden_dim={args.hidden_dim}"
    )

    mps = MPS(
        input_dim=D,
        output_dim=1,
        bond_dim=args.bond_dim,
        adaptive_mode=False,
        periodic_bc=False,
        feature_dim=args.feature_dim,
        parallel_eval=True,
    ).to(DEVICE)

    mlp_feature = LocalMLPFeatureMap(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
    ).to(DEVICE)

    # TorchMPS will call feature_map(torch.tensor(0)) internally;
    # LocalMLPFeatureMap moves that scalar to the correct device.
    mps.register_feature_map(mlp_feature)

    loss_fun = nn.MSELoss()
    optimizer = torch.optim.Adam(
        mps.parameters(), lr=args.lr, weight_decay=args.l2_reg
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",   # maximize R² on the full path-aug dataset
        factor=args.lr_factor,
        patience=args.lr_patience,
        verbose=True,
        min_lr=args.min_lr,
        threshold=1e-4,
    )

    max_grad_norm = args.grad_clip

    print(
        f"\nTraining MPS+MLP on path-augmented regression "
        f"(N_train={X_all.shape[0]}, D={D})\n"
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

            total_norm = torch.nn.utils.clip_grad_norm_(mps.parameters(), max_grad_norm)
            grad_norms.append(total_norm.item())
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)

        train_mse = train_loss / max(n_seen, 1)

        # Evaluate R² on the entire path-aug dataset (same loader)
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

        scheduler.step(train_r2)
        current_lr = optimizer.param_groups[0]["lr"]
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0

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

        # Divergence check
        if (not torch.isfinite(torch.tensor(train_mse, device=DEVICE))) or \
           (best_mse < float("inf") and train_mse > args.divergence_factor * best_mse):
            print("\n⚠️ Detected divergence (loss explosion).")
            print(
                f"Stopping at epoch {epoch}. Restoring best model from epoch {best_epoch} "
                f"(MSE={best_mse:.6e}, R²={best_r2:.6f})"
            )
            break

        elapsed = int(time.time() - train_start)
        print(f"### Epoch {epoch:03d} ###")
        print(f"Train MSE: {train_mse:.6e}")
        print(f"R² on path-aug data: {train_r2:.6f}")
        print(f"Avg grad norm: {avg_grad_norm:.4f} | LR: {current_lr:.2e}")
        print(
            f"Best so far: epoch {best_epoch}, MSE={best_mse:.6e}, R²={best_r2:.6f}, "
            f"{'Improved' if improved else 'No improve'} | "
            f"No improve count: {epochs_no_improve}/{args.early_stop_patience}"
        )
        print(f"Runtime so far: {elapsed} sec\n")

        # Early stopping
        if epochs_no_improve >= args.early_stop_patience:
            print(
                f"\n⚠️ Early stopping at epoch {epoch}. "
                f"Restoring best model from epoch {best_epoch} (R²={best_r2:.6f})."
            )
            break

        # Very high R² => can stop early
        if best_r2 > 0.999 and epoch >= 10:
            print(f"\n✓ Early exit: reached high R²={best_r2:.6f} at epoch {best_epoch}")
            break

    # ---------------------------------------------------------
    # Restore best model
    # ---------------------------------------------------------
    if best_state is not None:
        mps.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        print(
            f"\nRestored best MPS+MLP from epoch {best_epoch} "
            f"(MSE={best_mse:.6e}, R²={best_r2:.6f})"
        )
    else:
        print("\n⚠️ No best_state recorded; using last epoch parameters.")

    # ---------------------------------------------------------
    # 6) Save trained compressed MPS
    # ---------------------------------------------------------
    out_path = f"{args.prefix}_mps_compress.pt"
    torch.save(
        {
            "state_dict": mps.state_dict(),
            "input_dim": D,
            "bond_dim": args.bond_dim,
            "feature_dim": args.feature_dim,
            "hidden_dim": args.hidden_dim,
            "max_degree": args.max_degree,
            "compress_targets": compress_targets_path,
        },
        out_path,
    )
    print(f"\nSaved compressed MPS+MLP surrogate to {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
