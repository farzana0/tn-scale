#!/usr/bin/env python3
"""
train_mps_sqexp_paths.py

Train an MPS on path-augmented data for TN-SHAP, for the *exponential* teacher.

Teacher paths (in ORIGINAL x-space):
  - h(t)        = f(t * x)
  - g_on_i(t)   = f(x_on_i(t))   where x_on_i(t)[j]  = x[j]    if j == i, else t * x[j]
  - g_off_i(t)  = f(x_off_i(t))  where x_off_i(t)[j] = 0       if j == i, else t * x[j]

This script now constructs ALL of these points, so that the grid used in the
Gi-based TN-SHAP evaluation matches the grid used for training.

We:
  - Load sqexp teacher + data from poly_teacher.py.
  - Sample N_base = n_targets base points x_base.
  - Build Chebyshev nodes t in [0.2, 1.0].
  - Construct all path points in ORIGINAL space for those base points:
        - 1 h-path point per t
        - D "on" Gi points per t
        - D "off" Gi points per t
  - Train an MPS on these (X_all, Y_all_scaled).
  - Save everything under expo/:
        expo/<prefix>_tnshap_targets.pt
        expo/<prefix>_mps.pt
"""

import argparse
import os
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


def chebyshev_nodes_scaled(n_nodes, t_min=0.2, t_max=1.0, device=None, dtype=torch.float32):
    k = torch.arange(1, n_nodes + 1, dtype=torch.float64, device=device)
    u = torch.cos((2.0 * k - 1.0) / (2.0 * n_nodes) * torch.pi)
    u01 = (u + 1.0) / 2.0
    t = t_min + (t_max - t_min) * u01
    return t.to(dtype).to(device)


# (Currently unused, kept for future feature-map experiments)
class ScalarMLPFeatureMap(nn.Module):
    """
    Feature map: scalar -> MLP -> R^1

    Architecture:
      Linear(1, 32) -> ReLU -> Linear(32, 1)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (...,)
        x_in = x.unsqueeze(-1)           # (..., 1)
        out = self.net(x_in)            # (..., 1)
        return out                      # (..., 1)


# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix", type=str, default="sqexp_D50",
        help="Prefix to load teacher/data and to name outputs (e.g. 'sqexp_D50')."
    )
    parser.add_argument(
        "--max-degree", type=int, default=10,
        help="Polynomial degree in t for TN-SHAP interpolation."
    )
    parser.add_argument(
        "--n-targets", type=int, default=20,
        help="Number of base points whose paths we include (M for Shapley eval)."
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--bond-dim", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2-reg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)

    # Scheduler & early stopping
    parser.add_argument("--lr-factor", type=float, default=0.8,
                        help="Factor for ReduceLROnPlateau / StepLR.")
    parser.add_argument("--lr-patience", type=int, default=5,
                        help="(Unused now; kept for compatibility).")
    parser.add_argument("--min-lr", type=float, default=1e-7,
                        help="Minimum learning rate (not enforced by StepLR).")
    parser.add_argument("--early-stop-patience", type=int, default=30,
                        help="Epochs without MSE improvement before early stop.")

    # Gradient clipping
    parser.add_argument("--grad-clip", type=float, default=5.0,
                        help="Max gradient norm for clipping.")

    # Divergence detection
    parser.add_argument("--divergence-factor", type=float, default=50.0,
                        help="If MSE exceeds this factor * best_MSE, treat as divergence.")
    
  
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    expo_dir = "expo"
    os.makedirs(expo_dir, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Prefix: {args.prefix}, max_degree={args.max_degree}, N_targets={args.n_targets}")
    print(f"LR={args.lr}, LR factor={args.lr_factor}, "
          f"min_lr={args.min_lr}, early_stop_patience={args.early_stop_patience}")
    print(f"Grad clip: {args.grad_clip}, bond_dim={args.bond_dim}, l2_reg={args.l2_reg}")

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
    perm = torch.randperm(N_total, device=DEVICE)
    idxs = perm[:N_base]
    x_base = x_train[idxs]  # (N_base, D)

    print(f"Using N_base={N_base} base points out of {N_total}, D={D}")

    # ---------------------------------------------------------
    # 2) Build Chebyshev t in [0.2, 1.0] for path augmentation
    # ---------------------------------------------------------
    # We need max_degree+1 distinct nodes for Vandermonde interpolation.
    N_T_NODES = args.max_degree + 1
    t_nodes = chebyshev_nodes_scaled(
        n_nodes=N_T_NODES,
        t_min=0.2,
        t_max=1.0,
        device=DEVICE,
        dtype=x_base.dtype,
    )
    print(f"Using N_T_NODES={N_T_NODES} Chebyshev nodes in t for path augmentation.")

    # ---------------------------------------------------------
    # 3) Construct path-augmented dataset:
    #    for each x0, each t:
    #       - 1 h-path       : t * x0
    #       - D g_on_i paths : x_on_i(t)
    #       - D g_off_i paths: x_off_i(t)
    # ---------------------------------------------------------
    total_samples = (1 + 2 * D) * N_base * N_T_NODES
    X_all = torch.zeros(total_samples, D, device=DEVICE, dtype=x_base.dtype)
    Y_all = torch.zeros(total_samples, device=DEVICE, dtype=x_base.dtype)

    teacher.eval()
    idx = 0

    print("Generating path-augmented dataset (h + g_on_i + g_off_i paths)...")
    with torch.no_grad():
        for b in range(N_base):
            x0 = x_base[b]  # (D,)

            for t in t_nodes:
                # Base scaled vector for this t
                base = t * x0  # (D,)

                # -------------------
                # h-path: x_h(t) = t * x0
                # -------------------
                x_h = base
                y_h = teacher(x_h.unsqueeze(0)).squeeze(0)
                if y_h.ndim > 0:
                    y_h = y_h.squeeze(-1)
                X_all[idx] = x_h
                Y_all[idx] = y_h
                idx += 1

                # -------------------
                # g_on_i paths (vectorized)
                #   x_on_i(t)[j] = x[j] if j == i else t * x[j]
                # -------------------
                x_on_batch = base.unsqueeze(0).expand(D, -1).clone()  # (D, D)
                x_on_batch[torch.arange(D), torch.arange(D)] = x0     # clamp i to original x_i
                y_on_batch = teacher(x_on_batch)
                if y_on_batch.ndim > 1:
                    y_on_batch = y_on_batch.squeeze(-1)

                X_all[idx:idx + D] = x_on_batch
                Y_all[idx:idx + D] = y_on_batch
                idx += D

                # -------------------
                # g_off_i paths (vectorized)
                #   x_off_i(t)[j] = 0 if j == i else t * x[j]
                # -------------------
                x_off_batch = base.unsqueeze(0).expand(D, -1).clone()  # (D, D)
                x_off_batch[torch.arange(D), torch.arange(D)] = 0.0    # clamp i to baseline 0
                y_off_batch = teacher(x_off_batch)
                if y_off_batch.ndim > 1:
                    y_off_batch = y_off_batch.squeeze(-1)

                X_all[idx:idx + D] = x_off_batch
                Y_all[idx:idx + D] = y_off_batch
                idx += D

    assert idx == total_samples, "Index mismatch when building path-augmented dataset."

    print(
        f"Path-augmented dataset size (h + g_on_i + g_off_i): "
        f"{X_all.shape[0]} points = (1 + 2D) * N_base * N_T_NODES"
    )

    # ---------------------------------------------------------
    # 3.5) Scale outputs to avoid exploding magnitudes
    # ---------------------------------------------------------
    with torch.no_grad():
        Y_abs_max = torch.max(Y_all.abs())
        if Y_abs_max <= 0:
            Y_scale = torch.tensor(1.0, device=DEVICE, dtype=Y_all.dtype)
        else:
            Y_scale = Y_abs_max

    print(f"Scaling path-augmented targets by Y_scale = {Y_scale.item():.3e}")
    Y_all_scaled = Y_all / Y_scale

    # ---------------------------------------------------------
    # 4) Train MPS on ORIGINAL X_all with sqexp feature map
    # ---------------------------------------------------------
    train_ds = TensorDataset(X_all, Y_all_scaled)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, X_all.shape[0]),  # full-batch if batch_size > N
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    # Use simple identity-style feature map inside MPS (default [x, x^2])
    mps = MPS(
        input_dim=D,
        output_dim=1,
        bond_dim=args.bond_dim,
        adaptive_mode=False,
        periodic_bc=False,
        feature_dim=2,        # default embedding: [x, x^2]
        parallel_eval=True,
    )

    mps.to(DEVICE)

    loss_fun = nn.MSELoss()
    optimizer = torch.optim.Adam(
        mps.parameters(), lr=args.lr, weight_decay=args.l2_reg
    )

    # StepLR: reduce LR every 50 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=50,
        gamma=args.lr_factor,
        verbose=True,
    )

    max_grad_norm = args.grad_clip
    print(
        f"\nTraining MPS on sqexp path-augmented regression\n"
        f"D={D} (original space), bond_dim={args.bond_dim}, "
        f"N_train={X_all.shape[0]} (no held-out test set)\n"
    )

    train_start = time.time()
    best_mse = float("inf")
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

        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0

        # Track best model based on MSE
        improved = False
        if train_mse < best_mse - 1e-7:
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
                  f"(MSE={best_mse:.5f})")
            break

        elapsed = int(time.time() - train_start)
        print(f"### Epoch {epoch:03d} ###")
        print(f"Train MSE: {train_mse:.5f}")
        print(f"Avg grad norm: {avg_grad_norm:.4f} | LR: {current_lr:.2e}")
        print(
            f"Best so far: epoch {best_epoch}, MSE={best_mse:.5f}, "
            f"{'✓ Improved' if improved else 'No improve'} | "
            f"No improve count: {epochs_no_improve}/{args.early_stop_patience}"
        )
        print(f"Runtime so far: {elapsed} sec\n")

        if epochs_no_improve >= args.early_stop_patience:
            print(
                f"\n⚠️ Early stopping at epoch {epoch}. "
                f"Restoring best model from epoch {best_epoch} (MSE={best_mse:.5f})."
            )
            break

        if best_mse < 1e-10 and epoch >= 50:
            print(f"\n✓ Early exit: reached very low MSE={best_mse:.10f} at epoch {best_epoch}")
            break

    # ---------------------------------------------------------
    # Restore best model if we have it
    # ---------------------------------------------------------
    if best_state is not None:
        mps.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        print(f"\nRestored best MPS from epoch {best_epoch} (MSE={best_mse:.5f})")
    else:
        print("\n⚠️ No best_state recorded; using last epoch parameters.")

    # ---------------------------------------------------------
    # Compute final R² on full training set
    # ---------------------------------------------------------
    print("\nComputing final R² on training set...")
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
        final_r2 = r2_score(Y_all_eval, preds_all)

    print(f"Final Train R² (on path-augmented data): {final_r2:.6f}")

    # ---------------------------------------------------------
    # 5) Save TN-SHAP targets and full training dataset (under expo/)
    # ---------------------------------------------------------
    tnshap_path = os.path.join(expo_dir, f"{args.prefix}_tnshap_targets.pt")
    torch.save(
        {
            "x_base": x_base.detach().cpu(),    # (N_base, D)
            "t_nodes": t_nodes.detach().cpu(),  # (N_T_NODES,)
            "X_all": X_all.detach().cpu(),      # ((1+2D)*N_base*N_T_NODES, D)
            "Y_all": Y_all.detach().cpu(),      # ORIGINAL teacher values
            "Y_scale": Y_scale.detach().cpu(),  # global scale used for training
            "max_degree": args.max_degree,
            "feature_map": "sqexp_x2exp_then_one",
        },
        tnshap_path,
    )

    print(f"Saved TN-SHAP targets and t-nodes to {tnshap_path}")

    # ---------------------------------------------------------
    # 6) Save trained MPS (under expo/)
    # ---------------------------------------------------------
    mps_path = os.path.join(expo_dir, f"{args.prefix}_mps.pt")
    torch.save(
        {
            "state_dict": mps.state_dict(),
            "input_dim": D,
            "bond_dim": args.bond_dim,
            "feature_map": "sqexp_x2exp_then_one",
        },
        mps_path,
    )
    print(f"Saved trained MPS to {mps_path}")
    print("Done.")


if __name__ == "__main__":
    main()
