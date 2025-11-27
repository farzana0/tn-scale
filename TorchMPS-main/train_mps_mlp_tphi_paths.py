#!/usr/bin/env python3
"""
train_mps_mlp_tphi_paths.py

Train an MLP+MPS surrogate on path-augmented data for TN-SHAP, where
the surrogate lives in FEATURE space with paths:

    h-path:   z_h(t)   = t * φ(x)
    g_i-path: z_g^i(t)[j] = φ_j(x_j) if j == i else t * φ_j(x_j)

The teacher is evaluated in ORIGINAL x-space:

    h-path:   x_h(t)   = t * x
    g_i-path: x_g^i(t)[j] = x[j] if j == i else t * x[j]

We store:
  - For teacher:
        y = f(x_path)
  - For surrogate:
        base x, t, is_h, i, so we can build t*φ(x) later.

Feature map:
  - ScalarMLPFeatureMap: R -> R^3 (per coordinate)
  - Then we append 1, so per coordinate we have 4 dims.
  - Final surrogate input dim = D_aug = 4 * D.

Saves:
  - <PREFIX>_tnshap_targets.pt:
        x_base, t_nodes, X_all_path, Y_all, X_base_all, t_all, is_h_all, i_all,
        max_degree, feature_map name, mlp_hidden
  - <PREFIX>_mps_mlp_tphi.pt:
        mps_state_dict, phi_state_dict, D_aug, bond_dim, mlp_hidden, feature_map
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
    Chebyshev nodes of the first kind mapped from [-1,1] to [0,1].

    u_k = cos((2k - 1)/(2n) * pi), k=1..n
    t   = (u + 1)/2
    """
    if device is None:
        device = DEVICE
    k = torch.arange(1, n_nodes + 1, dtype=torch.float64, device=device)
    u = torch.cos((2.0 * k - 1.0) / (2.0 * n_nodes) * torch.pi)  # [-1,1]
    t = (u + 1.0) / 2.0
    return t.to(dtype).to(device)


# -----------------------
# Scalar MLP feature map  R -> R^3
# -----------------------

class ScalarMLPFeatureMap(nn.Module):
    """
    Per-coordinate feature map with two hidden layers.

    Input:   (B, D) tensor of scalars.
    Output:  (B, D, 3) features for each coordinate.
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
        x: (B, D) -> (B, D, 3)
        """
        B, D = x.shape
        # Use reshape instead of view to avoid non-contiguous errors
        x_flat = x.reshape(B * D, 1)
        h_flat = self.net(x_flat)         # (B*D, 3)
        h = h_flat.reshape(B, D, 3)       # (B, D, 3)
        return h


# -----------------------
# Build t * φ(x) features with g_i clamping
# -----------------------

def build_phi_t_features_base(
    x_base: torch.Tensor,
    t: torch.Tensor,
    is_h: torch.Tensor,
    i_idx: torch.Tensor,
    phi_module: ScalarMLPFeatureMap,
) -> torch.Tensor:
    """
    Build z = t * φ(x_base) in FEATURE space, with:

      h-path:   all coords scaled by t
      g_i-path: coord i NOT scaled, others scaled by t

    Args:
      x_base: (B, D)   base inputs x (NOT masked; original x)
      t:      (B,)     scalar t in [0,1] used along the path
      is_h:   (B,)     bool; True for h-path, False for g_i-path
      i_idx:  (B,)     long; index i for g_i, -1 for h
      phi_module: ScalarMLPFeatureMap

    Returns:
      Z: (B, D_aug) where D_aug = 4 * D
    """
    B, D = x_base.shape
    device = x_base.device

    # φ(x): (B, D, 3)
    phi_signal = phi_module(x_base)

    # Append constant 1 per coordinate -> (B, D, 4)
    ones = torch.ones(B, D, 1, device=device, dtype=x_base.dtype)
    phi_full = torch.cat([phi_signal, ones], dim=-1)  # (B, D, 4)

    # Build scaling mask for first 3 dims
    base_scale = t.view(B, 1, 1).expand(B, D, 1).clone()  # (B, D, 1)

    # For g_i paths: coord i should NOT be scaled by t, i.e. scale=1 there
    gi_mask = (~is_h).nonzero(as_tuple=False).view(-1)
    if gi_mask.numel() > 0:
        b_idx = gi_mask
        i_vals = i_idx[b_idx]  # (N_gi,)
        base_scale[b_idx, i_vals, 0] = 1.0

    # Apply scaling only on first 3 features
    phi_full[:, :, :3] = phi_full[:, :, :3] * base_scale

    # Flatten
    Z = phi_full.reshape(B, D * 4)
    return Z


# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix", type=str, default="sqexp_D50",
        help="Prefix used to load teacher/data and save MPS/targets."
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
    parser.add_argument("--lr-factor", type=float, default=0.8)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--early-stop-patience", type=int, default=30)

    # Gradient clipping
    parser.add_argument("--grad-clip", type=float, default=5.0)

    # Divergence guard (now only non-finite check)
    parser.add_argument("--divergence-factor", type=float, default=50.0)

    # MLP size
    parser.add_argument("--mlp-hidden", type=int, default=32)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Prefix={args.prefix}, max_degree={args.max_degree}, N_targets={args.n_targets}")
    print(f"bond_dim={args.bond_dim}, mlp_hidden={args.mlp_hidden}, "
          f"LR={args.lr}, l2={args.l2_reg}")
    print(f"Scheduler: factor={args.lr_factor}, patience={args.lr_patience}, "
          f"min_lr={args.min_lr}, early_stop={args.early_stop_patience}")
    print(f"Grad clip={args.grad_clip}\n")

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
    x_base_points = x_train[:N_base]  # (N_base, D)

    print(f"Using N_base={N_base} base points out of {N_total}, D={D}")

    # ---------------------------------------------------------
    # 2) Build Chebyshev t in [0,1]
    # ---------------------------------------------------------
    N_T_NODES = args.max_degree + 1
    t_nodes = chebyshev_nodes_unit_interval(
        N_T_NODES, device=DEVICE, dtype=x_train.dtype
    )
    print(f"Using N_T_NODES={N_T_NODES} Chebyshev nodes in t.\n")

    # ---------------------------------------------------------
    # 3) Construct path-augmented teacher dataset
    # ---------------------------------------------------------
    total_samples = (1 + D) * N_base * N_T_NODES

    # Teacher paths in ORIGINAL space
    X_all_path = torch.zeros(total_samples, D, device=DEVICE, dtype=x_train.dtype)
    Y_all = torch.zeros(total_samples, device=DEVICE, dtype=x_train.dtype)

    # Metadata for surrogate in FEATURE space (base x & path info)
    X_base_all = torch.zeros(total_samples, D, device=DEVICE, dtype=x_train.dtype)
    t_all = torch.zeros(total_samples, device=DEVICE, dtype=x_train.dtype)
    is_h_all = torch.zeros(total_samples, device=DEVICE, dtype=torch.bool)
    i_all = torch.full((total_samples,), -1, device=DEVICE, dtype=torch.long)

    teacher.eval()
    idx = 0
    print("Generating path-augmented dataset (teacher + metadata)...")
    with torch.no_grad():
        for b in range(N_base):
            x0 = x_base_points[b]  # (D,)

            for t_val in t_nodes:
                # h-path in ORIGINAL space
                x_h = t_val * x0
                y_h = teacher(x_h.unsqueeze(0)).squeeze(0)
                if y_h.ndim > 0:
                    y_h = y_h.squeeze(-1)

                X_all_path[idx] = x_h
                Y_all[idx] = y_h
                X_base_all[idx] = x0
                t_all[idx] = t_val
                is_h_all[idx] = True
                i_all[idx] = -1
                idx += 1

                # g_i paths in ORIGINAL space (vectorized)
                x_g_batch = t_val * x0.unsqueeze(0).expand(D, -1).clone()  # (D, D)
                x_g_batch[torch.arange(D), torch.arange(D)] = x0

                y_g_batch = teacher(x_g_batch)
                if y_g_batch.ndim > 1:
                    y_g_batch = y_g_batch.squeeze(-1)  # (D,)

                # store each g_i sample
                X_all_path[idx:idx + D] = x_g_batch
                Y_all[idx:idx + D] = y_g_batch

                X_base_all[idx:idx + D] = x0.unsqueeze(0).expand(D, -1)
                t_all[idx:idx + D] = t_val
                is_h_all[idx:idx + D] = False
                i_all[idx:idx + D] = torch.arange(D, device=DEVICE, dtype=torch.long)

                idx += D

    assert idx == total_samples, f"idx={idx} but total_samples={total_samples}"
    print(
        f"Path-augmented dataset size: {X_all_path.shape[0]} "
        f"= (1 + D)*N_base*N_T_NODES\n"
    )

    # ---------------------------------------------------------
    # 4) Train MLP+MPS on t * φ(x_base)
    # ---------------------------------------------------------
    D_aug = D * 4  # 3 features + 1 constant per coord
    train_ds = TensorDataset(X_base_all, Y_all, t_all, is_h_all, i_all)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, X_base_all.shape[0]),
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    phi_module = ScalarMLPFeatureMap(hidden_dim=args.mlp_hidden).to(DEVICE)

    mps = MPS(
        input_dim=D_aug,
        output_dim=1,
        bond_dim=args.bond_dim,
        adaptive_mode=False,
        periodic_bc=False,
        parallel_eval=True,
    ).to(DEVICE)

    loss_fun = nn.MSELoss()
    optimizer = torch.optim.Adam(
        list(mps.parameters()) + list(phi_module.parameters()),
        lr=args.lr,
        weight_decay=args.l2_reg,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
        verbose=True,
        min_lr=args.min_lr,
        threshold=1e-4,
    )

    max_grad_norm = args.grad_clip
    print(
        f"Training MLP+MPS(tφ(x)) surrogate\n"
        f"D={D}, D_aug={D_aug}, bond_dim={args.bond_dim}, "
        f"N_train={X_base_all.shape[0]}\n"
    )

    train_start = time.time()
    best_mse = float("inf")
    best_r2 = -float("inf")
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.num_epochs + 1):
        mps.train()
        phi_module.train()
        train_loss = 0.0
        n_seen = 0
        grad_norms = []

        for xb_base, yb, tb, is_hb, ib in train_loader:
            xb_base = xb_base.to(DEVICE)
            yb = yb.to(DEVICE)
            tb = tb.to(DEVICE)
            is_hb = is_hb.to(DEVICE)
            ib = ib.to(DEVICE)

            Zb = build_phi_t_features_base(xb_base, tb, is_hb, ib, phi_module)
            preds = mps(Zb).squeeze(-1)
            loss = loss_fun(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(
                list(mps.parameters()) + list(phi_module.parameters()),
                max_grad_norm
            )
            grad_norms.append(total_norm.item())
            optimizer.step()

            train_loss += loss.item() * xb_base.size(0)
            n_seen += xb_base.size(0)

        train_mse = train_loss / max(n_seen, 1)

        # Evaluate R² on whole training set (batched)
        mps.eval()
        phi_module.eval()
        with torch.no_grad():
            y_true_list = []
            y_pred_list = []
            for xb_base_eval, yb_eval, tb_eval, is_h_eval, ib_eval in train_loader:
                xb_base_eval = xb_base_eval.to(DEVICE)
                yb_eval = yb_eval.to(DEVICE)
                tb_eval = tb_eval.to(DEVICE)
                is_h_eval = is_h_eval.to(DEVICE)
                ib_eval = ib_eval.to(DEVICE)

                Z_eval = build_phi_t_features_base(
                    xb_base_eval, tb_eval, is_h_eval, ib_eval, phi_module
                )
                p_eval = mps(Z_eval).squeeze(-1)
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
            best_state = {
                "mps": {k: v.detach().cpu().clone() for k, v in mps.state_dict().items()},
                "phi": {k: v.detach().cpu().clone() for k, v in phi_module.state_dict().items()},
            }
            improved = True
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Only stop on non-finite loss now
        if not torch.isfinite(torch.tensor(train_mse, device=DEVICE)):
            print("\n⚠️  Detected non-finite loss (NaN/Inf).")
            print(f"Stopping at epoch {epoch}. Restoring best model from epoch {best_epoch} "
                  f"(MSE={best_mse:.5f}, R²={best_r2:.6f})")
            break

        elapsed = int(time.time() - train_start)
        print(f"### Epoch {epoch:03d} ###")
        print(f"Train MSE: {train_mse:.5e}")
        print(f"Train R2 (path data): {train_r2:.6f}")
        print(f"Avg grad norm: {avg_grad_norm:.4f} | LR: {current_lr:.2e}")
        print(
            f"Best so far: epoch {best_epoch}, MSE={best_mse:.5e}, R²={best_r2:.6f}, "
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

        if best_r2 > 0.995 and epoch >= 10:
            print(f"\n✓ Early exit: reached high R²={best_r2:.6f} at epoch {best_epoch}")
            break

    # ---------------------------------------------------------
    # Restore best model
    # ---------------------------------------------------------
    if best_state is not None:
        mps.load_state_dict({k: v.to(DEVICE) for k, v in best_state["mps"].items()})
        phi_module.load_state_dict({k: v.to(DEVICE) for k, v in best_state["phi"].items()})
        print(f"\nRestored best MPS+MLP from epoch {best_epoch} "
              f"(MSE={best_mse:.5e}, R²={best_r2:.6f})")
    else:
        print("\n⚠️ No best_state recorded; using last epoch parameters.")

    # ---------------------------------------------------------
    # 5) Save TN-SHAP targets + metadata
    # ---------------------------------------------------------
    torch.save(
        {
            "x_base": x_base_points.detach().cpu(),    # (N_base, D)
            "t_nodes": t_nodes.detach().cpu(),         # (N_T_NODES,)
            "X_all_path": X_all_path.detach().cpu(),   # (N_all, D) original path points
            "Y_all": Y_all.detach().cpu(),             # (N_all,)
            "X_base_all": X_base_all.detach().cpu(),   # (N_all, D) base x per sample
            "t_all": t_all.detach().cpu(),             # (N_all,)
            "is_h_all": is_h_all.detach().cpu(),       # (N_all,)
            "i_all": i_all.detach().cpu(),             # (N_all,)
            "max_degree": args.max_degree,
            "feature_map": "scalar_mlp3_plus_one_tphi",
            "mlp_hidden": args.mlp_hidden,
        },
        f"{args.prefix}_tnshap_targets.pt",
    )
    print(f"Saved TN-SHAP targets to {args.prefix}_tnshap_targets.pt")

    # ---------------------------------------------------------
    # 6) Save trained MPS+MLP
    # ---------------------------------------------------------
    torch.save(
        {
            "mps_state_dict": mps.state_dict(),
            "phi_state_dict": phi_module.state_dict(),
            "D_aug": D_aug,
            "bond_dim": args.bond_dim,
            "mlp_hidden": args.mlp_hidden,
            "feature_map": "scalar_mlp3_plus_one_tphi",
        },
        f"{args.prefix}_mps_mlp_tphi.pt",
    )
    print(f"Saved trained MPS+MLP(tφ) to {args.prefix}_mps_mlp_tphi.pt")
    print("Done.")


if __name__ == "__main__":
    main()
