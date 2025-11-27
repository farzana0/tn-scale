#!/usr/bin/env python3
"""
train_mps_sqexp_mlp_paths.py

Train an MPS on the *existing* path-augmented dataset for the sqexp teacher,
using a *learnable MLP feature map* per site:

    phi_MLP(s_i) ∈ R^{feature_dim}

The training data is taken directly from:

    <PREFIX>_tnshap_targets.pt

produced by the sqexp teacher script, i.e.:

    - X_all: path-augmented ORIGINAL points (h and g_i paths)
    - Y_all: teacher(X_all)

We:

  - Load X_all, Y_all (no extra interpolation in feature space).
  - Define an MPS with a trainable nn.Module feature map.
  - Train to maximize R² on the entire path-augmented dataset.
  - Save the trained MPS+MLP to <PREFIX>_mps_mlp.pt

Later you can plug this MPS into a TN-SHAP eval script that uses ORIGINAL
t*x paths, with the MLP feature map inside the MPS (no t-interpolation in φ).
"""

import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from torchmps import MPS
from poly_teacher import DEVICE


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


# Feature map removed - using raw input directly


# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix", type=str, default="sqexp_D50",
        help="Prefix used for *_tnshap_targets.pt (e.g. 'sqexp_D50')."
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-epochs", type=int, default=300)
    parser.add_argument("--bond-dim", type=int, default=80)
    parser.add_argument("--feature-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)  # Increased from 1e-3 to 5e-3
    parser.add_argument("--l2-reg", type=float, default=0.0)

    # Scheduler & early stopping
    parser.add_argument("--lr-factor", type=float, default=0.8,
                        help="Factor for ReduceLROnPlateau.")
    parser.add_argument("--lr-patience", type=int, default=5,
                        help="Patience (epochs) before LR reduction.")
    parser.add_argument("--min-lr", type=float, default=1e-7,
                        help="Minimum learning rate.")
    parser.add_argument("--early-stop-patience", type=int, default=40,
                        help="Epochs without R² improvement before early stop.")

    parser.add_argument("--grad-clip", type=float, default=5.0,
                        help="Max gradient norm for clipping.")
    parser.add_argument("--divergence-factor", type=float, default=50.0,
                        help="If MSE exceeds this factor * best_MSE, stop.")
    parser.add_argument("--val-frac", type=float, default=0.0,
                        help="Optional fraction of data to hold out as validation "
                             "(0.0 => use all as train). R² is always measured on "
                             "the full dataset for now.")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Prefix: {args.prefix}")
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
        f"divergence_factor={args.divergence_factor}, val_frac={args.val_frac}"
    )

    # ---------------------------------------------------------
    # 1) Load path-augmented dataset from *_tnshap_targets.pt
    # ---------------------------------------------------------
    targets_path = f"{args.prefix}_tnshap_targets.pt"
    print(f"\nLoading path-augmented data from {targets_path}")
    targets = torch.load(targets_path, map_location=DEVICE)
    X_all = targets["X_all"].to(DEVICE)  # ((1+D)*N_base*N_T_NODES, D)
    Y_all = targets["Y_all"].to(DEVICE)  # same length
    max_degree = int(targets["max_degree"])
    feature_map_name = targets.get("feature_map", "local_x_then_one")

    N_all, D = X_all.shape
    print(f"Loaded X_all with shape {X_all.shape}, Y_all with shape {Y_all.shape}")
    print(f"max_degree (for TN-SHAP later) = {max_degree}")
    print(f"Original targets feature_map tag: {feature_map_name}")

    # Optional val split (default 0 => use everything as train)
    if args.val_frac > 0.0:
        N_val = int(args.val_frac * N_all)
        N_train = N_all - N_val
        X_train, X_val = X_all[:N_train], X_all[N_train:]
        Y_train, Y_val = Y_all[:N_train], Y_all[N_train:]
        print(f"Using N_train={N_train}, N_val={N_val}")
    else:
        X_train, Y_train = X_all, Y_all
        X_val, Y_val = None, None
        N_train = N_all
        print(f"Using all N_train={N_train} points (no explicit validation set)")

    # ---------------------------------------------------------
    # 2) DataLoaders
    # ---------------------------------------------------------
    train_ds = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, N_train),
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    # For R² eval / full-batch evaluation
    full_ds = TensorDataset(X_all, Y_all)
    full_loader = DataLoader(
        full_ds,
        batch_size=min(args.batch_size, N_all),
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    # ---------------------------------------------------------
    # 3) Build MPS (no feature map - using raw input)
    # ---------------------------------------------------------
    print(
        f"\nBuilding MPS (no feature map): "
        f"D={D}, bond_dim={args.bond_dim}"
    )

    # Build MPS without feature map (simplest and fastest!)
    mps = MPS(
        input_dim=D,
        output_dim=1,
        bond_dim=args.bond_dim,
        adaptive_mode=False,
        periodic_bc=False,
        parallel_eval=True,
    ).to(DEVICE)

    loss_fun = nn.MSELoss()
    optimizer = torch.optim.Adam(
        mps.parameters(), lr=args.lr, weight_decay=args.l2_reg
    )
    
    # Simple ReduceLROnPlateau scheduler (no warmup, no cosine annealing)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",   # maximize R² on full dataset
        factor=args.lr_factor,
        patience=args.lr_patience,
        verbose=False,
        min_lr=args.min_lr,
        threshold=1e-4,
    )
    
    # Use automatic mixed precision for faster training
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == 'cuda' else None
    use_amp = DEVICE.type == 'cuda'

    max_grad_norm = args.grad_clip

    print(
        f"\nTraining MPS (raw input, no feature map) on sqexp path-augmented regression\n"
        f"N_train={N_train}, N_all={N_all}, D={D}\n"
        f"Using AMP: {use_amp} (much faster!)\n"
        f"LR Schedule: ReduceLROnPlateau (simple and stable)\n"
    )

    train_start = time.time()
    best_mse = float("inf")
    best_r2 = -float("inf")
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    # ---------------------------------------------------------
    # 4) Training loop
    # ---------------------------------------------------------
    for epoch in range(1, args.num_epochs + 1):
        epoch_start = time.time()
        mps.train()
        train_loss = 0.0
        n_seen = 0
        grad_norms = []

        for batch_idx, (xb, yb) in enumerate(train_loader):
            batch_start = time.time()
            
            optimizer.zero_grad()
            
            # Use automatic mixed precision if available
            if use_amp:
                with torch.cuda.amp.autocast():
                    preds = mps(xb).squeeze(-1)
                    loss = loss_fun(preds, yb)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(mps.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = mps(xb).squeeze(-1)
                loss = loss_fun(preds, yb)
                loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(mps.parameters(), max_grad_norm)
                optimizer.step()
            
            grad_norms.append(total_norm.item())

            train_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)
            
            # Print progress every 10 batches for first 3 epochs
            if epoch <= 3 and (batch_idx + 1) % 10 == 0:
                batch_time = time.time() - batch_start
                avg_loss = train_loss / n_seen
                print(f"  Epoch {epoch}, Batch {batch_idx+1}/{len(train_loader)}: "
                      f"avg_loss={avg_loss:.6e}, batch_time={batch_time:.2f}s")

        train_mse = train_loss / max(n_seen, 1)

        # Evaluate R² on full path-aug dataset
        mps.eval()
        with torch.no_grad():
            y_true_list = []
            y_pred_list = []
            for xb_eval, yb_eval in full_loader:
                p_eval = mps(xb_eval).squeeze(-1)
                y_true_list.append(yb_eval)
                y_pred_list.append(p_eval)

            Y_all_eval = torch.cat(y_true_list, dim=0)
            preds_all = torch.cat(y_pred_list, dim=0)
            full_r2 = r2_score(Y_all_eval, preds_all)

        # Optional: simple val MSE
        if X_val is not None:
            with torch.no_grad():
                p_val = mps(X_val).squeeze(-1)
                val_mse = loss_fun(p_val, Y_val).item()
        else:
            val_mse = None

        # Step scheduler
        scheduler.step(full_r2)
        
        current_lr = optimizer.param_groups[0]["lr"]
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0

        # Track best model & divergence
        improved = False
        if full_r2 > best_r2 + 1e-5:
            best_r2 = full_r2
            best_mse = train_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in mps.state_dict().items()}
            improved = True
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Only stop on non-finite loss now (remove exponential growth check)
        if not torch.isfinite(torch.tensor(train_mse, device=DEVICE)):
            print("\n⚠️  Detected non-finite loss (NaN/Inf).")
            print(
                f"Stopping at epoch {epoch}. Restoring best model from epoch {best_epoch} "
                f"(MSE={best_mse:.5f}, R²={best_r2:.6f})"
            )
            break

        epoch_time = time.time() - epoch_start
        elapsed = int(time.time() - train_start)
        print(f"\n### Epoch {epoch:03d} (took {epoch_time:.1f}s) ###")
        print(f"Train MSE: {train_mse:.6e}")
        if val_mse is not None:
            print(f"Val MSE:   {val_mse:.6e}")
        print(f"R² on full path-aug data: {full_r2:.6f}")
        print(f"Avg grad norm: {avg_grad_norm:.4f} | LR: {current_lr:.2e}")
        print(
            f"Best so far: epoch {best_epoch}, MSE={best_mse:.6e}, R²={best_r2:.6f}, "
            f"{'Improved' if improved else 'No improve'} | "
            f"No improve count: {epochs_no_improve}/{args.early_stop_patience}"
        )
        print(f"Runtime so far: {elapsed} sec\n")

        # Early stopping if no improvement
        if epochs_no_improve >= args.early_stop_patience:
            print(
                f"\n⚠️ Early stopping at epoch {epoch}. "
                f"Restoring best model from epoch {best_epoch} (R²={best_r2:.6f})."
            )
            break

        # If we reach high R², exit early (lower threshold for faster convergence)
        if best_r2 > 0.995 and epoch >= 10:
            print(f"\n✓ Early exit: reached high R²={best_r2:.6f} at epoch {best_epoch}")
            break

    # ---------------------------------------------------------
    # Restore best model if we have it
    # ---------------------------------------------------------
    if best_state is not None:
        mps.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        print(
            f"\nRestored best MPS (raw input) from epoch {best_epoch} "
            f"(MSE={best_mse:.6e}, R²={best_r2:.6f})"
        )
    else:
        print("\n⚠️ No best_state recorded; using last epoch parameters.")

    # ---------------------------------------------------------
    # 5) Save trained MPS (no feature map)
    # ---------------------------------------------------------
    out_path = f"{args.prefix}_mps_raw.pt"
    torch.save(
        {
            "state_dict": mps.state_dict(),
            "input_dim": D,
            "bond_dim": args.bond_dim,
            "feature_map": "none",  # No feature map
            "max_degree": max_degree,
            "path_aug_source": targets_path,
        },
        out_path,
    )
    print(f"\nSaved trained MPS (raw input) to {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
