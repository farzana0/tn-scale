#!/usr/bin/env python3
"""
train_dt_poly5_tnshap.py

Train a Decision Tree model on poly5 teacher data and compute TN-SHAP Shapley values.

This script:
  - Loads poly5 teacher and data from poly_teacher.py
  - Trains a sklearn DecisionTreeRegressor on the training data
  - Evaluates the model on test data
  - Computes TN-SHAP Shapley values for selected test points
  - Uses the same TN-SHAP methodology as tnshap_sqexp_teacher_vandermonde_eval.py

The key difference is that we use a Decision Tree model instead of an MPS model.
"""

import argparse
import os
import time
from typing import Tuple

import torch
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score as sklearn_r2_score

from poly_teacher import load_teacher, load_data, DEVICE


# -----------------------
# Utilities
# -----------------------

def chebyshev_nodes_unit_interval(n_nodes: int,
                                  device=None,
                                  dtype=torch.float32) -> torch.Tensor:
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


def build_vandermonde(t_nodes: torch.Tensor,
                      degree_max: int) -> torch.Tensor:
    """
    V[l, k] = t_nodes[l] ** k, for k = 0..degree_max and l = 0..L-1.
    """
    t = t_nodes.to(dtype=torch.float64)
    exponents = torch.arange(0, degree_max + 1, dtype=torch.float64,
                             device=t.device)
    V = t.unsqueeze(1) ** exponents.unsqueeze(0)  # (L, degree_max+1)
    return V


def solve_poly_coeffs(V: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Solve V c = y for polynomial coefficients c (square Vandermonde).

    V: (L, degree_max+1), y: (L,)
    returns c: (degree_max+1,)
    """
    V64 = V.to(dtype=torch.float64)
    y64 = y.to(dtype=torch.float64)
    c = torch.linalg.solve(V64, y64)
    return c.to(dtype=torch.float32)


def eval_h_on_nodes_dt(dt_model,
                       x: torch.Tensor,
                       t_nodes: torch.Tensor) -> torch.Tensor:
    """
    h(t) = f(t * x) for Decision Tree model, evaluated for all t_nodes.

    x: (D,)
    t_nodes: (L,)
    returns: (L,)
    """
    x_np = x.cpu().numpy()
    t_nodes_np = t_nodes.cpu().numpy()
    
    # Create batch: t * x for all t values
    X_batch_np = t_nodes_np.reshape(-1, 1) * x_np.reshape(1, -1)  # (L, D)
    
    # Predict with Decision Tree
    y_np = dt_model.predict(X_batch_np)
    
    return torch.from_numpy(y_np).to(x.device).float()


def eval_g_i_on_nodes_dt(dt_model,
                         x: torch.Tensor,
                         i: int,
                         t_nodes: torch.Tensor) -> torch.Tensor:
    """
    g_i(t) = f(x_i fixed, others scaled by t) for Decision Tree model.

    x: (D,)
    t_nodes: (L,)
    returns: (L,)
    """
    x_np = x.cpu().numpy()
    t_nodes_np = t_nodes.cpu().numpy()
    D = x.shape[0]
    
    # Create scaled batch
    X_scaled_np = t_nodes_np.reshape(-1, 1) * x_np.reshape(1, -1)  # (L, D)
    X_scaled_np[:, i] = x_np[i]  # clamp feature i to original value
    
    # Predict with Decision Tree
    y_np = dt_model.predict(X_scaled_np)
    
    return torch.from_numpy(y_np).to(x.device).float()


# -----------------------
# TN-SHAP via Vandermonde + recurrence (for Decision Tree)
# -----------------------

def tnshap_vandermonde_dt(
    dt_model,
    x: torch.Tensor,
    max_degree: int,
    t_nodes: torch.Tensor,
) -> torch.Tensor:
    """
    TN-SHAP for a Decision Tree model, with paths in ORIGINAL x-space: t * x.

    Uses:
      - h(t) = f(t x)
      - g_i(t) = f(x_i fixed, others scaled by t)
      - Chebyshev nodes t_nodes (>= max_degree+1)
      - Square Vandermonde system to recover polynomial coefficients
      - TN-SHAP recurrence to obtain β_{i,k}
      - Shapley_i = sum_k β_{i,k} / k
    """
    x = x.to(DEVICE)
    D = x.shape[0]

    assert t_nodes.shape[0] >= max_degree + 1, \
        "Need at least max_degree+1 t-nodes."
    # Use exactly max_degree+1 nodes for a square Vandermonde.
    t_nodes_use = t_nodes[: max_degree + 1]
    V = build_vandermonde(t_nodes_use, degree_max=max_degree)

    # --- h(t) ---
    h_vals = eval_h_on_nodes_dt(dt_model, x, t_nodes_use)
    alpha = solve_poly_coeffs(V, h_vals)            # (max_degree+1,)
    alpha64 = alpha.to(torch.float64)

    # Final Shapley vector
    phi_full = torch.zeros(D, device=DEVICE, dtype=torch.float32)

    # --- per-feature recurrence over g_i(t) ---
    for i in range(D):
        g_vals = eval_g_i_on_nodes_dt(dt_model, x, i, t_nodes_use)
        gamma = solve_poly_coeffs(V, g_vals)
        gamma64 = gamma.to(torch.float64)

        beta = torch.zeros(max_degree + 1,
                           dtype=torch.float64,
                           device=x.device)

        # k = max_degree
        beta[max_degree] = alpha64[max_degree] - gamma64[max_degree]

        # k = max_degree-1 ... 1
        for k in range(max_degree - 1, 0, -1):
            beta[k] = alpha64[k] + beta[k + 1] - gamma64[k]

        # consistency condition for k = 0:
        #    gamma[0] = alpha[0] + beta[1]
        # -> beta[1] = gamma[0] - alpha[0]
        beta1_from_gamma0 = gamma64[0] - alpha64[0]
        beta[1] = 0.5 * (beta[1] + beta1_from_gamma0)

        ks = torch.arange(1, max_degree + 1,
                          dtype=torch.float64,
                          device=x.device)
        phi_i = torch.sum(beta[1:] / ks)
        phi_full[i] = phi_i.to(torch.float32)

    return phi_full


# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix", type=str, default="poly5_D50",
        help="Prefix for poly5 teacher and data."
    )
    parser.add_argument(
        "--max-depth", type=int, default=10,
        help="Maximum depth of the Decision Tree."
    )
    parser.add_argument(
        "--min-samples-split", type=int, default=2,
        help="Minimum samples required to split an internal node."
    )
    parser.add_argument(
        "--min-samples-leaf", type=int, default=1,
        help="Minimum samples required to be at a leaf node."
    )
    parser.add_argument(
        "--tnshap-max-degree", type=int, default=10,
        help="Polynomial degree in t for TN-SHAP interpolation."
    )
    parser.add_argument(
        "--n-targets", type=int, default=20,
        help="Number of base points to evaluate TN-SHAP on."
    )
    parser.add_argument(
        "--use-train", action="store_true",
        help="If set, draw TN-SHAP targets from x_train instead of x_test."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for Decision Tree."
    )
    args = parser.parse_args()

    print(f"=== Training Decision Tree on poly5 data and computing TN-SHAP ===")
    print(f"Device: {DEVICE}")
    print(f"Prefix: {args.prefix}")
    print(f"Decision Tree params: max_depth={args.max_depth}, "
          f"min_samples_split={args.min_samples_split}, "
          f"min_samples_leaf={args.min_samples_leaf}, seed={args.seed}")
    print(f"TN-SHAP params: max_degree={args.tnshap_max_degree}, n_targets={args.n_targets}")

    # ---------------------------------------------------------
    # 1) Load poly5 teacher and data
    # ---------------------------------------------------------
    print("\n--- Loading poly5 teacher and data ---")
    teacher = load_teacher(args.prefix)
    if isinstance(teacher, torch.nn.Module):
        teacher.to(DEVICE)

    x_train, y_train, x_test, y_test = load_data(args.prefix)
    x_train = x_train.to(DEVICE)
    y_train = y_train.to(DEVICE)
    x_test = x_test.to(DEVICE)
    y_test = y_test.to(DEVICE)

    N_train, D = x_train.shape
    N_test = x_test.shape[0]
    
    print(f"Training set: {N_train} samples, dimension D={D}")
    print(f"Test set: {N_test} samples")
    
    # Check if teacher has support S
    has_support = hasattr(teacher, "S")
    if has_support:
        S = list(teacher.S)
        k_active = len(S)
        print(f"Ground truth active set S (size {k_active}): {S}")
    else:
        print("Teacher has no attribute 'S'; skipping support accuracy.")

    # ---------------------------------------------------------
    # 2) Train Decision Tree
    # ---------------------------------------------------------
    print("\n--- Training Decision Tree ---")
    train_start = time.time()
    
    dt_model = DecisionTreeRegressor(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
    )
    
    # Convert to numpy for sklearn
    x_train_np = x_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy()
    x_test_np = x_test.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    
    # Train
    dt_model.fit(x_train_np, y_train_np)
    
    train_time = time.time() - train_start
    print(f"Training completed in {train_time:.2f} seconds")

    # ---------------------------------------------------------
    # 3) Evaluate Decision Tree
    # ---------------------------------------------------------
    print("\n--- Evaluating Decision Tree ---")
    
    # Training set performance
    y_train_pred = dt_model.predict(x_train_np)
    train_mse = mean_squared_error(y_train_np, y_train_pred)
    train_r2 = sklearn_r2_score(y_train_np, y_train_pred)
    
    # Test set performance
    y_test_pred = dt_model.predict(x_test_np)
    test_mse = mean_squared_error(y_test_np, y_test_pred)
    test_r2 = sklearn_r2_score(y_test_np, y_test_pred)
    
    print(f"Training set: MSE={train_mse:.6f}, R²={train_r2:.6f}")
    print(f"Test set:     MSE={test_mse:.6f}, R²={test_r2:.6f}")

    # ---------------------------------------------------------
    # 4) Prepare for TN-SHAP computation
    # ---------------------------------------------------------
    print("\n--- Preparing TN-SHAP computation ---")
    
    # Choose base points: by default from test if available, else from train
    if (not args.use_train) and (x_test is not None) and (x_test.shape[0] > 0):
        X_pool = x_test
        pool_name = "x_test"
    else:
        X_pool = x_train
        pool_name = "x_train"

    N_pool = X_pool.shape[0]
    N_targets = min(args.n_targets, N_pool)
    X_targets = X_pool[:N_targets]

    print(f"Using {N_targets} base points from {pool_name}")

    # Build Chebyshev nodes for interpolation in t
    max_degree = int(args.tnshap_max_degree)
    K = max_degree + 1
    t_nodes = chebyshev_nodes_unit_interval(
        K, device=DEVICE, dtype=X_targets.dtype
    )
    print(f"Using K={K} Chebyshev nodes in t for interpolation (degree <= {max_degree})")

    # ---------------------------------------------------------
    # 5) Compute TN-SHAP Shapley values
    # ---------------------------------------------------------
    print("\n--- Computing TN-SHAP Shapley values ---")
    
    shap_all = torch.zeros(N_targets, D, device=DEVICE, dtype=torch.float32)
    t_per_point = []
    
    if has_support:
        S_set = set(S)
        support_acc = []
    else:
        support_acc = None

    for idx in range(N_targets):
        x0 = X_targets[idx]
        t0 = time.time()
        
        phi = tnshap_vandermonde_dt(
            dt_model,
            x0,
            max_degree=max_degree,
            t_nodes=t_nodes,
        )
        
        t1 = time.time()
        shap_all[idx] = phi
        t_per_point.append(t1 - t0)

        if has_support:
            top_idx = torch.topk(phi.abs(), k_active).indices.tolist()
            correct = len(set(top_idx) & S_set)
            acc = correct / k_active
            support_acc.append(acc)
            print(
                f"[Point {idx:02d}] support acc={acc:.2f}, "
                f"time={1000*(t1-t0):.2f} ms"
            )
        else:
            print(f"[Point {idx:02d}] time={1000*(t1-t0):.2f} ms")

    # ---------------------------------------------------------
    # 6) Aggregate TN-SHAP statistics
    # ---------------------------------------------------------
    print("\n--- TN-SHAP Statistics ---")
    
    t_tensor = torch.tensor(t_per_point, device=DEVICE)
    mean_t = t_tensor.mean().item()
    std_t = t_tensor.std(unbiased=True).item() if len(t_tensor) > 1 else 0.0
    print(f"Per-point TN-SHAP time: mean={mean_t*1000:.2f} ms, std={std_t*1000:.2f} ms")

    if has_support and support_acc:
        support_acc_tensor = torch.tensor(support_acc, device=DEVICE)
        mean_acc = support_acc_tensor.mean().item()
        std_acc = support_acc_tensor.std(unbiased=True).item() \
            if len(support_acc_tensor) > 1 else 0.0
        print(f"Shapley support accuracy: mean={mean_acc:.4f}, std={std_acc:.4f}")

    # ---------------------------------------------------------
    # 7) Save results
    # ---------------------------------------------------------
    print("\n--- Saving results ---")
    
    # Create output directory if needed
    os.makedirs("dt_results", exist_ok=True)
    
    # Save TN-SHAP results
    tnshap_path = f"dt_results/{args.prefix}_dt_tnshap.pt"
    torch.save(
        {
            "x_targets": X_targets.detach().cpu(),     # (N_targets, D)
            "shapley": shap_all.detach().cpu(),        # (N_targets, D)
            "t_nodes": t_nodes.detach().cpu(),         # (K,)
            "max_degree": max_degree,
            "prefix": args.prefix,
            "has_support": has_support,
            "support": S if has_support else None,
            "dt_params": {
                "max_depth": args.max_depth,
                "min_samples_split": args.min_samples_split,
                "min_samples_leaf": args.min_samples_leaf,
                "seed": args.seed,
            },
            "performance": {
                "train_mse": train_mse,
                "train_r2": train_r2,
                "test_mse": test_mse,
                "test_r2": test_r2,
            },
        },
        tnshap_path,
    )
    print(f"Saved TN-SHAP results to {tnshap_path}")
    
    # Save Decision Tree model
    import pickle
    dt_model_path = f"dt_results/{args.prefix}_dt_model.pkl"
    with open(dt_model_path, "wb") as f:
        pickle.dump(dt_model, f)
    print(f"Saved Decision Tree model to {dt_model_path}")
    
    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
