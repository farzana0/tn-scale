#!/usr/bin/env python3
"""
tnshap_sqexp_vandermonde_eval.py

TN-SHAP-style Shapley computation *only for the exponential teacher*.

Given:
  - A teacher f(x) (sqexp teacher loaded via `load_teacher(prefix)`),
  - A dataset (x_train, y_train, x_test, y_test) from `load_data(prefix)`,

we:

  - Pick N_TARGETS base points (by default from x_test if non-empty, else x_train).
  - Build K = max_degree+1 Chebyshev nodes t in [0, 1].
  - For each base point x:
      * Evaluate h(t)   = f(t * x)         at all t nodes.
      * For each i, g_i(t) = f(x_i fixed, others scaled by t).
      * Fit h and each g_i with a degree <= max_degree polynomial in t
        by solving a square Vandermonde system.
      * Use the TN-SHAP recurrence to obtain β_{i,k} from the coefficients
        and compute Shapley_i(x) = sum_{k=1..max_degree} β_{i,k} / k.

We:
  - Print some basic information.
  - If the teacher has an attribute `S` (active subset of features),
    we compute support accuracy of TN-SHAP (top-|S| coordinates).
  - Save all Shapley vectors and base points to:
        <prefix>_sqexp_tnshap_teacher.pt
"""

import argparse
import time
from typing import Tuple

import torch

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


def eval_h_on_nodes_teacher(eval_fn,
                            x: torch.Tensor,
                            t_nodes: torch.Tensor) -> torch.Tensor:
    """
    h(t) = f(t * x) for teacher, evaluated for all t_nodes.

    x: (D,)
    t_nodes: (L,)
    returns: (L,)
    """
    x = x.to(DEVICE)
    t_nodes = t_nodes.to(x.device)
    X_batch = t_nodes.unsqueeze(1) * x.unsqueeze(0)  # (L, D)
    with torch.no_grad():
        y = eval_fn(X_batch)
        if y.ndim > 1:
            y = y.squeeze(-1)
    return y


def eval_g_i_on_nodes_teacher(eval_fn,
                              x: torch.Tensor,
                              i: int,
                              t_nodes: torch.Tensor) -> torch.Tensor:
    """
    g_i(t) = f(x_i fixed, others scaled by t) for teacher.

    x: (D,)
    t_nodes: (L,)
    returns: (L,)
    """
    x = x.to(DEVICE)
    t_nodes = t_nodes.to(x.device)
    D = x.shape[0]

    X_scaled = t_nodes.unsqueeze(1) * x.unsqueeze(0)  # (L, D)
    X_scaled[:, i] = x[i]  # clamp feature i

    with torch.no_grad():
        y = eval_fn(X_scaled)
        if y.ndim > 1:
            y = y.squeeze(-1)
    return y


# -----------------------
# TN-SHAP via Vandermonde + recurrence (teacher only)
# -----------------------

def tnshap_vandermonde_teacher(
    eval_fn,
    x: torch.Tensor,
    max_degree: int,
    t_nodes: torch.Tensor,
) -> torch.Tensor:
    """
    TN-SHAP for the TEACHER, with paths in ORIGINAL x-space: t * x.

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
    h_vals = eval_h_on_nodes_teacher(eval_fn, x, t_nodes_use)
    alpha = solve_poly_coeffs(V, h_vals)            # (max_degree+1,)
    alpha64 = alpha.to(torch.float64)

    # Final Shapley vector
    phi_full = torch.zeros(D, device=DEVICE, dtype=torch.float32)

    # --- per-feature recurrence over g_i(t) ---
    for i in range(D):
        g_vals = eval_g_i_on_nodes_teacher(eval_fn, x, i, t_nodes_use)
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
        "--prefix", type=str, default="sqexp_D50",
        help="Prefix for teacher and data (as in train_mps_sqexp_paths.py)."
    )
    parser.add_argument(
        "--max-degree", type=int, default=10,
        help="Polynomial degree in t for TN-SHAP interpolation."
    )
    parser.add_argument(
        "--n-targets", type=int, default=20,
        help="Number of base points to evaluate TN-SHAP on."
    )
    parser.add_argument(
        "--use-train", action="store_true",
        help="If set, draw targets from x_train instead of x_test."
    )
    args = parser.parse_args()

    prefix = args.prefix
    print(f"=== TN-SHAP sqexp Vandermonde Eval (Teacher Only) for PREFIX={prefix} ===")
    print(f"max_degree={args.max_degree}, n_targets={args.n_targets}, use_train={args.use_train}")

    # 1) Load teacher and data
    teacher = load_teacher(prefix)
    if isinstance(teacher, torch.nn.Module):
        teacher.to(DEVICE)

    x_train, y_train, x_test, y_test = load_data(prefix)
    x_train = x_train.to(DEVICE)
    x_test = x_test.to(DEVICE)

    # Choose base points: by default from test if available, else from train
    if (not args.use_train) and (x_test is not None) and (x_test.shape[0] > 0):
        X_pool = x_test
        pool_name = "x_test"
    else:
        X_pool = x_train
        pool_name = "x_train"

    N_pool, D = X_pool.shape
    N_targets = min(args.n_targets, N_pool)
    X_targets = X_pool[:N_targets]

    print(f"Using {N_targets} base points from {pool_name}, dimension D={D}")

    # 2) Build Chebyshev nodes for interpolation in t
    max_degree = int(args.max_degree)
    K = max_degree + 1
    t_nodes = chebyshev_nodes_unit_interval(
        K, device=DEVICE, dtype=X_targets.dtype
    )
    print(f"Using K={K} Chebyshev nodes in t for interpolation (degree <= {max_degree}).")

    # helper to evaluate teacher on batches
    def eval_fn_teacher(x_batch: torch.Tensor) -> torch.Tensor:
        x_batch = x_batch.to(DEVICE)
        with torch.no_grad():
            y = teacher(x_batch)
            if y.ndim > 1:
                y = y.squeeze(-1)
        return y

    # 3) Compute TN-SHAP Shapley vectors for each target point
    shap_all = torch.zeros(N_targets, D, device=DEVICE, dtype=torch.float32)

    # If teacher has a "support" S, we can evaluate support recovery
    has_support = hasattr(teacher, "S")
    if has_support:
        S = list(teacher.S)
        S_set = set(S)
        k_active = len(S)
        print(f"Teacher has support S of size {k_active}: {S}")
        support_acc = []
    else:
        print("Teacher has no attribute 'S'; skipping support accuracy.")
        support_acc = None

    t_per_point = []

    for idx in range(N_targets):
        x0 = X_targets[idx]
        t0 = time.time()
        phi = tnshap_vandermonde_teacher(
            eval_fn_teacher,
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

    # 4) Aggregate stats
    t_tensor = torch.tensor(t_per_point, device=DEVICE)
    mean_t = t_tensor.mean().item()
    std_t = t_tensor.std(unbiased=True).item() if len(t_tensor) > 1 else 0.0
    print("==== Per-point TN-SHAP evaluation time (teacher) ====")
    print(f"mean = {mean_t*1000:.2f} ms, std = {std_t*1000:.2f} ms")

    if has_support and support_acc:
        support_acc_tensor = torch.tensor(support_acc, device=DEVICE)
        mean_acc = support_acc_tensor.mean().item()
        std_acc = support_acc_tensor.std(unbiased=True).item() \
            if len(support_acc_tensor) > 1 else 0.0
        print("==== Shapley support accuracy (teacher vs S) ====")
        print(f"mean acc = {mean_acc:.4f}, std = {std_acc:.4f}")

    # 5) Save results
    out_path = f"{prefix}_sqexp_tnshap_teacher.pt"
    torch.save(
        {
            "x_targets": X_targets.detach().cpu(),     # (N_targets, D)
            "shapley": shap_all.detach().cpu(),       # (N_targets, D)
            "t_nodes": t_nodes.detach().cpu(),        # (K,)
            "max_degree": max_degree,
            "prefix": prefix,
            "has_support": has_support,
            "support": teacher.S if has_support else None,
        },
        out_path,
    )
    print(f"Saved TN-SHAP teacher Shapley results to {out_path}")


if __name__ == "__main__":
    main()
