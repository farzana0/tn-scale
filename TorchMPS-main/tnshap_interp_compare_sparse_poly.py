#!/usr/bin/env python3
"""
tnshap_vandermonde_compare_sparse_poly.py

TN-SHAP-style Shapley computation using:

  - Diagonal evaluations in a scalar parameter t
  - Chebyshev nodes in [0, 1]
  - Vandermonde system to recover degree-wise aggregates
  - A recurrence to extract per-feature degree aggregates β_{i,k}
  - Shapley_i = sum_k β_{i,k} / k

This is implemented generically for:

  - The teacher (eval_fn_teacher)
  - The trained MPS surrogate (eval_fn_mps)

We:
  - Load path-augmented training data and t-nodes from <prefix>_tnshap_targets.pt
  - Compute R2 on that dataset for teacher and MPS
  - Compute TN-SHAP on x_base (N_TARGETS points)
  - Report mean/std of:
      - correlation on S
      - support accuracy (top-|S| vs true S) for teacher and MPS
      - eval runtime
"""

import argparse
import math
import time

import torch
from torchmps import MPS

from poly_teacher import load_teacher, load_data, DEVICE
from train_mps_paths import r2_score, chebyshev_nodes_unit_interval


PREFIX_DEFAULT = "poly"


# -----------------------
# Helpers from before
# -----------------------

def augment_with_one(x: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
    return torch.cat([ones, x], dim=1)


def load_mps(prefix: str) -> MPS:
    state = torch.load(f"{prefix}_mps.pt", map_location=DEVICE)
    D_aug = state["D_aug"]
    bond_dim = state["bond_dim"]

    mps = MPS(
        input_dim=D_aug,
        output_dim=1,
        bond_dim=bond_dim,
        adaptive_mode=False,
        periodic_bc=False,
    ).to(DEVICE)
    mps.load_state_dict(state["state_dict"])
    mps.eval()
    return mps


# -----------------------
# Vandermonde utilities
# -----------------------

def build_vandermonde(t_nodes: torch.Tensor, degree_max: int) -> torch.Tensor:
    """
    Build Vandermonde matrix V where:

      V[l, k] = t_nodes[l] ** k

    for k = 0..degree_max and l = 0..n_nodes-1.
    """
    t = t_nodes.to(dtype=torch.float64)
    exponents = torch.arange(0, degree_max + 1, dtype=torch.float64, device=t.device)
    V = t.unsqueeze(1) ** exponents.unsqueeze(0)
    return V


def solve_poly_coeffs(V: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Solve V c = y for the polynomial coefficients c, where V is Vandermonde
    (n_nodes x (degree_max+1)), y is (n_nodes,).
    """
    V64 = V.to(dtype=torch.float64)
    y64 = y.to(dtype=torch.float64)
    c = torch.linalg.solve(V64, y64)
    return c.to(dtype=torch.float32)


# -----------------------
# Diagonal evaluations h(t), g_i(t)
# -----------------------

def eval_h_on_nodes(eval_fn, x: torch.Tensor, t_nodes: torch.Tensor) -> torch.Tensor:
    """
    h(t) = f(t * x), evaluated on all t_nodes.

    x: (D,)
    t_nodes: (L,)
    returns: (L,)
    """
    x = x.to(DEVICE)
    t_nodes = t_nodes.to(x.device)
    X_batch = (t_nodes.unsqueeze(1) * x.unsqueeze(0))  # (L, D)
    with torch.no_grad():
        y = eval_fn(X_batch)
        if y.ndim > 1:
            y = y.squeeze(-1)
    return y


def eval_g_i_on_nodes(eval_fn, x: torch.Tensor, i: int, t_nodes: torch.Tensor) -> torch.Tensor:
    """
    g_i(t) = f(x_i fixed, others scaled by t), evaluated on all t_nodes.

    More precisely, for each t:

      x_i(t)[j] = x[j] if j == i
                  t * x[j] otherwise
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
# TN-SHAP via degree aggregates (Vandermonde + recurrence)
# -----------------------

def tnshap_vandermonde_for_point(
    eval_fn,
    x: torch.Tensor,
    max_degree: int,
    t_nodes: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Shapley values φ_i for all features i at point x using:

      - h(t) = f(t * x) -> coefficients α_k
      - g_i(t) = f(x_i fixed, others scaled by t) -> coefficients γ_{i,k}
      - Recurrence to get β_{i,k}
      - φ_i = sum_k β_{i,k} / k

    Assumes baseline is 0 (v(S) = f(x_S)).
    """
    x = x.to(DEVICE)
    D = x.shape[0]

    # Ensure we have at least max_degree+1 nodes
    assert t_nodes.shape[0] >= max_degree + 1, "Need at least max_degree+1 t-nodes."
    t_nodes_sq = t_nodes[: max_degree + 1]
    V = build_vandermonde(t_nodes_sq, degree_max=max_degree)

    # 1) h(t) = f(t*x)
    h_vals = eval_h_on_nodes(eval_fn, x, t_nodes_sq)
    alpha = solve_poly_coeffs(V, h_vals)  # (max_degree+1,)
    alpha64 = alpha.to(torch.float64)

    phi_full = torch.zeros(D, device=DEVICE, dtype=torch.float32)

    # 2) For each feature i, recover β_{i,k}
    for i in range(D):
        g_vals = eval_g_i_on_nodes(eval_fn, x, i, t_nodes_sq)
        gamma = solve_poly_coeffs(V, g_vals)
        gamma64 = gamma.to(torch.float64)

        beta = torch.zeros(max_degree + 1, dtype=torch.float64, device=x.device)

        # k = max_degree: γ_d = α_d - β_d  => β_d = α_d - γ_d
        beta[max_degree] = alpha64[max_degree] - gamma64[max_degree]

        # k = max_degree-1 down to 1: γ_k = α_k + β_{k+1} - β_k => β_k = α_k + β_{k+1} - γ_k
        for k in range(max_degree - 1, 0, -1):
            beta[k] = alpha64[k] + beta[k + 1] - gamma64[k]

        # consistency for k=0: γ_0 = α_0 + β_1 => β_1 = γ_0 - α_0
        beta1_from_gamma0 = gamma64[0] - alpha64[0]
        beta[1] = 0.5 * (beta[1] + beta1_from_gamma0)

        ks = torch.arange(1, max_degree + 1, dtype=torch.float64, device=x.device)
        phi_i = torch.sum(beta[1:] / ks)
        phi_full[i] = phi_i.to(torch.float32)

    return phi_full


# -----------------------
# Main comparison
# -----------------------

def list_mean_std(xs):
    if len(xs) == 0:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, math.sqrt(var)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default=PREFIX_DEFAULT,
                        help="Prefix for teacher/MPS/targets files.")
    parser.add_argument("--override-max-degree", type=int, default=None,
                        help="If set, overrides max_degree saved in tnshap_targets.")
    args = parser.parse_args()

    prefix = args.prefix
    print(f"Prefix: {prefix}")

    # ---------------------------------------------------------
    # 1) Load TN-SHAP targets + t-nodes + training dataset
    # ---------------------------------------------------------
    targets = torch.load(f"{prefix}_tnshap_targets.pt", map_location=DEVICE)
    X_targets = targets["x_base"].to(DEVICE)     # (N_TARGETS, D)
    t_nodes   = targets["t_nodes"].to(DEVICE)    # (N_T_NODES,)
    X_all     = targets["X_all"].to(DEVICE)      # ((1 + D)*N_base*N_T_NODES, D)
    Y_all     = targets["Y_all"].to(DEVICE)      # same length
    max_degree_saved = int(targets["max_degree"])

    N_TARGETS = X_targets.shape[0]
    D         = X_targets.shape[1]

    max_degree = args.override_max_degree if args.override_max_degree is not None else max_degree_saved
    max_degree = int(max_degree)
    print(f"N_TARGETS={N_TARGETS}, D={D}, max_degree={max_degree}")

    # ---------------------------------------------------------
    # 2) Load teacher and MPS
    # ---------------------------------------------------------
    teacher = load_teacher(prefix)
    mps = load_mps(prefix)

    def eval_fn_teacher(x_batch: torch.Tensor) -> torch.Tensor:
        x_batch = x_batch.to(DEVICE)
        with torch.no_grad():
            return teacher(x_batch)

    def eval_fn_mps(x_batch: torch.Tensor) -> torch.Tensor:
        x_batch = x_batch.to(DEVICE)
        x_aug = augment_with_one(x_batch)
        with torch.no_grad():
            return mps(x_aug).squeeze(-1)

    # ---------------------------------------------------------
    # 3) Sanity: R2 of MPS vs TEACHER on THE SAME TN-SHAP TRAINING DATA
    # ---------------------------------------------------------
    with torch.no_grad():
        y_teacher_all = eval_fn_teacher(X_all)   # teacher(X_all)
        y_mps_all     = eval_fn_mps(X_all)       # mps(X_all)

    r2_teacher = r2_score(Y_all, y_teacher_all)  # checks data consistency
    r2_mps     = r2_score(Y_all, y_mps_all)

    print(f"Teacher vs path-aug ground truth: R2 = {r2_teacher:.4f}")
    print(f"MPS vs path-aug ground truth:     R2 = {r2_mps:.4f}")

    # ---------------------------------------------------------
    # 4) Shapley via TN-SHAP on TEACHER and MPS
    # ---------------------------------------------------------
    S = teacher.S
    print(f"Active subset S (size {len(S)}): {S}")
    k_active = len(S)
    S_set = set(S)

    phis_teacher_all = []
    phis_mps_all     = []
    corrs            = []
    acc_teacher_list = []
    acc_mps_list     = []
    phi_min_list     = []
    phi_max_list     = []

    mse_h_list       = []
    mse_g_mean_list  = []
    mse_g_max_list   = []

    eval_start = time.time()

    for idx in range(N_TARGETS):
        x0 = X_targets[idx]  # base point

        # 1) Evaluate TN-SHAP
        phi_teacher = tnshap_vandermonde_for_point(
            eval_fn_teacher,
            x0,
            max_degree=max_degree,
            t_nodes=t_nodes,
        )
        phi_mps = tnshap_vandermonde_for_point(
            eval_fn_mps,
            x0,
            max_degree=max_degree,
            t_nodes=t_nodes,
        )

        phis_teacher_all.append(phi_teacher.unsqueeze(0))
        phis_mps_all.append(phi_mps.unsqueeze(0))

        # 2) Measure function error on queried paths
        h_teacher = eval_h_on_nodes(eval_fn_teacher, x0, t_nodes)
        h_mps     = eval_h_on_nodes(eval_fn_mps,     x0, t_nodes)
        mse_h = torch.mean((h_teacher - h_mps) ** 2).item()

        mse_g_list = []
        for i in range(D):
            g_teacher = eval_g_i_on_nodes(eval_fn_teacher, x0, i, t_nodes)
            g_mps     = eval_g_i_on_nodes(eval_fn_mps,     x0, i, t_nodes)
            mse_g_i = torch.mean((g_teacher - g_mps) ** 2).item()
            mse_g_list.append(mse_g_i)

        mse_g_mean = sum(mse_g_list) / len(mse_g_list)
        mse_g_max  = max(mse_g_list)
        mse_h_list.append(mse_h)
        mse_g_mean_list.append(mse_g_mean)
        mse_g_max_list.append(mse_g_max)

        # 3) Correlation + support accuracy
        s_t = phi_teacher[S]
        s_m = phi_mps[S]
        if torch.std(s_t) < 1e-8 or torch.std(s_m) < 1e-8:
            corr = float("nan")
        else:
            c = torch.corrcoef(torch.stack([s_t, s_m]))[0, 1]
            corr = float(c.item())
        corrs.append(corr)

        top_teacher = torch.topk(phi_teacher.abs(), k_active).indices.tolist()
        top_mps     = torch.topk(phi_mps.abs(),     k_active).indices.tolist()

        correct_teacher = len(set(top_teacher) & S_set)
        correct_mps     = len(set(top_mps)     & S_set)

        acc_teacher = correct_teacher / k_active
        acc_mps     = correct_mps     / k_active

        acc_teacher_list.append(acc_teacher)
        acc_mps_list.append(acc_mps)

        phi_norm_S = phi_teacher[S].abs()
        phi_min = phi_norm_S.min().item()
        phi_max = phi_norm_S.max().item()
        phi_min_list.append(phi_min)
        phi_max_list.append(phi_max)

        print(
            f"[Point {idx:02d}] "
            f"corr={corr:.4f} | "
            f"acc teacher={acc_teacher:.2f}, MPS={acc_mps:.2f} | "
            f"MSE h={mse_h:.3e}, "
            f"MSE g mean={mse_g_mean:.3e}, "
            f"MSE g max={mse_g_max:.3e}"
        )
        print(
            f"[Point {idx:02d}] "
            f"|phi_teacher[S]| min={phi_min:.3e}, max={phi_max:.3e}"
        )

    eval_total_time = time.time() - eval_start
    eval_time_per_point = eval_total_time / max(N_TARGETS, 1)

    phis_teacher_all = torch.cat(phis_teacher_all, dim=0)
    phis_mps_all     = torch.cat(phis_mps_all,     dim=0)

    # Means/stds
    finite_corrs = [c for c in corrs if not math.isnan(c)]
    mean_corr, std_corr = list_mean_std(finite_corrs)

    mean_acc_teacher, std_acc_teacher = list_mean_std(acc_teacher_list)
    mean_acc_mps,     std_acc_mps     = list_mean_std(acc_mps_list)

    mean_mse_h,      std_mse_h      = list_mean_std(mse_h_list)
    mean_mse_g_mean, std_mse_g_mean = list_mean_std(mse_g_mean_list)
    mean_mse_g_max,  std_mse_g_max  = list_mean_std(mse_g_max_list)

    mean_phi_min, std_phi_min = list_mean_std(phi_min_list)
    mean_phi_max, std_phi_max = list_mean_std(phi_max_list)

    print("\n==== Summary over targets (TN-SHAP via Vandermonde compression) ====")
    print(f"Mean attribution corr on S:      {mean_corr:.4f} ± {std_corr:.4f}")
    print(f"Mean support acc on S (teacher): {mean_acc_teacher:.4f} ± {std_acc_teacher:.4f}")
    print(f"Mean support acc on S (MPS):     {mean_acc_mps:.4f} ± {std_acc_mps:.4f}")

    print(f"\nFunction MSE on paths (MPS vs teacher):")
    print(f"  h(t):   mean={mean_mse_h:.3e} ± {std_mse_h:.3e}")
    print(f"  g_i(t): mean MSE={mean_mse_g_mean:.3e} ± {std_mse_g_mean:.3e}")
    print(f"          max MSE={mean_mse_g_max:.3e} ± {std_mse_g_max:.3e}")

    print(f"\n|phi_teacher[S]| stats:")
    print(f"  min over S: mean={mean_phi_min:.3e} ± {std_phi_min:.3e}")
    print(f"  max over S: mean={mean_phi_max:.3e} ± {std_phi_max:.3e}")

    print(f"\nEvaluation runtime:")
    print(f"  total:      {eval_total_time:.3f} sec")
    print(f"  per point:  {eval_time_per_point:.6f} sec")

    print("\nDone. This script now gives you:")
    print("- TN-SHAP on the TRUE teacher, on exactly the same base points and t-nodes.")
    print("- TN-SHAP on the MPS, trained on exactly the same h/g_i path points.")
    print("- Mean/std of correlation, support accuracy, path-MSE, and eval runtime.")


if __name__ == "__main__":
    main()
