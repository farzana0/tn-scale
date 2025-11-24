#!/usr/bin/env python3
"""
tnshap_vandermonde_compare_sparse_poly.py

TN-SHAP-style Shapley computation using:

  - Diagonal evaluations in a scalar parameter t
  - Chebyshev (or any) interpolation nodes in [0, 1]
  - Vandermonde system to recover degree-wise aggregates
  - A recurrence to extract per-feature degree aggregates β_{i,k}
  - Shapley_i = sum_k β_{i,k} / k

This is implemented generically for:

  - The sparse polynomial teacher (eval_fn_teacher)
  - The trained MPS surrogate (eval_fn_mps)

No permutation or subset enumeration.
"""

import math
import torch
from torchmps import MPS

from poly_teacher import load_teacher, load_data, DEVICE
from train_mps import r2_score

PREFIX = "poly"


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
# Interpolation nodes (Chebyshev -> [0,1])
# -----------------------

def chebyshev_nodes_unit_interval(n_nodes: int, device=None, dtype=torch.float32):
    """
    Chebyshev nodes of the first kind, mapped from [-1, 1] to [0, 1].

    Standard Chebyshev nodes in [-1, 1]:
        u_k = cos((2k-1)/(2n) * pi),  k=1..n

    Map to t in [0,1] via: t = (u + 1) / 2.
    """
    if device is None:
        device = DEVICE
    k = torch.arange(1, n_nodes + 1, dtype=torch.float64, device=device)
    u = torch.cos((2.0 * k - 1.0) / (2.0 * n_nodes) * torch.pi)  # [-1,1]
    t = (u + 1.0) / 2.0  # [0,1]
    return t.to(dtype)


# -----------------------
# Vandermonde utilities
# -----------------------

def build_vandermonde(t_nodes: torch.Tensor, degree_max: int) -> torch.Tensor:
    """
    Build Vandermonde matrix V where:

      V[l, k] = t_nodes[l] ** k

    for k = 0..degree_max and l = 0..n_nodes-1.

    We'll usually take n_nodes = degree_max + 1 to get a square system.
    """
    t = t_nodes.to(dtype=torch.float64)
    n_nodes = t.shape[0]
    exponents = torch.arange(0, degree_max + 1, dtype=torch.float64, device=t.device)
    # shape: (n_nodes, degree_max+1)
    V = t.unsqueeze(1) ** exponents.unsqueeze(0)
    return V


def solve_poly_coeffs(V: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Solve V c = y for the polynomial coefficients c, where V is Vandermonde
    (n_nodes x (degree_max+1)), y is (n_nodes,).

    We assume V is square (n_nodes == degree_max+1) and well-conditioned enough.
    """
    V64 = V.to(dtype=torch.float64)
    y64 = y.to(dtype=torch.float64)
    # shape: (degree_max+1,)
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
    # Build batch: for each t, compute t * x
    X_batch = (t_nodes.unsqueeze(1) * x.unsqueeze(0))  # (L, D)
    with torch.no_grad():
        y = eval_fn(X_batch)  # (L,) or (L,1)
        if y.ndim > 1:
            y = y.squeeze(-1)
    return y


def eval_g_i_on_nodes(eval_fn, x: torch.Tensor, i: int, t_nodes: torch.Tensor) -> torch.Tensor:
    """
    g_i(t) = f(x_i fixed, others scaled by t), evaluated on all t_nodes.

    More precisely, for each t:

      x_i(t)[j] = x[j] if j == i
                  t * x[j] otherwise

    This corresponds to z_i = 1, z_j = t in the multilinear-extension view.
    """
    x = x.to(DEVICE)
    t_nodes = t_nodes.to(x.device)
    D = x.shape[0]

    # Start with all features scaled: t * x
    X_scaled = t_nodes.unsqueeze(1) * x.unsqueeze(0)  # (L, D)

    # Then override column i with the original x[i] (constant across t)
    X_scaled[:, i] = x[i]

    with torch.no_grad():
        y = eval_fn(X_scaled)  # (L,) or (L,1)
        if y.ndim > 1:
            y = y.squeeze(-1)
    return y


# -----------------------
# TN-SHAP via degree aggregates (Vandermonde + recurrence)
# -----------------------

def tnshap_vandermonde_for_point(
    eval_fn,
    x: torch.Tensor,
    baseline_val: float = 0.0,  # currently always 0, but kept for symmetry
    max_degree: int | None = None,
    t_nodes: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute Shapley values φ_i for all features i at point x using:

      - h(t) = f(t * x) -> coefficients α_k
      - g_i(t) = f(x_i fixed, others scaled by t) -> coefficients γ_{i,k}
      - Recurrence to get β_{i,k}
      - φ_i = sum_k β_{i,k} / k

    Assumes baseline is 0 (so x_off = 0), which is consistent with
    the v(S) = f(x_S) game and the TN-SHAP multilinear-extension view.

    eval_fn: callable (B, D) -> (B,)
    x: (D,)
    baseline_val: scalar, currently unused (we assume 0)
    max_degree: maximum polynomial degree to fit (<= D). If None, use D.
    t_nodes: optional precomputed nodes; if None, we use Chebyshev nodes of size max_degree+1.

    Returns:
        phi_full: (D,) tensor of Shapley values.
    """
    x = x.to(DEVICE)
    D = x.shape[0]
    if max_degree is None:
        max_degree = D  # safe upper bound

    # Choose number of nodes = max_degree + 1 for exact interpolation of degree <= max_degree
    if t_nodes is None:
        t_nodes = chebyshev_nodes_unit_interval(max_degree + 1, device=x.device, dtype=x.dtype)
    else:
        t_nodes = t_nodes.to(device=x.device, dtype=x.dtype)
        assert t_nodes.shape[0] >= max_degree + 1, "Need at least max_degree+1 nodes for interpolation."

    # We only use the first max_degree+1 nodes to build a square Vandermonde
    t_nodes_sq = t_nodes[: max_degree + 1]  # (L,) with L = max_degree+1
    V = build_vandermonde(t_nodes_sq, degree_max=max_degree)  # (L, max_degree+1)

    # 1) Compute h(t) = f(t * x), fit α_k
    h_vals = eval_h_on_nodes(eval_fn, x, t_nodes_sq)  # (L,)
    alpha = solve_poly_coeffs(V, h_vals)  # (max_degree+1,) with alpha[k] = α_k

    # 2) For each feature i, compute g_i(t) and its coefficients γ_{i,k}
    phi_full = torch.zeros(D, device=DEVICE, dtype=torch.float32)

    # Pre-store α_0..α_max_degree as float64 for better stability in recurrence
    alpha64 = alpha.to(torch.float64)

    for i in range(D):
        # Evaluate g_i(t) on the same nodes
        g_vals = eval_g_i_on_nodes(eval_fn, x, i, t_nodes_sq)  # (L,)
        gamma = solve_poly_coeffs(V, g_vals)  # (max_degree+1,)
        gamma64 = gamma.to(torch.float64)

        # Recurrence to recover β_{i,k} for k = 1..max_degree

        beta = torch.zeros(max_degree + 1, dtype=torch.float64, device=x.device)  # β_k, index by k

        # k = max_degree: γ_d = α_d - β_d  => β_d = α_d - γ_d
        beta[max_degree] = alpha64[max_degree] - gamma64[max_degree]

        # k = max_degree-1 down to 1: γ_k = α_k + β_{k+1} - β_k => β_k = α_k + β_{k+1} - γ_k
        for k in range(max_degree - 1, 0, -1):
            beta[k] = alpha64[k] + beta[k + 1] - gamma64[k]

        # consistency for k=0: γ_0 = α_0 + β_1 => β_1 = γ_0 - α_0
        # (we can either assert or overwrite; here we'll average with recurrence result)
        beta1_from_gamma0 = gamma64[0] - alpha64[0]
        beta[1] = 0.5 * (beta[1] + beta1_from_gamma0)

        # Shapley: φ_i = sum_{k=1}^max_degree β_k / k
        ks = torch.arange(1, max_degree + 1, dtype=torch.float64, device=x.device)
        phi_i = torch.sum(beta[1:] / ks)
        phi_full[i] = phi_i.to(torch.float32)

    return phi_full


# -----------------------
# Main comparison
# -----------------------

def main():
    # ---------------------------------------------------------
    # 1) Load saved TN-SHAP targets + t-nodes + training dataset
    # ---------------------------------------------------------
    targets = torch.load(f"{PREFIX}_tnshap_targets.pt", map_location=DEVICE)
    X_targets = targets["x_base"].to(DEVICE)     # (N_TARGETS, D) base points used in training/eval
    t_nodes   = targets["t_nodes"].to(DEVICE)    # (N_T_NODES,)
    X_all     = targets["X_all"].to(DEVICE)      # ((1 + D)*N_base*N_T_NODES, D)
    Y_all     = targets["Y_all"].to(DEVICE)      # same length
    max_degree_saved = int(targets["max_degree"])

    N_TARGETS = X_targets.shape[0]
    D         = X_targets.shape[1]

    # max_degree for TN-SHAP interpolation = true polynomial degree (not D)
    # max_degree = min(max_degree_saved, t_nodes.shape[0] - 1)
    max_degree = 9

    # ---------------------------------------------------------
    # 2) Load teacher and MPS
    # ---------------------------------------------------------
    teacher = load_teacher(PREFIX)
    mps = load_mps(PREFIX)

    # Two eval functions: one for exact polynomial (teacher), one for MPS surrogate
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
        y_teacher_all = eval_fn_teacher(X_all)   # teacher(X_all) – should match Y_all
        y_mps_all     = eval_fn_mps(X_all)       # mps(X_all)

    r2_teacher = r2_score(Y_all, y_teacher_all)  # checks data consistency
    r2_mps     = r2_score(Y_all, y_mps_all)

    print(f"Teacher vs path-aug ground truth: R2 = {r2_teacher:.4f}")
    print(f"MPS vs path-aug ground truth:     R2 = {r2_mps:.4f}")

    # ---------------------------------------------------------
    # 4) Shapley via TN-SHAP (multilinear extension) on TEACHER and MPS
    # ---------------------------------------------------------
    S = teacher.S
    print(f"Active subset S (teacher support): {S}")
    k_active = len(S)
    S_set = set(S)

    phis_teacher_all = []
    phis_mps_all     = []
    corrs            = []
    acc_teacher_list = []
    acc_mps_list     = []

    for idx in range(N_TARGETS):
        x0 = X_targets[idx]  # base point used in training

        # --------------------------------------------
        # 1) Evaluate TN-SHAP
        # --------------------------------------------
        phi_teacher = tnshap_vandermonde_for_point(
            eval_fn_teacher,
            x0,
            baseline_val=0.0,
            max_degree=max_degree,
            t_nodes=t_nodes,
        )
        phi_mps = tnshap_vandermonde_for_point(
            eval_fn_mps,
            x0,
            baseline_val=0.0,
            max_degree=max_degree,
            t_nodes=t_nodes,
        )

        phis_teacher_all.append(phi_teacher.unsqueeze(0))
        phis_mps_all.append(phi_mps.unsqueeze(0))

        # --------------------------------------------
        # 2) Measure function error on queried paths
        # --------------------------------------------

        # ---- h(t) path error ----
        h_teacher = eval_h_on_nodes(eval_fn_teacher, x0, t_nodes)
        h_mps     = eval_h_on_nodes(eval_fn_mps,     x0, t_nodes)

        mse_h = torch.mean((h_teacher - h_mps) ** 2).item()

        # ---- g_i(t) path error, per feature ----
        mse_g_list = []

        for i in range(D):
            g_teacher = eval_g_i_on_nodes(eval_fn_teacher, x0, i, t_nodes)
            g_mps     = eval_g_i_on_nodes(eval_fn_mps,     x0, i, t_nodes)

            mse_g_i = torch.mean((g_teacher - g_mps) ** 2).item()
            mse_g_list.append(mse_g_i)

        mse_g_mean = sum(mse_g_list) / len(mse_g_list)
        mse_g_max  = max(mse_g_list)

        # --------------------------------------------
        # 3) Correlation + support accuracy
        # --------------------------------------------
        s_t = phi_teacher[S]
        s_m = phi_mps[S]
        if torch.std(s_t) < 1e-8 or torch.std(s_m) < 1e-8:
            corr = float("nan")
        else:
            c = torch.corrcoef(torch.stack([s_t, s_m]))[0, 1]
            corr = float(c.item())
        corrs.append(corr)

        # support accuracy
        top_teacher = torch.topk(phi_teacher.abs(), k_active).indices.tolist()
        top_mps     = torch.topk(phi_mps.abs(),     k_active).indices.tolist()

        correct_teacher = len(set(top_teacher) & S_set)
        correct_mps     = len(set(top_mps)     & S_set)

        acc_teacher = correct_teacher / k_active
        acc_mps     = correct_mps     / k_active

        acc_teacher_list.append(acc_teacher)
        acc_mps_list.append(acc_mps)

        # --------------------------------------------
        # 4) Print everything
        # --------------------------------------------
        print(
            f"[Point {idx:02d}] "
            f"corr={corr:.4f} | "
            f"acc teacher={acc_teacher:.2f}, MPS={acc_mps:.2f} | "
            f"MSE h={mse_h:.3e}, "
            f"MSE g mean={mse_g_mean:.3e}, "
            f"MSE g max={mse_g_max:.3e}"
        )
        phi_norm_S = phi_teacher[S].abs()
        print(
            f"[Point {idx:02d}] "
            f"|phi_teacher[S]| min={phi_norm_S.min().item():.3e}, "
            f"max={phi_norm_S.max().item():.3e}"
)



    phis_teacher_all = torch.cat(phis_teacher_all, dim=0)
    phis_mps_all     = torch.cat(phis_mps_all,     dim=0)

    finite_corrs = [c for c in corrs if not math.isnan(c)]
    mean_corr = sum(finite_corrs) / len(finite_corrs) if finite_corrs else float("nan")

    mean_acc_teacher = sum(acc_teacher_list) / len(acc_teacher_list)
    mean_acc_mps     = sum(acc_mps_list)     / len(acc_mps_list)

    print("\n==== Summary over targets (TN-SHAP via Vandermonde compression) ====")
    print(f"Mean attribution correlation on S:      {mean_corr:.4f}")
    print(f"Mean support accuracy on S (teacher):  {mean_acc_teacher:.4f}")
    print(f"Mean support accuracy on S (MPS):      {mean_acc_mps:.4f}")

    print("\nDone. This script now gives you:")
    print("- TN-SHAP (multilinear extension) on the TRUE teacher, "
          "on exactly the same base points and t-nodes.")
    print("- TN-SHAP on the MPS, trained on exactly the same h/g_i path points.")
    print("- If training MSE → 0 on X_all, then TN-SHAP(MPS) == TN-SHAP(teacher) "
          "up to numerical precision.")


if __name__ == "__main__":
    main()
