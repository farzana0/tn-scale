#!/usr/bin/env python3
"""
tnshap_vandermonde_compare_sparse_poly.py

TN-SHAP-style Shapley computation using:

  - Diagonal evaluations in a scalar parameter t
  - Chebyshev (or any) interpolation nodes in [0, 1]
  - Vandermonde system to recover degree-wise aggregates
  - A recurrence to extract per-feature degree aggregates β_{i,k}
  - Shapley_i = sum_k β_{i,k} / k

We compare:

  - TN-SHAP on the true teacher (multilinear extension via t)
  - TN-SHAP on the trained MPS surrogate

and report:
  - R² of MPS vs teacher on all path-augmented data
  - Correlation of Shapley on S
  - Support recovery accuracy (top-|S|)
"""

import argparse
import math
import torch
from torch.utils.data import DataLoader, TensorDataset

from torchmps import MPS
from poly_teacher import load_teacher, DEVICE

# -----------------------
# Small utilities
# -----------------------

def r2_score(y_true, y_pred) -> float:
    y_true = y_true.detach()
    y_pred = y_pred.detach()
    var = torch.var(y_true)
    if var < 1e-12:
        return 1.0 if torch.allclose(y_true, y_pred) else 0.0
    return float(1.0 - torch.mean((y_true - y_pred) ** 2) / (var + 1e-12))


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
    return t.to(dtype).to(device)


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
    V = t.unsqueeze(1) ** exponents.unsqueeze(0)  # (L, degree_max+1)
    return V


def solve_poly_coeffs(V: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Solve V c = y for the polynomial coefficients c.
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
    g_i(t) = f(x_i fixed, others scaled by t).
    """
    x = x.to(DEVICE)
    t_nodes = t_nodes.to(x.device)
    D = x.shape[0]

    X_scaled = t_nodes.unsqueeze(1) * x.unsqueeze(0)  # (L, D)
    X_scaled[:, i] = x[i]

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
      - g_i(t) = f(x_i fixed, others scaled by t) -> γ_{i,k}
      - Recurrence to get β_{i,k}
      - φ_i = sum_k β_{i,k} / k
    """
    x = x.to(DEVICE)
    D = x.shape[0]

    # Use the first max_degree+1 nodes to build a square Vandermonde
    assert t_nodes.shape[0] >= max_degree + 1
    t_nodes_sq = t_nodes[: max_degree + 1]
    V = build_vandermonde(t_nodes_sq, degree_max=max_degree)

    # 1) h(t) path
    h_vals = eval_h_on_nodes(eval_fn, x, t_nodes_sq)
    alpha = solve_poly_coeffs(V, h_vals)  # (max_degree+1,)
    alpha64 = alpha.to(torch.float64)

    phi_full = torch.zeros(D, device=DEVICE, dtype=torch.float32)

    # 2) g_i(t) paths, recurrence per feature
    for i in range(D):
        g_vals = eval_g_i_on_nodes(eval_fn, x, i, t_nodes_sq)
        gamma = solve_poly_coeffs(V, g_vals)
        gamma64 = gamma.to(torch.float64)

        beta = torch.zeros(max_degree + 1, dtype=torch.float64, device=x.device)

        # k = max_degree: γ_d = α_d - β_d  => β_d = α_d - γ_d
        beta[max_degree] = alpha64[max_degree] - gamma64[max_degree]

        # k = max_degree-1 ... 1: γ_k = α_k + β_{k+1} - β_k
        for k in range(max_degree - 1, 0, -1):
            beta[k] = alpha64[k] + beta[k + 1] - gamma64[k]

        # consistency for k=0: γ_0 = α_0 + β_1
        beta1_from_gamma0 = gamma64[0] - alpha64[0]
        beta[1] = 0.5 * (beta[1] + beta1_from_gamma0)

        ks = torch.arange(1, max_degree + 1, dtype=torch.float64, device=x.device)
        phi_i = torch.sum(beta[1:] / ks)
        phi_full[i] = phi_i.to(torch.float32)

    return phi_full


# -----------------------
# Main comparison
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="poly",
                        help="Prefix for *_teacher.pt, *_tnshap_targets.pt, *_mps.pt")
    parser.add_argument("--max-degree", type=int, default=None,
                        help="Override polynomial degree in t; "
                             "default = value stored in *_tnshap_targets.pt")
    parser.add_argument("--n-targets", type=int, default=None,
                        help="Optional: number of base points to evaluate TN-SHAP on.")
    parser.add_argument("--eval-batch-size", type=int, default=2048,
                        help="Batch size for function R² sanity check to avoid OOM.")
    args = parser.parse_args()

    prefix = args.prefix
    print(f"=== TN-SHAP Vandermonde Eval for PREFIX={prefix} ===")

    # ---------------------------------------------------------
    # 1) Load saved TN-SHAP targets + t-nodes + training dataset
    # ---------------------------------------------------------
    targets = torch.load(f"{prefix}_tnshap_targets.pt", map_location=DEVICE)
    X_targets = targets["x_base"].to(DEVICE)   # (N_base, D)
    t_nodes   = targets["t_nodes"].to(DEVICE)  # (N_T_NODES,)
    X_all     = targets["X_all"].to(DEVICE)    # path-augmented dataset
    Y_all     = targets["Y_all"].to(DEVICE)    # same length
    max_degree_saved = int(targets["max_degree"])

    if args.n_targets is not None:
        X_targets = X_targets[: args.n_targets]

    N_TARGETS = X_targets.shape[0]
    D         = X_targets.shape[1]

    max_degree = args.max_degree if args.max_degree is not None else max_degree_saved
    max_degree = int(max_degree)
    assert max_degree + 1 <= t_nodes.shape[0], \
        "t_nodes must contain at least max_degree+1 nodes."

    print(f"Using N_TARGETS={N_TARGETS}, D={D}, max_degree={max_degree}")

    # ---------------------------------------------------------
    # 2) Load teacher and MPS
    # ---------------------------------------------------------
    teacher = load_teacher(prefix)
    mps = load_mps(prefix)

    def eval_fn_teacher(x_batch: torch.Tensor) -> torch.Tensor:
        x_batch = x_batch.to(DEVICE)
        with torch.no_grad():
            y = teacher(x_batch)
            if y.ndim > 1:
                y = y.squeeze(-1)
        return y

    def eval_fn_mps(x_batch: torch.Tensor) -> torch.Tensor:
        x_batch = x_batch.to(DEVICE)
        x_aug = augment_with_one(x_batch)
        with torch.no_grad():
            y = mps(x_aug).squeeze(-1)
        return y

    # ---------------------------------------------------------
    # 3) Sanity: R2 of MPS vs TEACHER on ALL PATH-AUG DATA (batched, no OOM)
    # ---------------------------------------------------------
    eval_ds = TensorDataset(X_all, Y_all)  # Y_all is teacher(x_path)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
    )

    teacher_preds = []
    mps_preds = []
    y_true_list = []

    mps.eval()
    teacher.eval()
    with torch.no_grad():
        for xb, yb in eval_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            y_t = eval_fn_teacher(xb)
            y_m = eval_fn_mps(xb)

            teacher_preds.append(y_t)
            mps_preds.append(y_m)
            y_true_list.append(yb)

    Y_true_all      = torch.cat(y_true_list, dim=0)
    teacher_all_pred = torch.cat(teacher_preds, dim=0)
    mps_all_pred     = torch.cat(mps_preds, dim=0)

    r2_teacher = r2_score(Y_true_all, teacher_all_pred)
    r2_mps     = r2_score(Y_true_all, mps_all_pred)

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
        x0 = X_targets[idx]

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

        # Correlation on S
        s_t = phi_teacher[S]
        s_m = phi_mps[S]
        if torch.std(s_t) < 1e-8 or torch.std(s_m) < 1e-8:
            corr = float("nan")
        else:
            c = torch.corrcoef(torch.stack([s_t, s_m]))[0, 1]
            corr = float(c.item())
        corrs.append(corr)

        # Support accuracy (top-|S|)
        top_teacher = torch.topk(phi_teacher.abs(), k_active).indices.tolist()
        top_mps     = torch.topk(phi_mps.abs(),     k_active).indices.tolist()

        correct_teacher = len(set(top_teacher) & S_set)
        correct_mps     = len(set(top_mps)     & S_set)

        acc_teacher = correct_teacher / k_active
        acc_mps     = correct_mps     / k_active

        acc_teacher_list.append(acc_teacher)
        acc_mps_list.append(acc_mps)

        phi_norm_S = phi_teacher[S].abs()
        print(
            f"[Point {idx:02d}] corr={corr:.4f} | "
            f"acc teacher={acc_teacher:.2f}, MPS={acc_mps:.2f} | "
            f"|phi_teacher[S]| min={phi_norm_S.min().item():.3e}, "
            f"max={phi_norm_S.max().item():.3e}"
        )

    
    acc_teacher = torch.tensor(acc_teacher_list)
    acc_mps = torch.tensor(acc_mps_list)

    mean_acc_teacher = acc_teacher.mean().item()
    std_acc_teacher = acc_teacher.std(unbiased=True).item()

    mean_acc_mps = acc_mps.mean().item()
    std_acc_mps = acc_mps.std(unbiased=True).item()

    print("==== Shapley Support Accuracy on S ====")
    print(f"Teacher Mean Acc: {mean_acc_teacher:.4f}")
    print(f"Teacher Std:      {std_acc_teacher:.4f}")
    print(f"MPS Mean Acc:     {mean_acc_mps:.4f}")
    print(f"MPS Std:          {std_acc_mps:.4f}")



if __name__ == "__main__":
    main()
