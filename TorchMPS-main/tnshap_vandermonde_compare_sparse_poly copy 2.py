#!/usr/bin/env python3
"""
tnshap_vandermonde_compare_sparse_poly.py

TN-SHAP-style Shapley computation using:

  - Diagonal evaluations in a scalar parameter t
  - Chebyshev interpolation nodes in [0, 1]
  - Vandermonde system to recover degree-wise aggregates
  - A recurrence to extract per-feature degree aggregates β_{i,k}
  - Shapley_i = sum_k β_{i,k} / k

We compare:

  - TN-SHAP on the true teacher (multilinear extension via t in ORIGINAL x-space)
  - TN-SHAP on the trained MPS surrogate, evaluated on the SAME ORIGINAL paths
    (t * x and clamped g_i(t)), while the MPS internally applies the feature map
    phi(s_j) = [s_j, 1]

and report:
  - R² of MPS vs teacher on the path-aug ground truth (same path points as training)
  - Correlation of Shapley on S
  - Support recovery accuracy (top-|S|)
  - Per-point TN-SHAP evaluation times (teacher & MPS)
"""

import argparse
import time
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


# Local feature map: phi(s_j) = [s_j, 1]
def local_feature_map(x: torch.Tensor) -> torch.Tensor:
    ones = torch.ones_like(x)
    return torch.stack([x, ones], dim=-1)


# -----------------------
# Vandermonde utilities
# -----------------------

def build_vandermonde(t_nodes: torch.Tensor, degree_max: int) -> torch.Tensor:
    """
    V[l, k] = t_nodes[l] ** k, for k = 0..degree_max and l = 0..n_nodes-1.
    """
    t = t_nodes.to(dtype=torch.float64)
    exponents = torch.arange(0, degree_max + 1, dtype=torch.float64, device=t.device)
    V = t.unsqueeze(1) ** exponents.unsqueeze(0)  # (L, degree_max+1)
    return V


def solve_poly_coeffs(V: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Solve V c = y for polynomial coefficients c."""
    V64 = V.to(dtype=torch.float64)
    y64 = y.to(dtype=torch.float64)
    c = torch.linalg.solve(V64, y64)
    return c.to(dtype=torch.float32)


# -----------------------
# Diagonal evals in ORIGINAL x-space (teacher)
# -----------------------

def eval_h_on_nodes_teacher(eval_fn, x: torch.Tensor, t_nodes: torch.Tensor) -> torch.Tensor:
    """
    h(t) = f(t * x) for TEACHER, evaluated on all t_nodes in ORIGINAL space.
    """
    x = x.to(DEVICE)
    t_nodes = t_nodes.to(x.device)
    X_batch = (t_nodes.unsqueeze(1) * x.unsqueeze(0))  # (L, D)
    with torch.no_grad():
        y = eval_fn(X_batch)
        if y.ndim > 1:
            y = y.squeeze(-1)
    return y


def eval_g_i_on_nodes_teacher(eval_fn, x: torch.Tensor, i: int, t_nodes: torch.Tensor) -> torch.Tensor:
    """
    g_i(t) = f(x_i fixed, others scaled by t) for TEACHER in ORIGINAL space.
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
# Diagonal evals in ORIGINAL x-space for MPS
# -----------------------

def eval_h_on_nodes_mps(mps: MPS, x: torch.Tensor, t_nodes: torch.Tensor) -> torch.Tensor:
    """
    h(t) for MPS in ORIGINAL space:
      - input to MPS is s = t * x
      - MPS internally applies phi(s_j) = [s_j, 1]
    """
    x = x.to(DEVICE)
    t_nodes = t_nodes.to(x.device)

    X_batch = (t_nodes.unsqueeze(1) * x.unsqueeze(0))  # (L, D)
    with torch.no_grad():
        y = mps(X_batch).squeeze(-1)
    return y


def eval_g_i_on_nodes_mps(mps: MPS, x: torch.Tensor, i: int, t_nodes: torch.Tensor) -> torch.Tensor:
    """
    g_i(t) for MPS in ORIGINAL space:
      - s_j(t) = t * x_j for j != i
      - s_i(t) = x_i  (clamped)
    """
    x = x.to(DEVICE)
    t_nodes = t_nodes.to(x.device)
    D = x.shape[0]

    X_scaled = t_nodes.unsqueeze(1) * x.unsqueeze(0)  # (L, D)
    X_scaled[:, i] = x[i]

    with torch.no_grad():
        y = mps(X_scaled).squeeze(-1)
    return y


# -----------------------
# Load MPS with correct feature map
# -----------------------

def load_mps(prefix: str):
    state = torch.load(f"{prefix}_mps.pt", map_location=DEVICE)
    input_dim = state["input_dim"]
    bond_dim = state["bond_dim"]
    fmap_name = state.get("feature_map", "local_x_then_one")

    mps = MPS(
        input_dim=input_dim,
        output_dim=1,
        bond_dim=bond_dim,
        adaptive_mode=False,
        periodic_bc=False,
        feature_dim=2,
    ).to(DEVICE)
    mps.register_feature_map(local_feature_map)
    mps.load_state_dict(state["state_dict"])
    mps.eval()

    return mps, fmap_name


# -----------------------
# TN-SHAP via degree aggregates (Vandermonde + recurrence)
# -----------------------

def tnshap_vandermonde_teacher(
    eval_fn,
    x: torch.Tensor,
    max_degree: int,
    t_nodes: torch.Tensor,
) -> torch.Tensor:
    """
    TN-SHAP for the TEACHER, with paths in ORIGINAL x-space: t * x.
    """
    x = x.to(DEVICE)
    D = x.shape[0]

    assert t_nodes.shape[0] >= max_degree + 1
    t_nodes_sq = t_nodes[: max_degree + 1]
    V = build_vandermonde(t_nodes_sq, degree_max=max_degree)

    # h(t)
    h_vals = eval_h_on_nodes_teacher(eval_fn, x, t_nodes_sq)
    alpha = solve_poly_coeffs(V, h_vals)
    alpha64 = alpha.to(torch.float64)

    phi_full = torch.zeros(D, device=DEVICE, dtype=torch.float32)

    for i in range(D):
        g_vals = eval_g_i_on_nodes_teacher(eval_fn, x, i, t_nodes_sq)
        gamma = solve_poly_coeffs(V, g_vals)
        gamma64 = gamma.to(torch.float64)

        beta = torch.zeros(max_degree + 1, dtype=torch.float64, device=x.device)

        # k = max_degree
        beta[max_degree] = alpha64[max_degree] - gamma64[max_degree]

        # k = max_degree-1 ... 1
        for k in range(max_degree - 1, 0, -1):
            beta[k] = alpha64[k] + beta[k + 1] - gamma64[k]

        # consistency k=0
        beta1_from_gamma0 = gamma64[0] - alpha64[0]
        beta[1] = 0.5 * (beta[1] + beta1_from_gamma0)

        ks = torch.arange(1, max_degree + 1, dtype=torch.float64, device=x.device)
        phi_i = torch.sum(beta[1:] / ks)
        phi_full[i] = phi_i.to(torch.float32)

    return phi_full


def tnshap_vandermonde_mps(
    mps: MPS,
    x: torch.Tensor,
    max_degree: int,
    t_nodes: torch.Tensor,
) -> torch.Tensor:
    """
    TN-SHAP for the MPS surrogate, with paths in ORIGINAL x-space:
      - h(t): MPS(t * x)
      - g_i(t): MPS(x_i fixed, t * x_{-i})
    """
    x = x.to(DEVICE)
    D = x.shape[0]

    assert t_nodes.shape[0] >= max_degree + 1
    t_nodes_sq = t_nodes[: max_degree + 1]
    V = build_vandermonde(t_nodes_sq, degree_max=max_degree)

    # h(t)
    h_vals = eval_h_on_nodes_mps(mps, x, t_nodes_sq)
    alpha = solve_poly_coeffs(V, h_vals)
    alpha64 = alpha.to(torch.float64)

    phi_full = torch.zeros(D, device=DEVICE, dtype=torch.float32)

    for i in range(D):
        g_vals = eval_g_i_on_nodes_mps(mps, x, i, t_nodes_sq)
        gamma = solve_poly_coeffs(V, g_vals)
        gamma64 = gamma.to(torch.float64)

        beta = torch.zeros(max_degree + 1, dtype=torch.float64, device=x.device)

        # k = max_degree
        beta[max_degree] = alpha64[max_degree] - gamma64[max_degree]

        # k = max_degree-1 ... 1
        for k in range(max_degree - 1, 0, -1):
            beta[k] = alpha64[k] + beta[k + 1] - gamma64[k]

        # consistency k=0
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
                        help="Batch size for R² sanity check to avoid OOM.")
    args = parser.parse_args()

    prefix = args.prefix
    print(f"=== TN-SHAP Vandermonde Eval for PREFIX={prefix} ===")

    # 1) Load TN-SHAP targets + t-nodes + path-aug metadata
    targets = torch.load(f"{prefix}_tnshap_targets.pt", map_location=DEVICE)
    X_targets = targets["x_base"].to(DEVICE)   # (N_base, D)
    t_nodes   = targets["t_nodes"].to(DEVICE)  # (N_T_NODES,)
    X_all     = targets["X_all"].to(DEVICE)    # (N_all, D)
    Y_all     = targets["Y_all"].to(DEVICE)    # (N_all,)
    max_degree_saved = int(targets["max_degree"])
    fmap_name_targets = targets.get("feature_map", "local_x_then_one")

    if args.n_targets is not None:
        X_targets = X_targets[: args.n_targets]

    N_TARGETS = X_targets.shape[0]
    D         = X_targets.shape[1]

    max_degree = args.max_degree if args.max_degree is not None else max_degree_saved
    max_degree = int(max_degree)
    assert max_degree + 1 <= t_nodes.shape[0], \
        "t_nodes must contain at least max_degree+1 nodes."

    print(f"Using N_TARGETS={N_TARGETS}, D={D}, max_degree={max_degree}")
    print(f"feature_map (targets metadata): {fmap_name_targets}")

    # 2) Load teacher and MPS
    teacher = load_teacher(prefix)
    if isinstance(teacher, torch.nn.Module):
        teacher.to(DEVICE)
    mps, fmap_name_mps = load_mps(prefix)

    if fmap_name_mps != fmap_name_targets:
        print(f"[WARN] feature_map mismatch: tnshap_targets={fmap_name_targets}, mps={fmap_name_mps}")

    def eval_fn_teacher(x_batch: torch.Tensor) -> torch.Tensor:
        x_batch = x_batch.to(DEVICE)
        with torch.no_grad():
            y = teacher(x_batch)
            if y.ndim > 1:
                y = y.squeeze(-1)
        return y

    # 3) R² sanity on path-aug teacher data, in ORIGINAL space
    teacher_pred_all = eval_fn_teacher(X_all)
    r2_teacher = r2_score(Y_all, teacher_pred_all)

    eval_ds = TensorDataset(X_all, Y_all)
    eval_loader = DataLoader(eval_ds, batch_size=args.eval_batch_size,
                             shuffle=False, drop_last=False)

    mps_preds = []
    y_true_list = []

    mps.eval()
    with torch.no_grad():
        for xb, yb in eval_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            y_m = mps(xb).squeeze(-1)
            mps_preds.append(y_m)
            y_true_list.append(yb)

    Y_true_all = torch.cat(y_true_list, dim=0)
    mps_all_pred = torch.cat(mps_preds, dim=0)

    r2_mps = r2_score(Y_true_all, mps_all_pred)

    print(f"Teacher vs path-aug ground truth: R2 = {r2_teacher:.4f}")
    print(f"MPS vs path-aug ground truth:     R2 = {r2_mps:.4f}")

    # 4) Shapley via TN-SHAP on TEACHER and MPS + per-point timing
    S = teacher.S
    print(f"Active subset S (teacher support): {S}")
    k_active = len(S)
    S_set = set(S)

    acc_teacher_list = []
    acc_mps_list     = []
    corrs            = []
    t_teacher_list   = []
    t_mps_list       = []

    for idx in range(N_TARGETS):
        x0 = X_targets[idx]

        # Time teacher TN-SHAP
        t0 = time.time()
        phi_teacher = tnshap_vandermonde_teacher(
            eval_fn_teacher,
            x0,
            max_degree=max_degree,
            t_nodes=t_nodes,
        )
        t1 = time.time()
        t_teacher = t1 - t0

        # Time MPS TN-SHAP
        t2 = time.time()
        phi_mps = tnshap_vandermonde_mps(
            mps,
            x0,
            max_degree=max_degree,
            t_nodes=t_nodes,
        )
        t3 = time.time()
        t_mps = t3 - t2

        t_teacher_list.append(t_teacher)
        t_mps_list.append(t_mps)

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
            f"max={phi_norm_S.max().item():.3e} | "
            f"t_teacher={t_teacher*1000:.2f} ms, t_mps={t_mps*1000:.2f} ms"
        )

    acc_teacher = torch.tensor(acc_teacher_list, device=DEVICE)
    acc_mps     = torch.tensor(acc_mps_list, device=DEVICE)

    mean_acc_teacher = acc_teacher.mean().item()
    std_acc_teacher  = acc_teacher.std(unbiased=True).item() if len(acc_teacher) > 1 else 0.0

    mean_acc_mps = acc_mps.mean().item()
    std_acc_mps  = acc_mps.std(unbiased=True).item() if len(acc_mps) > 1 else 0.0

    print("==== Shapley Support Accuracy on S ====")
    print(f"Teacher Mean Acc: {mean_acc_teacher:.4f}")
    print(f"Teacher Std:      {std_acc_teacher:.4f}")
    print(f"MPS Mean Acc:     {mean_acc_mps:.4f}")
    print(f"MPS Std:          {std_acc_mps:.4f}")

    # Per-point runtime stats
    t_teacher_tensor = torch.tensor(t_teacher_list, device=DEVICE)
    t_mps_tensor     = torch.tensor(t_mps_list, device=DEVICE)

    mean_t_teacher = t_teacher_tensor.mean().item()
    std_t_teacher  = t_teacher_tensor.std(unbiased=True).item() if len(t_teacher_tensor) > 1 else 0.0

    mean_t_mps = t_mps_tensor.mean().item()
    std_t_mps  = t_mps_tensor.std(unbiased=True).item() if len(t_mps_tensor) > 1 else 0.0

    print("==== Per-point TN-SHAP evaluation time ====")
    print(f"Teacher TN-SHAP: mean = {mean_t_teacher*1000:.2f} ms, std = {std_t_teacher*1000:.2f} ms")
    print(f"MPS TN-SHAP:     mean = {mean_t_mps*1000:.2f} ms, std = {std_t_mps*1000:.2f} ms")


if __name__ == "__main__":
    main()
