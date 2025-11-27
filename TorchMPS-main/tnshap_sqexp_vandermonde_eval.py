#!/usr/bin/env python3
"""
tnshap_sqexp_vandermonde_eval.py

TN-SHAP-style Shapley computation for the *exponential* teacher and its MPS surrogate:

  - Diagonal evaluations in scalar parameter t
  - Chebyshev interpolation nodes in [0, 1]
  - Vandermonde system to recover degree-wise aggregates
  - Recurrence to extract per-feature degree aggregates β_{i,k}
  - Shapley_i = sum_k β_{i,k} / k

We reuse the exact path points built in train_mps_sqexp_paths.py:

  expo/<prefix>_tnshap_targets.pt

so the same (x_base, t_nodes, X_all, Y_all) are used for both training and evaluation.
"""

import argparse
import os
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


# Local feature map for the *exponential* case:
#   phi(s_i) = [exp(s_i^2), 1]

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
    """Solve V c = y for polynomial coefficients c (square Vandermonde)."""
    V64 = V.to(dtype=torch.float64)
    y64 = y.to(dtype=torch.float64)
    c = torch.linalg.lstsq(V64, y64, rcond=None).solution
    return c.to(dtype=torch.float32)


# -----------------------
# Diagonal evals in ORIGINAL x-space (teacher)
# -----------------------

def eval_h_on_nodes_teacher(eval_fn, x: torch.Tensor, t_nodes: torch.Tensor) -> torch.Tensor:
    """
    h(t) = f(t * x) for TEACHER, evaluated on all t_nodes in ORIGINAL space.
    eval_fn is already the *scaled* teacher (divided by Y_scale).
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
    eval_fn is already the *scaled* teacher (divided by Y_scale).
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
      - MPS internally applies its registered feature map
        (here: phi(s_i) = [exp(s_i^2), 1]).
      - Outputs are already in the *scaled* space the MPS was trained on.
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
      - Outputs are already in the *scaled* space.
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

def load_mps(prefix: str, expo_dir: str = "expo"):
    path = os.path.join(expo_dir, f"{prefix}_mps.pt")
    state = torch.load(path, map_location=DEVICE)
    input_dim = state["input_dim"]
    bond_dim = state["bond_dim"]

    mps = MPS(
        input_dim=input_dim,
        output_dim=1,
        bond_dim=bond_dim,
        adaptive_mode=False,
        periodic_bc=False,
        feature_dim=2,
        parallel_eval=True,
    ).to(DEVICE)
    # IMPORTANT: register the same feature map used in training.
    mps.load_state_dict(state["state_dict"])
    mps.eval()

    return mps


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

    eval_fn should be the *scaled* teacher (teacher / Y_scale),
    so the resulting Shapley values are also in the scaled units.
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

    The MPS outputs are in the same *scaled* space as the teacher used for training.
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
    parser.add_argument("--prefix", type=str, default="sqexp_D50",
                        help="Prefix for *_teacher.pt and expo/<prefix>_*.pt")
    parser.add_argument("--max-degree", type=int, default=None,
                        help="Override polynomial degree in t; "
                             "default = value stored in tnshap_targets file.")
    parser.add_argument("--n-targets", type=int, default=None,
                        help="Optional: number of base points to evaluate TN-SHAP on.")
    parser.add_argument("--eval-batch-size", type=int, default=4096,
                        help="Batch size for R² sanity check.")
    parser.add_argument("--expo-dir", type=str, default="expo",
                        help="Directory where tnshap_targets and mps are stored.")
    args = parser.parse_args()

    prefix = args.prefix
    expo_dir = args.expo_dir
    print(f"=== TN-SHAP sqexp Vandermonde Eval for PREFIX={prefix} ===")
    print(f"Loading artifacts from expo_dir={expo_dir}")

    # 1) Load TN-SHAP targets + t-nodes + path-aug metadata
    tn_path = os.path.join(expo_dir, f"{prefix}_tnshap_targets.pt")
    targets = torch.load(tn_path, map_location=DEVICE)
    X_targets = targets["x_base"].to(DEVICE)   # (N_base, D)
    t_nodes   = targets["t_nodes"].to(DEVICE)  # (N_T_NODES,)
    X_all     = targets["X_all"].to(DEVICE)    # (N_all, D)
    Y_all     = targets["Y_all"].to(DEVICE)    # (N_all,)
    max_degree_saved = int(targets["max_degree"])

    Y_scale = targets.get("Y_scale", None)
    if Y_scale is None:
        # Backward compatibility: if no scale stored, use 1.0 (no scaling)
        Y_scale = torch.tensor(1.0, device=DEVICE, dtype=Y_all.dtype)
    else:
        Y_scale = Y_scale.to(DEVICE).to(Y_all.dtype)

    print(f"Using Y_scale = {Y_scale.item():.3e} for teacher/MPS scaling")

    if args.n_targets is not None:
        X_targets = X_targets[: args.n_targets]

    N_TARGETS = X_targets.shape[0]
    D         = X_targets.shape[1]

    max_degree = args.max_degree if args.max_degree is not None else max_degree_saved
    max_degree = int(max_degree)
    assert max_degree + 1 <= t_nodes.shape[0], \
        "t_nodes must contain at least max_degree+1 nodes."

    print(f"Using N_TARGETS={N_TARGETS}, D={D}, max_degree={max_degree}")

    # 2) Load teacher and MPS
    teacher = load_teacher(prefix)
    if isinstance(teacher, torch.nn.Module):
        teacher.to(DEVICE)
    mps = load_mps(prefix, expo_dir=expo_dir)

    def eval_fn_teacher_scaled(x_batch: torch.Tensor) -> torch.Tensor:
        """
        Teacher evaluated and divided by Y_scale so it matches
        the function the MPS was trained on.
        """
        x_batch = x_batch.to(DEVICE)
        with torch.no_grad():
            y = teacher(x_batch)
            if y.ndim > 1:
                y = y.squeeze(-1)
            y = y / Y_scale
        return y

    # 3) R² sanity on path-aug teacher data (scaled)
    Y_all_scaled = Y_all / Y_scale
    teacher_pred_all_scaled = eval_fn_teacher_scaled(X_all)
    r2_teacher = r2_score(Y_all_scaled, teacher_pred_all_scaled)

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
            # MPS outputs scaled values
            y_m = mps(xb).squeeze(-1)
            mps_preds.append(y_m)
            y_true_list.append(yb / Y_scale)   # scale teacher labels

    Y_true_all_scaled = torch.cat(y_true_list, dim=0)
    mps_all_pred = torch.cat(mps_preds, dim=0)

    r2_mps = r2_score(Y_true_all_scaled, mps_all_pred)

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

        # Time teacher TN-SHAP (scaled)
        t0 = time.time()
        phi_teacher = tnshap_vandermonde_teacher(
            eval_fn_teacher_scaled,
            x0,
            max_degree=max_degree,
            t_nodes=t_nodes,
        )
        t1 = time.time()
        t_teacher = t1 - t0

        # Time MPS TN-SHAP (scaled)
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

        # Correlation on S (using scaled values)
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
        print(top_teacher)
        print(top_mps)

        correct_teacher = len(set(top_teacher) & S_set)
        correct_mps     = len(set(top_mps)     & S_set)

        acc_teacher = correct_teacher / k_active
        acc_mps     = correct_mps     / k_active

        acc_teacher_list.append(acc_teacher)
        acc_mps_list.append(acc_mps)

        # NEW: diagnostics on magnitudes and mismatch over S
        phi_norm_S_teacher = phi_teacher[S].abs()
        phi_norm_S_mps     = phi_mps[S].abs()
        diff_S             = (phi_mps[S] - phi_teacher[S]).abs()

        print(
            f"[Point {idx:02d}] corr={corr:.4f} | "
            f"acc teacher={acc_teacher:.2f}, MPS={acc_mps:.2f} | "
            f"|phi_teacher[S]| min={phi_norm_S_teacher.min().item():.3e}, "
            f"max={phi_norm_S_teacher.max().item():.3e} | "
            f"|phi_mps[S]| min={phi_norm_S_mps.min().item():.3e}, "
            f"max={phi_norm_S_mps.max().item():.3e} | "
            f"max|diff[S]|={diff_S.max().item():.3e} | "
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
