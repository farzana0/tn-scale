#!/usr/bin/env python3
"""
tnshap_sqexp_vandermonde_eval_gi.py

TN-SHAP-style Shapley computation based on *Gi(t)* only:

  For each feature i and each interpolation node t_ℓ:
    G_i(t_ℓ; x) = f(x_on_i(t_ℓ)) - f(x_off_i(t_ℓ))

where:
  - x_on_i(t)[j]  = x[j]      if j == i, else t * x[j]
  - x_off_i(t)[j] = 0         if j == i, else t * x[j]

We then interpolate G_i(t; x) as a polynomial in t using a Vandermonde system
(or lstsq for stability) to recover coefficients m_s^(i), and approximate:

    Shapley_i(x) ≈ sum_{s=0}^{max_degree} α_s m_s^(i),

where α_s are the usual Shapley weights w.r.t. total feature dimension D.

This file is designed to be drop-in compatible with the previous
tnshap_sqexp_vandermonde_eval.py in your bash pipeline.
"""

import argparse
import os
import time
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


def shapley_weights(D: int, device=None, dtype=torch.float64) -> torch.Tensor:
    """
    Shapley weights α_s for s = 0,...,D-1:

        α_s = s!(D-s-1)! / D!
    """
    if device is None:
        device = DEVICE
    alphas = torch.empty(D, dtype=dtype, device=device)
    D_fact = math.factorial(D)
    for s in range(D):
        num = math.factorial(s) * math.factorial(D - s - 1)
        alphas[s] = num / D_fact
    return alphas


def build_vandermonde(t_nodes: torch.Tensor, degree_max: int) -> torch.Tensor:
    """
    V[l, r] = t_nodes[l] ** r  for r = 0,...,degree_max.

    t_nodes: shape [L]
    returns: [L, degree_max + 1]
    """
    t = t_nodes.to(dtype=torch.float64)
    exps = torch.arange(0, degree_max + 1, dtype=torch.float64, device=t.device)
    V = t.unsqueeze(1) ** exps.unsqueeze(0)  # [L, degree_max+1]
    return V


# -----------------------
# Load MPS with the right shape
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
        feature_dim=2,        # matches training (sqexp_x2exp_then_one)
        parallel_eval=True,
    ).to(DEVICE)

    # We rely on the feature map already being baked into the saved state
    # via register_feature_map at training time.
    mps.load_state_dict(state["state_dict"])
    mps.eval()

    return mps


# -----------------------
# Gi(t; x) evaluation utilities
# -----------------------

def eval_Gi_teacher(
    eval_fn,
    x: torch.Tensor,
    i: int,
    t_nodes: torch.Tensor,
) -> torch.Tensor:
    """
    Compute vector G_i(t_ℓ; x) for TEACHER, using paths:

        x_on_i(t)[j]  = x[j]    if j == i, else t * x[j]
        x_off_i(t)[j] = 0       if j == i, else t * x[j]

    eval_fn returns *scaled* outputs (matching training scaling).
    """
    x = x.to(DEVICE)
    t_nodes = t_nodes.to(x.device)
    D = x.shape[0]
    L = t_nodes.shape[0]

    # Base: t * x for all j
    base = t_nodes.unsqueeze(1) * x.unsqueeze(0)   # [L, D]

    # ON: clamp feature i to x_i (original value)
    X_on = base.clone()
    X_on[:, i] = x[i]

    # OFF: clamp feature i to baseline 0
    X_off = base.clone()
    X_off[:, i] = 0.0

    with torch.no_grad():
        y_on = eval_fn(X_on)         # [L]
        y_off = eval_fn(X_off)       # [L]

    h = y_on - y_off                # [L]
    return h


def eval_Gi_mps(
    mps: MPS,
    x: torch.Tensor,
    i: int,
    t_nodes: torch.Tensor,
) -> torch.Tensor:
    """
    Compute G_i(t_ℓ; x) for MPS surrogate with the same paths as above.
    The MPS was trained on ORIGINAL x-space; feature map is applied inside MPS.
    """
    x = x.to(DEVICE)
    t_nodes = t_nodes.to(x.device)
    D = x.shape[0]
    L = t_nodes.shape[0]

    base = t_nodes.unsqueeze(1) * x.unsqueeze(0)  # [L, D]

    X_on = base.clone()
    X_on[:, i] = x[i]

    X_off = base.clone()
    X_off[:, i] = 0.0

    with torch.no_grad():
        y_on = mps(X_on).squeeze(-1)   # [L]
        y_off = mps(X_off).squeeze(-1)

    h = y_on - y_off
    return h


# -----------------------
# TN-SHAP via Gi(t) + Vandermonde
# -----------------------

def tnshap_gi_teacher(
    eval_fn,
    x: torch.Tensor,
    max_degree: int,
    t_nodes: torch.Tensor,
) -> torch.Tensor:
    """
    TN-SHAP approximation for TEACHER via Gi(t;x).

    Returns:
        phi_teacher: [D] tensor of approximate Shapley values.
    """
    x = x.to(DEVICE)
    D = x.shape[0]

    # Use first max_degree+1 nodes for polynomial of degree ≤ max_degree
    assert t_nodes.shape[0] >= max_degree + 1, "Need at least max_degree+1 nodes."
    t_sub = t_nodes[: max_degree + 1]
    L = t_sub.shape[0]

    V = build_vandermonde(t_sub, degree_max=max_degree)  # [L, max_degree+1]
    alphas = shapley_weights(D, device=DEVICE, dtype=torch.float64)  # [D]

    phi = torch.zeros(D, device=DEVICE, dtype=torch.float32)

    for i in range(D):
        # 1) Collect Gi(t_ℓ; x) on these nodes
        h = eval_Gi_teacher(eval_fn, x, i, t_sub)        # [L]
        h64 = h.to(torch.float64)

        # 2) Solve V m^(i) ≈ h  (lstsq for stability)
        #    m_i: [max_degree+1] containing degree-wise coefficients
        m_i, *_ = torch.linalg.lstsq(V, h64.unsqueeze(-1))
        m_i = m_i.squeeze(-1)  # [max_degree+1]

        # 3) Approximate Shapley using α_s m_s, with s>max_degree treated as 0
        #    so we use α_s for s=0..max_degree and zeros thereafter.
        phi_i = torch.sum(alphas[: max_degree + 1] * m_i)
        phi[i] = phi_i.to(torch.float32)

    return phi


def tnshap_gi_mps(
    mps: MPS,
    x: torch.Tensor,
    max_degree: int,
    t_nodes: torch.Tensor,
) -> torch.Tensor:
    """
    TN-SHAP approximation for MPS surrogate via Gi(t;x).

    Returns:
        phi_mps: [D] tensor of approximate Shapley values.
    """
    x = x.to(DEVICE)
    D = x.shape[0]

    assert t_nodes.shape[0] >= max_degree + 1
    t_sub = t_nodes[: max_degree + 1]
    L = t_sub.shape[0]

    V = build_vandermonde(t_sub, degree_max=max_degree)  # [L, max_degree+1]
    alphas = shapley_weights(D, device=DEVICE, dtype=torch.float64)

    phi = torch.zeros(D, device=DEVICE, dtype=torch.float32)

    for i in range(D):
        h = eval_Gi_mps(mps, x, i, t_sub)   # [L]
        h64 = h.to(torch.float64)

        m_i, *_ = torch.linalg.lstsq(V, h64.unsqueeze(-1))
        m_i = m_i.squeeze(-1)  # [max_degree+1]

        phi_i = torch.sum(alphas[: max_degree + 1] * m_i)
        phi[i] = phi_i.to(torch.float32)

    return phi


# -----------------------
# Main pipeline
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
    print(f"=== TN-SHAP (Gi-based) sqexp Eval for PREFIX={prefix} ===")
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

    # eval_fn that returns *scaled* outputs so it matches MPS training targets
    def eval_fn_teacher_scaled(x_batch: torch.Tensor) -> torch.Tensor:
        x_batch = x_batch.to(DEVICE)
        with torch.no_grad():
            y = teacher(x_batch)
            if y.ndim > 1:
                y = y.squeeze(-1)
            y = y / Y_scale
        return y

    # 3) R² sanity on path-aug teacher data
    Y_all_scaled = Y_all / Y_scale
    teacher_pred_scaled = eval_fn_teacher_scaled(X_all)
    r2_teacher = r2_score(Y_all_scaled, teacher_pred_scaled)

    eval_ds = TensorDataset(X_all, Y_all_scaled)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
    )

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

    Y_true_all_scaled = torch.cat(y_true_list, dim=0)
    mps_all_pred = torch.cat(mps_preds, dim=0)
    r2_mps = r2_score(Y_true_all_scaled, mps_all_pred)

    print(f"Teacher vs path-aug ground truth: R2 = {r2_teacher:.4f}")
    print(f"MPS vs path-aug ground truth:     R2 = {r2_mps:.4f}")

    # 4) TN-SHAP via Gi-based method on TEACHER and MPS
    S = getattr(teacher, "S", list(range(D)))  # active subset if available
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

        # Teacher TN-SHAP (Gi-based)
        t0 = time.time()
        phi_teacher = tnshap_gi_teacher(
            eval_fn_teacher_scaled,
            x0,
            max_degree=max_degree,
            t_nodes=t_nodes,
        )
        t1 = time.time()
        t_teacher = t1 - t0

        # MPS TN-SHAP (Gi-based)
        t2 = time.time()
        phi_mps = tnshap_gi_mps(
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
        
        print(f"Top-{k_active} teacher indices: {top_teacher}")
        print(f"Top-{k_active} MPS indices:     {top_mps}")

        correct_teacher = len(set(top_teacher) & S_set)
        correct_mps     = len(set(top_mps)     & S_set)

        acc_teacher = correct_teacher / k_active
        acc_mps     = correct_mps     / k_active

        acc_teacher_list.append(acc_teacher)
        acc_mps_list.append(acc_mps)

        phi_norm_S = phi_teacher[S].abs()
        phi_norm_mps_S = phi_mps[S].abs()
        max_diff_S = (phi_teacher[S] - phi_mps[S]).abs().max().item()

        print(
            f"[Point {idx:02d}] corr={corr:.4f} | "
            f"acc teacher={acc_teacher:.2f}, MPS={acc_mps:.2f} | "
            f"|phi_teacher[S]| min={phi_norm_S.min().item():.3e}, "
            f"max={phi_norm_S.max().item():.3e} | "
            f"|phi_mps[S]| min={phi_norm_mps_S.min().item():.3e}, "
            f"max={phi_norm_mps_S.max().item():.3e} | "
            f"max|diff[S]|={max_diff_S:.3e} | "
            f"t_teacher={t_teacher*1000:.2f} ms, t_mps={t_mps*1000:.2f} ms"
        )

    acc_teacher_t = torch.tensor(acc_teacher_list, device=DEVICE)
    acc_mps_t     = torch.tensor(acc_mps_list, device=DEVICE)

    mean_acc_teacher = acc_teacher_t.mean().item()
    std_acc_teacher  = acc_teacher_t.std(unbiased=True).item() if len(acc_teacher_t) > 1 else 0.0

    mean_acc_mps = acc_mps_t.mean().item()
    std_acc_mps  = acc_mps_t.std(unbiased=True).item() if len(acc_mps_t) > 1 else 0.0

    print("==== Shapley Support Accuracy on S ====")
    print(f"Teacher Mean Acc: {mean_acc_teacher:.4f}")
    print(f"Teacher Std:      {std_acc_teacher:.4f}")
    print(f"MPS Mean Acc:     {mean_acc_mps:.4f}")
    print(f"MPS Std:          {std_acc_mps:.4f}")

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
