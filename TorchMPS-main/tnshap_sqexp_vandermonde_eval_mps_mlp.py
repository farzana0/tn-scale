#!/usr/bin/env python3
"""
tnshap_sqexp_vandermonde_eval_mps_mlp.py

TN-SHAP-style Shapley comparison for:

  - Exponential teacher f(x) in ORIGINAL space
  - MPS+MLP surrogate trained on path-augmented ORIGINAL data

We:

  - Load path-augmented dataset and TN-SHAP metadata from:
        <PREFIX>_tnshap_targets.pt
    which contains:
        x_base, t_nodes, X_all, Y_all, max_degree, feature_map tag

  - Load teacher:     load_teacher(prefix)
  - Load MPS+MLP:     from <PREFIX>_mps_mlp.pt

  - Compute R² of:
        teacher vs Y_all (sanity, ~1)
        MPS vs Y_all

  - For N_TARGETS base points x (from x_base):
        * Run TN-SHAP via Chebyshev t-nodes + Vandermonde
          on ORIGINAL x-space paths:
              h(t)   = f(t * x)
              g_i(t) = f(x_i fixed, t * x_{-i})
        * Compute Shapley for teacher (phi_teacher)
        * Compute Shapley for MPS+MLP (phi_mps)

  - If teacher has a support set S, report:
        * per-point correlation on S
        * per-point support accuracy (top-|S|) for teacher (sanity)
        * per-point support accuracy (top-|S|) for MPS

  - Report mean/std of:
        * R²
        * Shapley support accuracy
        * per-point TN-SHAP runtime for teacher and MPS

  - Save Shapley results to:
        <PREFIX>_sqexp_tnshap_mps_mlp_eval.pt
"""

import argparse
import time
import torch

from torch.utils.data import TensorDataset, DataLoader

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


def build_vandermonde(t_nodes: torch.Tensor, degree_max: int) -> torch.Tensor:
    """
    V[l, k] = t_nodes[l] ** k, for k = 0..degree_max and l = 0..L-1.
    """
    t = t_nodes.to(dtype=torch.float64)
    exponents = torch.arange(0, degree_max + 1,
                             dtype=torch.float64,
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


# -----------------------
# Diagonal evals (teacher & MPS) in ORIGINAL x-space
# -----------------------

def eval_h_on_nodes(eval_fn, x: torch.Tensor, t_nodes: torch.Tensor) -> torch.Tensor:
    """
    h(t) = f(t * x), generic eval_fn.
    """
    x = x.to(DEVICE)
    t_nodes = t_nodes.to(x.device)
    X_batch = t_nodes.unsqueeze(1) * x.unsqueeze(0)  # (L, D)
    y = eval_fn(X_batch)
    if y.ndim > 1:
        y = y.squeeze(-1)
    return y


def eval_g_i_on_nodes(eval_fn, x: torch.Tensor, i: int, t_nodes: torch.Tensor) -> torch.Tensor:
    """
    g_i(t) = f(x_i fixed, others scaled by t), generic eval_fn.
    """
    x = x.to(DEVICE)
    t_nodes = t_nodes.to(x.device)
    D = x.shape[0]

    X_scaled = t_nodes.unsqueeze(1) * x.unsqueeze(0)  # (L, D)
    X_scaled[:, i] = x[i]

    y = eval_fn(X_scaled)
    if y.ndim > 1:
        y = y.squeeze(-1)
    return y


# -----------------------
# TN-SHAP via Vandermonde + recurrence (generic eval_fn)
# -----------------------

def tnshap_vandermonde(
    eval_fn,
    x: torch.Tensor,
    max_degree: int,
    t_nodes: torch.Tensor,
) -> torch.Tensor:
    """
    TN-SHAP for an arbitrary function f using ORIGINAL x-space paths:

        h(t)   = f(t * x)
        g_i(t) = f(x_i fixed, t * x_{-i})

    Uses:
      - Chebyshev t_nodes (>= max_degree+1)
      - Square Vandermonde system to get polynomial coefficients
      - TN-SHAP recurrence to obtain β_{i,k}
      - Shapley_i = sum_{k=1..max_degree} β_{i,k} / k
    """
    x = x.to(DEVICE)
    D = x.shape[0]

    assert t_nodes.shape[0] >= max_degree + 1, \
        "Need at least max_degree+1 t-nodes."

    # Use exactly max_degree+1 nodes for a square Vandermonde.
    t_nodes_use = t_nodes[: max_degree + 1]
    V = build_vandermonde(t_nodes_use, degree_max=max_degree)

    # --- h(t) ---
    h_vals = eval_h_on_nodes(eval_fn, x, t_nodes_use)
    alpha = solve_poly_coeffs(V, h_vals)
    alpha64 = alpha.to(torch.float64)

    phi_full = torch.zeros(D, device=DEVICE, dtype=torch.float32)

    # --- per-feature recurrence over g_i(t) ---
    for i in range(D):
        g_vals = eval_g_i_on_nodes(eval_fn, x, i, t_nodes_use)
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

        # consistency k = 0:
        #   gamma[0] = alpha[0] + beta[1]  =>  beta[1] = gamma[0] - alpha[0]
        beta1_from_gamma0 = gamma64[0] - alpha64[0]
        beta[1] = 0.5 * (beta[1] + beta1_from_gamma0)

        ks = torch.arange(1, max_degree + 1,
                          dtype=torch.float64,
                          device=x.device)
        phi_i = torch.sum(beta[1:] / ks)
        phi_full[i] = phi_i.to(torch.float32)

    return phi_full


# -----------------------
# LocalMLPFeatureMap (must match training)
# -----------------------

class LocalMLPFeatureMap(torch.nn.Module):
    """
    Per-site MLP feature map used in training:

        phi(s) : R -> R^{feature_dim}
    """
    def __init__(self, feature_dim: int = 4, hidden_dim: int = 16):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (...,)
        x_in = x.unsqueeze(-1)  # (..., 1)
        out = self.net(x_in)   # (..., feature_dim)
        return out


def load_mps_mlp(prefix: str):
    """
    Load the MPS+MLP surrogate from <prefix>_mps_mlp.pt
    """
    state = torch.load(f"{prefix}_mps_mlp.pt", map_location=DEVICE)
    input_dim = state["input_dim"]
    bond_dim = state["bond_dim"]
    feature_dim = state["feature_dim"]
    hidden_dim = state["hidden_dim"]

    mps = MPS(
        input_dim=input_dim,
        output_dim=1,
        bond_dim=bond_dim,
        adaptive_mode=False,
        periodic_bc=False,
        feature_dim=feature_dim,
        parallel_eval=True,
    )

    mlp_feature = LocalMLPFeatureMap(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
    )

    mps.register_feature_map(mlp_feature)
    mps.to(DEVICE)
    mps.load_state_dict(state["state_dict"])
    mps.eval()

    return mps


# -----------------------
# Main comparison
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix", type=str, default="sqexp_D50",
        help="Prefix for *_tnshap_targets.pt and *_mps_mlp.pt"
    )
    parser.add_argument(
        "--max-degree", type=int, default=None,
        help="Override polynomial degree in t; "
             "default = value stored in *_tnshap_targets.pt"
    )
    parser.add_argument(
        "--n-targets", type=int, default=None,
        help="Number of base points from x_base to evaluate."
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=4096,
        help="Batch size for R² sanity checks."
    )
    args = parser.parse_args()

    prefix = args.prefix
    print(f"=== TN-SHAP sqexp Vandermonde Eval (Teacher vs MPS+MLP) for PREFIX={prefix} ===")

    # 1) Load TN-SHAP targets + path-aug data
    targets_path = f"{prefix}_tnshap_targets.pt"
    print(f"Loading TN-SHAP targets from {targets_path}")
    targets = torch.load(targets_path, map_location=DEVICE)

    X_targets = targets["x_base"].to(DEVICE)   # (N_base, D)
    t_nodes   = targets["t_nodes"].to(DEVICE)  # (K,)
    X_all     = targets["X_all"].to(DEVICE)    # (N_all, D)
    Y_all     = targets["Y_all"].to(DEVICE)    # (N_all,)
    max_degree_saved = int(targets["max_degree"])
    fmap_name_targets = targets.get("feature_map", "sqexp_x2exp_then_one")

    if args.n_targets is not None:
        X_targets = X_targets[: args.n_targets]

    N_TARGETS = X_targets.shape[0]
    D         = X_targets.shape[1]

    # degree in t
    max_degree = args.max_degree if args.max_degree is not None else max_degree_saved
    max_degree = int(max_degree)
    assert max_degree + 1 <= t_nodes.shape[0], \
        "t_nodes must contain at least max_degree+1 nodes."

    print(f"Using N_TARGETS={N_TARGETS}, D={D}, max_degree={max_degree}")
    print(f"feature_map (targets metadata): {fmap_name_targets}")

    # 2) Load teacher & MPS+MLP
    teacher = load_teacher(prefix)
    if isinstance(teacher, torch.nn.Module):
        teacher.to(DEVICE)
    mps = load_mps_mlp(prefix)

    def eval_fn_teacher(x_batch: torch.Tensor) -> torch.Tensor:
        x_batch = x_batch.to(DEVICE)
        with torch.no_grad():
            y = teacher(x_batch)
            if y.ndim > 1:
                y = y.squeeze(-1)
        return y

    def eval_fn_mps(x_batch: torch.Tensor) -> torch.Tensor:
        x_batch = x_batch.to(DEVICE)
        with torch.no_grad():
            y = mps(x_batch).squeeze(-1)
        return y

    # 3) R² sanity: teacher & MPS vs Y_all (path-aug ground truth)
    print("\n=== R² sanity on full path-aug dataset ===")
    with torch.no_grad():
        teacher_pred_all = eval_fn_teacher(X_all)
        mps_pred_all = eval_fn_mps(X_all)

    r2_teacher = r2_score(Y_all, teacher_pred_all)
    r2_mps = r2_score(Y_all, mps_pred_all)

    print(f"Teacher vs ground truth: R² = {r2_teacher:.6f}")
    print(f"MPS+MLP vs ground truth: R² = {r2_mps:.6f}\n")

    # 4) TN-SHAP Shapley eval per point
    has_support = hasattr(teacher, "S")
    if has_support:
        S = list(teacher.S)
        S_set = set(S)
        k_active = len(S)
        print(f"Teacher has support S (size {k_active}): {S}")
    else:
        print("Teacher has no attribute 'S'; support accuracy will be skipped.")
        S, S_set, k_active = None, None, None

    acc_teacher_list = []
    acc_mps_list     = []
    corrs            = []
    t_teacher_list   = []
    t_mps_list       = []

    shap_teacher_all = torch.zeros(N_TARGETS, D, device=DEVICE, dtype=torch.float32)
    shap_mps_all     = torch.zeros_like(shap_teacher_all)

    for idx in range(N_TARGETS):
        x0 = X_targets[idx]

        # Teacher TN-SHAP
        t0 = time.time()
        phi_teacher = tnshap_vandermonde(
            eval_fn_teacher,
            x0,
            max_degree=max_degree,
            t_nodes=t_nodes,
        )
        t1 = time.time()
        t_teacher = t1 - t0

        # MPS TN-SHAP
        t2 = time.time()
        phi_mps = tnshap_vandermonde(
            eval_fn_mps,
            x0,
            max_degree=max_degree,
            t_nodes=t_nodes,
        )
        t3 = time.time()
        t_mps = t3 - t2

        shap_teacher_all[idx] = phi_teacher
        shap_mps_all[idx]     = phi_mps

        t_teacher_list.append(t_teacher)
        t_mps_list.append(t_mps)

        if has_support:
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

            print(
                f"[Point {idx:02d}] corr={corr:.4f} | "
                f"acc teacher={acc_teacher:.2f}, MPS={acc_mps:.2f} | "
                f"t_teacher={t_teacher*1000:.2f} ms, t_mps={t_mps*1000:.2f} ms"
            )
        else:
            print(
                f"[Point {idx:02d}] "
                f"t_teacher={t_teacher*1000:.2f} ms, t_mps={t_mps*1000:.2f} ms"
            )

    # 5) Aggregate metrics
    t_teacher_tensor = torch.tensor(t_teacher_list, device=DEVICE)
    t_mps_tensor     = torch.tensor(t_mps_list, device=DEVICE)

    mean_t_teacher = t_teacher_tensor.mean().item()
    std_t_teacher  = t_teacher_tensor.std(unbiased=True).item() if len(t_teacher_tensor) > 1 else 0.0

    mean_t_mps = t_mps_tensor.mean().item()
    std_t_mps  = t_mps_tensor.std(unbiased=True).item() if len(t_mps_tensor) > 1 else 0.0

    print("\n==== Per-point TN-SHAP evaluation time ====")
    print(f"Teacher TN-SHAP: mean = {mean_t_teacher*1000:.2f} ms, std = {std_t_teacher*1000:.2f} ms")
    print(f"MPS TN-SHAP:     mean = {mean_t_mps*1000:.2f} ms, std = {std_t_mps*1000:.2f} ms")

    if has_support and acc_teacher_list:
        acc_teacher_tensor = torch.tensor(acc_teacher_list, device=DEVICE)
        acc_mps_tensor     = torch.tensor(acc_mps_list, device=DEVICE)

        mean_acc_teacher = acc_teacher_tensor.mean().item()
        std_acc_teacher  = acc_teacher_tensor.std(unbiased=True).item() if len(acc_teacher_tensor) > 1 else 0.0

        mean_acc_mps = acc_mps_tensor.mean().item()
        std_acc_mps  = acc_mps_tensor.std(unbiased=True).item() if len(acc_mps_tensor) > 1 else 0.0

        print("\n==== Shapley Support Accuracy on S ====")
        print(f"Teacher Mean Acc: {mean_acc_teacher:.4f}, Std: {std_acc_teacher:.4f}")
        print(f"MPS Mean Acc:     {mean_acc_mps:.4f}, Std: {std_acc_mps:.4f}")

        # Backward-compatible nanmean/std for corrs
        corr_tensor = torch.tensor(corrs, device=DEVICE)
        valid = ~torch.isnan(corr_tensor)
        if valid.any():
            mean_corr = corr_tensor[valid].mean().item()
            std_corr = corr_tensor[valid].std(unbiased=True).item() if valid.sum() > 1 else 0.0
        else:
            mean_corr, std_corr = float("nan"), float("nan")

        print("==== Correlation (phi_teacher[S], phi_mps[S]) ====")
        print(f"mean corr = {mean_corr:.4f}, std corr = {std_corr:.4f}")

    # 6) Save results
    out_path = f"{prefix}_sqexp_tnshap_mps_mlp_eval.pt"
    torch.save(
        {
            "x_targets": X_targets.detach().cpu(),
            "shap_teacher": shap_teacher_all.detach().cpu(),
            "shap_mps": shap_mps_all.detach().cpu(),
            "t_nodes": t_nodes.detach().cpu(),
            "max_degree": max_degree,
            "r2_teacher_vs_paths": r2_teacher,
            "r2_mps_vs_paths": r2_mps,
            "has_support": hasattr(teacher, "S"),
            "support": teacher.S if hasattr(teacher, "S") else None,
            "t_teacher_ms": t_teacher_tensor.cpu() * 1000.0,
            "t_mps_ms": t_mps_tensor.cpu() * 1000.0,
        },
        out_path,
    )
    print(f"\nSaved TN-SHAP teacher & MPS+MLP Shapley results to {out_path}")


if __name__ == "__main__":
    main()
