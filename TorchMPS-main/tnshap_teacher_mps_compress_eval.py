#!/usr/bin/env python3
"""
tnshap_teacher_mps_compress_eval.py

Compare TN-SHAP on:

  - Teacher f(x) in ORIGINAL x-space
  - Compressed MPS+MLP surrogate (trained on the same path-augmented data)

Using:

  - Paths:
        h(t)   = f(t * x)
        g_i(t) = f(x_i fixed, t * x_{-i})

  - Chebyshev t-nodes in [0,1]
  - Vandermonde interpolation in t
  - TN-SHAP recurrence to recover per-feature Shapley values

Inputs:

  - <PREFIX>_tnshap_compress_targets.pt:
        x_base, t_nodes, X_all, Y_all, max_degree
  - <PREFIX>_mps_compress.pt:
        state_dict, input_dim, bond_dim, feature_dim, hidden_dim, max_degree

Outputs (printed):

  - R²(teacher vs Y_all), R²(MPS vs Y_all)
  - Per-point Shapley correlation on S, support accuracy
  - Per-point TN-SHAP runtimes for teacher & MPS
"""

import argparse
import time
import torch
import torch.nn as nn

from torchmps import MPS
from poly_teacher import load_teacher, DEVICE


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
    Solve V c = y (square Vandermonde) for polynomial coefficients c.
    """
    V64 = V.to(dtype=torch.float64)
    y64 = y.to(dtype=torch.float64)
    c = torch.linalg.solve(V64, y64)
    return c.to(dtype=torch.float32)


# -----------------------
# Diagonal evals (generic eval_fn) in ORIGINAL space
# -----------------------

def eval_h_on_nodes(eval_fn, x: torch.Tensor, t_nodes: torch.Tensor) -> torch.Tensor:
    """
    h(t) = f(t * x).
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
    g_i(t) = f(x_i fixed, t * x_{-i}).
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
# TN-SHAP via Vandermonde + recurrence
# -----------------------

def tnshap_vandermonde(
    eval_fn,
    x: torch.Tensor,
    max_degree: int,
    t_nodes: torch.Tensor,
) -> torch.Tensor:
    """
    TN-SHAP for an arbitrary function f on ORIGINAL x-space paths:

        h(t)   = f(t * x)
        g_i(t) = f(x_i fixed, t * x_{-i})

    Using:
      - Chebyshev t_nodes (>= max_degree+1)
      - Square Vandermonde in t
      - TN-SHAP recurrence to get β_{i,k}
      - Shapley_i = sum_{k=1..max_degree} β_{i,k} / k
    """
    x = x.to(DEVICE)
    D = x.shape[0]

    assert t_nodes.shape[0] >= max_degree + 1, \
        "Need at least max_degree+1 t-nodes."

    t_nodes_use = t_nodes[: max_degree + 1]
    V = build_vandermonde(t_nodes_use, degree_max=max_degree)

    # h(t)
    h_vals = eval_h_on_nodes(eval_fn, x, t_nodes_use)
    alpha = solve_poly_coeffs(V, h_vals)
    alpha64 = alpha.to(torch.float64)

    phi_full = torch.zeros(D, device=DEVICE, dtype=torch.float32)

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

        # consistency at k = 0:
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
# Local MLP feature map (must match training)
# -----------------------

class LocalMLPFeatureMap(nn.Module):
    """
    Per-site MLP feature map used in train_mps_teacher_paths_compress.py

        φ(s) : R -> R^{feature_dim}
    """
    def __init__(self, feature_dim: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        param_device = next(self.parameters()).device
        x = x.to(param_device)
        x_in = x.unsqueeze(-1)
        out = self.net(x_in)
        return out


def load_mps_compress(prefix: str):
    """
    Load the compressed MPS+MLP from <prefix>_mps_compress.pt
    """
    state = torch.load(f"{prefix}_mps_compress.pt", map_location=DEVICE)
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
    ).to(DEVICE)

    mlp_feature = LocalMLPFeatureMap(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
    ).to(DEVICE)

    mps.register_feature_map(mlp_feature)
    mps.load_state_dict(state["state_dict"])
    mps.eval()

    return mps


# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix", type=str, default="sqexp_D50",
        help="Prefix for *_tnshap_compress_targets.pt and *_mps_compress.pt"
    )
    parser.add_argument(
        "--n-targets", type=int, default=None,
        help="Number of base points x_base to evaluate TN-SHAP on "
             "(default: all x_base from the compress targets file)."
    )
    args = parser.parse_args()

    prefix = args.prefix
    print(f"=== TN-SHAP Teacher vs Compressed MPS for PREFIX={prefix} ===")

    # 1) Load TN-SHAP compression targets
    targets = torch.load(f"{prefix}_tnshap_compress_targets.pt", map_location=DEVICE)
    X_base = targets["x_base"].to(DEVICE)   # (N_base, D)
    t_nodes = targets["t_nodes"].to(DEVICE) # (N_T_NODES,)
    X_all   = targets["X_all"].to(DEVICE)   # (N_all, D)
    Y_all   = targets["Y_all"].to(DEVICE)   # (N_all,)
    max_degree_saved = int(targets["max_degree"])

    if args.n_targets is not None:
        X_base = X_base[: args.n_targets]

    N_TARGETS = X_base.shape[0]
    D = X_base.shape[1]
    N_all = X_all.shape[0]

    max_degree = max_degree_saved
    assert max_degree + 1 <= t_nodes.shape[0], \
        "t_nodes must contain at least max_degree+1 nodes."

    print(f"Using N_TARGETS={N_TARGETS}, D={D}, max_degree={max_degree}")
    print(f"Total path-aug points N_all={N_all}\n")

    # 2) Load teacher & compressed MPS
    teacher = load_teacher(prefix)
    if isinstance(teacher, nn.Module):
        teacher.to(DEVICE)

    mps = load_mps_compress(prefix)

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

    # 3) R² sanity on path-aug dataset
    print("=== R² on path-aug dataset (X_all, Y_all) ===")
    with torch.no_grad():
        y_teacher = eval_fn_teacher(X_all)
        y_mps = eval_fn_mps(X_all)

    r2_teacher = r2_score(Y_all, y_teacher)
    r2_mps = r2_score(Y_all, y_mps)

    print(f"Teacher vs ground truth: R² = {r2_teacher:.6f}")
    print(f"MPS vs ground truth:     R² = {r2_mps:.6f}\n")

    # 4) TN-SHAP Shapley eval per base point
    has_support = hasattr(teacher, "S")
    if has_support:
        S = list(teacher.S)
        S_set = set(S)
        k_active = len(S)
        print(f"Teacher has support S (size {k_active}): {S}\n")
    else:
        print("Teacher has no attribute 'S'; skipping support accuracy.\n")
        S, S_set, k_active = None, None, None

    acc_teacher_list = []
    acc_mps_list     = []
    corrs            = []
    t_teacher_list   = []
    t_mps_list       = []

    for idx in range(N_TARGETS):
        x0 = X_base[idx]

        # Teacher TN-SHAP
        t0 = time.time()
        phi_teacher = tnshap_vandermonde(
            eval_fn_teacher,
            x0,
            max_degree=max_degree,
            t_nodes=t_nodes,
        )
        t1 = time.time()
        dt_teacher = t1 - t0

        # MPS TN-SHAP
        t2 = time.time()
        phi_mps = tnshap_vandermonde(
            eval_fn_mps,
            x0,
            max_degree=max_degree,
            t_nodes=t_nodes,
        )
        t3 = time.time()
        dt_mps = t3 - t2

        t_teacher_list.append(dt_teacher)
        t_mps_list.append(dt_mps)

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
                f"t_teacher={dt_teacher*1000:.2f} ms, t_mps={dt_mps*1000:.2f} ms"
            )
        else:
            print(
                f"[Point {idx:02d}] "
                f"t_teacher={dt_teacher*1000:.2f} ms, t_mps={dt_mps*1000:.2f} ms"
            )

    # 5) Aggregate stats
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

        corr_tensor = torch.tensor(corrs, device=DEVICE)
        valid = ~torch.isnan(corr_tensor)
        if valid.any():
            mean_corr = corr_tensor[valid].mean().item()
            std_corr = corr_tensor[valid].std(unbiased=True).item() if valid.sum() > 1 else 0.0
        else:
            mean_corr, std_corr = float("nan"), float("nan")

        print("==== Correlation (phi_teacher[S], phi_mps[S]) ====")
        print(f"mean corr = {mean_corr:.4f}, std corr = {std_corr:.4f}")


if __name__ == "__main__":
    main()
