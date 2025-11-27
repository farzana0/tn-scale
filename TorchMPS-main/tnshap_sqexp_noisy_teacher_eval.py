#!/usr/bin/env python3
"""
tnshap_sqexp_noisy_teacher_eval.py

TN-SHAP-style Shapley computation for:

  - Clean exponential teacher  f(x)
  - Noisy teacher              f̃(x) = f(x) + eps

Both are evaluated only via:
  - h(t)   = f(t * x)
  - g_i(t) = f(x_i fixed, others scaled by t)

We:
  - Pick N_TARGETS base points (from test set by default, otherwise train).
  - Build Chebyshev nodes t in [0, 1] (K = max_degree+1).
  - For each base point x:
      * Compute Shapley via TN-SHAP + Vandermonde on f (clean).
      * Compute Shapley via TN-SHAP + Vandermonde on f̃ (noisy).
  - Optionally use teacher.S as ground-truth support and compute:
      * Support accuracy (clean vs S)
      * Support accuracy (noisy vs S)
  - Print correlation between clean and noisy Shapley on each point.
  - Save everything to <prefix>_sqexp_tnshap_teacher_noisy.pt.
"""

import argparse
import time

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


def eval_h_on_nodes(eval_fn,
                    x: torch.Tensor,
                    t_nodes: torch.Tensor) -> torch.Tensor:
    """
    Generic h(t) = f(t * x) for arbitrary eval_fn.

    x: (D,)
    t_nodes: (L,)
    returns: (L,)
    """
    x = x.to(DEVICE)
    t_nodes = t_nodes.to(x.device)
    X_batch = t_nodes.unsqueeze(1) * x.unsqueeze(0)  # (L, D)
    y = eval_fn(X_batch)
    if y.ndim > 1:
        y = y.squeeze(-1)
    return y


def eval_g_i_on_nodes(eval_fn,
                      x: torch.Tensor,
                      i: int,
                      t_nodes: torch.Tensor) -> torch.Tensor:
    """
    Generic g_i(t) = f(x_i fixed, others scaled by t) for arbitrary eval_fn.

    x: (D,)
    t_nodes: (L,)
    returns: (L,)
    """
    x = x.to(DEVICE)
    t_nodes = t_nodes.to(x.device)
    D = x.shape[0]

    X_scaled = t_nodes.unsqueeze(1) * x.unsqueeze(0)  # (L, D)
    X_scaled[:, i] = x[i]  # clamp feature i

    y = eval_fn(X_scaled)
    if y.ndim > 1:
        y = y.squeeze(-1)
    return y


# -----------------------
# TN-SHAP via Vandermonde + recurrence (generic eval_fn)
# -----------------------

def tnshap_vandermonde_generic(
    eval_fn,
    x: torch.Tensor,
    max_degree: int,
    t_nodes: torch.Tensor,
) -> torch.Tensor:
    """
    TN-SHAP for an arbitrary function via path construction in ORIGINAL x-space:

      - h(t)   = f(t * x)
      - g_i(t) = f(x_i fixed, t * x_{-i})

    Uses Chebyshev nodes t_nodes (>= max_degree+1) and:
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
    h_vals = eval_h_on_nodes(eval_fn, x, t_nodes_use)
    alpha = solve_poly_coeffs(V, h_vals)            # (max_degree+1,)
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
    parser.add_argument(
        "--noise-std", type=float, default=1e-3,
        help="Noise std as a multiple of std(y_train). "
             "f̃(x) = f(x) + noise_std * std(y_train) * N(0,1)."
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for noise."
    )
    args = parser.parse_args()

    prefix = args.prefix
    print(f"=== TN-SHAP sqexp Noisy-Teacher Eval for PREFIX={prefix} ===")
    print(f"max_degree={args.max_degree}, n_targets={args.n_targets}, use_train={args.use_train}")
    print(f"noise_std={args.noise-std if hasattr(args, 'noise-std') else args.noise_std}, seed={args.seed}")

    # Fix seed for reproducibility of noise
    torch.manual_seed(args.seed)

    # 1) Load teacher and data
    teacher = load_teacher(prefix)
    if isinstance(teacher, torch.nn.Module):
        teacher.to(DEVICE)

    x_train, y_train, x_test, y_test = load_data(prefix)
    x_train = x_train.to(DEVICE)
    x_test = x_test.to(DEVICE)

    # Flatten y_train to compute std for noise scaling
    if y_train is not None:
        y_train_flat = y_train.view(-1)
        y_scale = y_train_flat.std().item()
        if y_scale < 1e-12:
            y_scale = 1.0
    else:
        y_scale = 1.0
    print(f"Using y_scale={y_scale:.4e} for noise.")

    # Choose base points: default from test if available, else from train
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

    # 2) Build Chebyshev nodes in t
    max_degree = int(args.max_degree)
    K = max_degree + 1
    t_nodes = chebyshev_nodes_unit_interval(
        K, device=DEVICE, dtype=X_targets.dtype
    )
    print(f"Using K={K} Chebyshev nodes in t for interpolation (degree <= {max_degree}).")

    # Clean teacher eval_fn
    def eval_fn_teacher(x_batch: torch.Tensor) -> torch.Tensor:
        x_batch = x_batch.to(DEVICE)
        with torch.no_grad():
            y = teacher(x_batch)
            if y.ndim > 1:
                y = y.squeeze(-1)
        return y

    # Noisy teacher eval_fn
    noise_std = args.noise_std * y_scale

    def eval_fn_noisy(x_batch: torch.Tensor) -> torch.Tensor:
        x_batch = x_batch.to(DEVICE)
        with torch.no_grad():
            y_clean = teacher(x_batch)
            if y_clean.ndim > 1:
                y_clean = y_clean.squeeze(-1)
        noise = noise_std * torch.randn_like(y_clean)
        return y_clean + noise

    # 3) Compute TN-SHAP Shapley vectors
    shap_clean = torch.zeros(N_targets, D, device=DEVICE, dtype=torch.float32)
    shap_noisy = torch.zeros_like(shap_clean)

    # Check if teacher has ground-truth support S
    has_support = hasattr(teacher, "S")
    if has_support:
        S = list(teacher.S)
        S_set = set(S)
        k_active = len(S)
        print(f"Teacher has support S of size {k_active}: {S}")
        support_acc_clean = []
        support_acc_noisy = []
    else:
        print("Teacher has no attribute 'S'; skipping support accuracy.")
        support_acc_clean = None
        support_acc_noisy = None

    t_clean_list = []
    t_noisy_list = []
    corr_list = []

    for idx in range(N_targets):
        x0 = X_targets[idx]

        # Clean
        t0 = time.time()
        phi_clean = tnshap_vandermonde_generic(
            eval_fn_teacher,
            x0,
            max_degree=max_degree,
            t_nodes=t_nodes,
        )
        t1 = time.time()

        # Noisy
        t2 = time.time()
        phi_noisy = tnshap_vandermonde_generic(
            eval_fn_noisy,
            x0,
            max_degree=max_degree,
            t_nodes=t_nodes,
        )
        t3 = time.time()

        shap_clean[idx] = phi_clean
        shap_noisy[idx] = phi_noisy

        t_clean = t1 - t0
        t_noisy = t3 - t2
        t_clean_list.append(t_clean)
        t_noisy_list.append(t_noisy)

        # Correlation between clean and noisy Shapley
        if torch.std(phi_clean) < 1e-8 or torch.std(phi_noisy) < 1e-8:
            corr = float("nan")
        else:
            c = torch.corrcoef(torch.stack([phi_clean, phi_noisy]))[0, 1]
            corr = float(c.item())
        corr_list.append(corr)

        msg = f"[Point {idx:02d}] corr(clean,noisy)={corr:.4f}, " \
              f"t_clean={1000*t_clean:.2f} ms, t_noisy={1000*t_noisy:.2f} ms"

        if has_support:
            top_clean = torch.topk(phi_clean.abs(), k_active).indices.tolist()
            top_noisy = torch.topk(phi_noisy.abs(), k_active).indices.tolist()

            correct_clean = len(set(top_clean) & S_set)
            correct_noisy = len(set(top_noisy) & S_set)

            acc_clean = correct_clean / k_active
            acc_noisy = correct_noisy / k_active

            support_acc_clean.append(acc_clean)
            support_acc_noisy.append(acc_noisy)

            msg += f", acc_clean={acc_clean:.2f}, acc_noisy={acc_noisy:.2f}"

        print(msg)

    # 4) Aggregate stats
    t_clean_tensor = torch.tensor(t_clean_list, device=DEVICE)
    t_noisy_tensor = torch.tensor(t_noisy_list, device=DEVICE)
    corr_tensor = torch.tensor(corr_list, device=DEVICE)

    mean_t_clean = t_clean_tensor.mean().item()
    std_t_clean = t_clean_tensor.std(unbiased=True).item() if len(t_clean_tensor) > 1 else 0.0

    mean_t_noisy = t_noisy_tensor.mean().item()
    std_t_noisy = t_noisy_tensor.std(unbiased=True).item() if len(t_noisy_tensor) > 1 else 0.0

    # Backward-compatible nanmean / nanstd
    valid = ~torch.isnan(corr_tensor)

    if valid.any():
        mean_corr = corr_tensor[valid].mean().item()
        std_corr = corr_tensor[valid].std(unbiased=True).item() if valid.sum() > 1 else 0.0
    else:
        mean_corr = float("nan")
        std_corr = float("nan")

    print("==== Per-point TN-SHAP evaluation time ====")
    print(f"Clean teacher: mean = {mean_t_clean*1000:.2f} ms, std = {std_t_clean*1000:.2f} ms")
    print(f"Noisy teacher: mean = {mean_t_noisy*1000:.2f} ms, std = {std_t_noisy*1000:.2f} ms")

    print("==== Correlation between clean and noisy Shapley ====")
    print(f"mean corr = {mean_corr:.4f}, std corr = {std_corr:.4f}")

    if has_support and support_acc_clean is not None:
        sac_clean = torch.tensor(support_acc_clean, device=DEVICE)
        sac_noisy = torch.tensor(support_acc_noisy, device=DEVICE)

        mean_acc_clean = sac_clean.mean().item()
        std_acc_clean = sac_clean.std(unbiased=True).item() if len(sac_clean) > 1 else 0.0

        mean_acc_noisy = sac_noisy.mean().item()
        std_acc_noisy = sac_noisy.std(unbiased=True).item() if len(sac_noisy) > 1 else 0.0

        print("==== Shapley support accuracy (vs teacher.S) ====")
        print(f"Clean  mean acc = {mean_acc_clean:.4f}, std = {std_acc_clean:.4f}")
        print(f"Noisy  mean acc = {mean_acc_noisy:.4f}, std = {std_acc_noisy:.4f}")

    # 5) Save results
    out_path = f"{prefix}_sqexp_tnshap_teacher_noisy.pt"
    torch.save(
        {
            "x_targets": X_targets.detach().cpu(),         # (N_targets, D)
            "shap_clean": shap_clean.detach().cpu(),       # (N_targets, D)
            "shap_noisy": shap_noisy.detach().cpu(),       # (N_targets, D)
            "t_nodes": t_nodes.detach().cpu(),             # (K,)
            "max_degree": max_degree,
            "prefix": prefix,
            "has_support": has_support,
            "support": teacher.S if has_support else None,
            "noise_std_effective": noise_std,
            "corr_clean_noisy": corr_tensor.detach().cpu(),
        },
        out_path,
    )
    print(f"Saved clean & noisy TN-SHAP results to {out_path}")


if __name__ == "__main__":
    main()
