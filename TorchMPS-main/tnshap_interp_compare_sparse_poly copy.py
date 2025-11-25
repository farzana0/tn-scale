#!/usr/bin/env python3
"""
tnshap_interp_compare_sparse_poly.py

Compare a path-integral / interpolation-style attribution (with feature masks)
for:

  - The original sparse polynomial teacher (eval_fn_teacher)
  - The trained MPS surrogate (eval_fn_mps)

We use:
  - A straight-line path from baseline (0) to x
  - Interpolation nodes t in [0, 1]
  - For each feature i, we compare f(x_on(t)) vs f(x_off(t)):
      x_on(t): point on path with feature i active
      x_off(t): same path, but feature i forced to baseline

Then we approximate:
    phi_i  ≈  ∑_l w_l [ f(x_on(t_l)) - f(x_off(t_l)) ]

This gives an Integrated-Gradients-style path attribution using
"matrix selectors" (feature masking), applied identically to the
teacher and the MPS.

Run:
    python tnshap_interp_compare_sparse_poly.py
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
# Path-integral attribution with feature selectors
# -----------------------

def compute_path_interp_attributions(
    eval_fn,
    x: torch.Tensor,
    subset_indices,
    baseline_val: float = 0.0,
    n_nodes: int = 32,
) -> torch.Tensor:
    """
    Path-integral / interpolation-style attribution for a single point x.

    Path: straight line from baseline (0) to x:
        x(t) = baseline + t * (x - baseline) = t * x  (since baseline = 0)

    For each feature i:
        - x_on(t)  = x(t) with feature i as in x(t)
        - x_off(t) = x(t) but feature i forced to baseline (0)
        - Delta_i(t) = f(x_on(t)) - f(x_off(t))

    We approximate:
        phi_i ≈ (1 / n_nodes) * sum_l Delta_i(t_l)
    (simple Riemann sum; you can switch to quadrature weights if you want)

    eval_fn: callable mapping (B, D) -> (B,)
    x: (D,) tensor (single point)
    subset_indices: list of feature indices we care about (e.g., S)
    baseline_val: scalar for "off" features (here 0)
    """
    x = x.to(DEVICE)
    D = x.shape[0]
    J = list(subset_indices)
    nJ = len(J)

    # Interpolation nodes in [0,1]
    t_nodes = chebyshev_nodes_unit_interval(n_nodes, device=x.device, dtype=x.dtype)

    phi_full = torch.zeros(D, device=DEVICE, dtype=torch.float32)

    # Simple equal-weight Riemann sum; you can replace by proper Chebyshev weights if desired
    weight = 1.0 / n_nodes

    for i in J:
        contrib_i = 0.0

        for t in t_nodes:
            # Straight-line path: x(t) = t * x
            xt = t * x

            # x_on(t): full xt
            x_on = xt

            # x_off(t): same xt, but feature i masked to baseline
            x_off = xt.clone()
            x_off[i] = baseline_val

            X_batch = torch.stack([x_on, x_off], dim=0)  # (2, D)
            with torch.no_grad():
                y_batch = eval_fn(X_batch)               # (2,)
                if y_batch.ndim > 1:
                    y_batch = y_batch.squeeze(-1)
            f_on, f_off = y_batch[0], y_batch[1]

            Delta = f_on - f_off
            contrib_i += weight * Delta

        phi_full[i] = contrib_i

    return phi_full


# -----------------------
# Main comparison
# -----------------------

def main():
    # Load teacher, data, and MPS
    teacher = load_teacher(PREFIX)
    x_train, y_train, x_test, y_test = load_data(PREFIX)
    mps = load_mps(PREFIX)

    x_test = x_test.to(DEVICE)
    y_test = y_test.to(DEVICE)

    # Two eval functions: one for exact polynomial, one for MPS surrogate
    def eval_fn_teacher(x_batch: torch.Tensor) -> torch.Tensor:
        x_batch = x_batch.to(DEVICE)
        with torch.no_grad():
            return teacher(x_batch)

    def eval_fn_mps(x_batch: torch.Tensor) -> torch.Tensor:
        x_batch = x_batch.to(DEVICE)
        x_aug = augment_with_one(x_batch)
        with torch.no_grad():
            return mps(x_aug).squeeze(-1)

    # Active subset S from the teacher (true important features)
    S = teacher.S
    print(f"Active subset S (teacher support): {S}")
    k_active = len(S)
    S_set = set(S)

    # Sanity: R2 of MPS vs teacher on test set
    with torch.no_grad():
        y_teacher = eval_fn_teacher(x_test)
        y_mps = eval_fn_mps(x_test)
    r2_teacher = r2_score(y_test, y_teacher)
    r2_mps = r2_score(y_test, y_mps)
    print(f"Teacher vs ground truth: R2 = {r2_teacher:.4f}")
    print(f"MPS vs ground truth:     R2 = {r2_mps:.4f}")

    # Choose a subset of test points for attribution comparison
    N_TARGETS = 20
    N_NODES = 32  # interpolation nodes on the path
    X_targets = x_test[:N_TARGETS]

    phis_teacher_all = []
    phis_mps_all = []
    corrs = []

    for idx in range(N_TARGETS):
        x0 = X_targets[idx]

        phi_teacher = compute_path_interp_attributions(
            eval_fn_teacher,
            x0,
            subset_indices=S,
            baseline_val=0.0,
            n_nodes=N_NODES,
        )
        phi_mps = compute_path_interp_attributions(
            eval_fn_mps,
            x0,
            subset_indices=S,
            baseline_val=0.0,
            n_nodes=N_NODES,
        )

        phis_teacher_all.append(phi_teacher.unsqueeze(0))
        phis_mps_all.append(phi_mps.unsqueeze(0))

        # Correlation over active subset S only
        s_t = phi_teacher[S]
        s_m = phi_mps[S]
        if torch.std(s_t) < 1e-8 or torch.std(s_m) < 1e-8:
            corr = float("nan")
        else:
            c = torch.corrcoef(torch.stack([s_t, s_m]))[0, 1]
            corr = float(c.item())
        corrs.append(corr)

        print(f"[Point {idx:02d}] Path-interp corr on S = {corr:.4f}")

    phis_teacher_all = torch.cat(phis_teacher_all, dim=0)  # (N_TARGETS, D)
    phis_mps_all = torch.cat(phis_mps_all, dim=0)          # (N_TARGETS, D)

    # Average correlation
    finite_corrs = [c for c in corrs if not math.isnan(c)]
    mean_corr = sum(finite_corrs) / len(finite_corrs) if finite_corrs else float("nan")

    print("\n==== Summary over targets (path-integral / interpolation attribution) ====")
    print(f"Mean attribution correlation on S: {mean_corr:.4f}")

    # Support recovery: top-k_active by |phi| vs S, for teacher and MPS
    correct_support_teacher = 0
    correct_support_mps = 0

    for idx in range(N_TARGETS):
        phi_t = phis_teacher_all[idx]
        phi_m = phis_mps_all[idx]

        top_teacher = torch.topk(phi_t.abs(), k_active).indices.tolist()
        top_mps = torch.topk(phi_m.abs(), k_active).indices.tolist()

        if set(top_teacher) == S_set:
            correct_support_teacher += 1
        if set(top_mps) == S_set:
            correct_support_mps += 1

    print(f"Teacher path-attr support recovery (top-|S| == S): {correct_support_teacher}/{N_TARGETS}")
    print(f"MPS path-attr support recovery (top-|S| == S):     {correct_support_mps}/{N_TARGETS}")

    print("\nDone. This script now gives you:")
    print("- Interpolation/path-integral attributions using the TRUE function.")
    print("- The SAME interpolation/path-integral attributions using the MPS (with feature masks).")
    print("- A direct comparison of correlations and support recovery on S.")


if __name__ == "__main__":
    main()
