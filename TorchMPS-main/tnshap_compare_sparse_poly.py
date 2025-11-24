#!/usr/bin/env python3
"""
tnshap_compare_sparse_poly.py

End-to-end comparison of Shapley values for:
- Original sparse polynomial teacher (eval_fn_teacher)
- Trained MPS surrogate (eval_fn_mps)

We compute **exact Shapley values** on the active subset S (size k_active),
using the definitional formula over all coalitions (2^k subsets).
This is a clean way to debug whether MPS recovers the same interactions
as the original function for the important features.

Run:
    python tnshap_compare_sparse_poly.py
"""

import math
import torch
from torchmps import MPS

from poly_teacher import load_teacher, load_data, DEVICE

PREFIX = "poly"


# -----------------------
# Load MPS (same as eval_compare.py)
# -----------------------
def augment_with_one(x: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
    return torch.cat([ones, x], dim=1)

def r2_score(y_true, y_pred) -> float:
    y_true = y_true.detach()
    y_pred = y_pred.detach()
    var = torch.var(y_true)
    if var < 1e-12:
        return 1.0 if torch.allclose(y_true, y_pred) else 0.0
    return float(1.0 - torch.mean((y_true - y_pred) ** 2) / (var + 1e-12))

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
# Exact Shapley on subset of features
# -----------------------

def compute_exact_shap_subset(
    eval_fn,
    x: torch.Tensor,
    subset_indices,
    baseline_val: float = 0.0,
) -> torch.Tensor:
    """
    Compute exact Shapley values on a subset J of features for a single point x.

    eval_fn: callable mapping (B, D) tensor -> (B,) tensor
    x: (D,) tensor (single point)
    subset_indices: list/tuple of feature indices J = [j0, j1, ..., j_{n-1}]
    baseline_val: scalar baseline value for features 'off' in a coalition

    Returns:
        phi_full: (D,) Shapley vector where:
            phi_full[j] = Shapley value for feature j if j in J,
                          0 for j not in J.
    """
    x = x.to(DEVICE)
    D = x.shape[0]
    J = list(subset_indices)
    n = len(J)

    num_coalitions = 2 ** n

    # Precompute all masked inputs for all coalitions T ⊆ J
    X_all = []
    for mask_int in range(num_coalitions):
        z = x.clone()
        for k in range(n):
            j = J[k]
            bit = (mask_int >> k) & 1
            if bit == 0:
                z[j] = baseline_val
            # else keep x[j]
        X_all.append(z.unsqueeze(0))  # (1, D)

    X_all = torch.cat(X_all, dim=0)  # (2^n, D)
    with torch.no_grad():
        y_all = eval_fn(X_all)       # (2^n,)
        if y_all.ndim > 1:
            y_all = y_all.squeeze(-1)

    # Compute Shapley for each feature in J
    phi_full = torch.zeros(D, device=DEVICE, dtype=torch.float32)

    n_fact = math.factorial(n)
    # Precompute coalition sizes
    coalition_sizes = []
    for mask_int in range(num_coalitions):
        size = 0
        tmp = mask_int
        while tmp:
            size += tmp & 1
            tmp >>= 1
        coalition_sizes.append(size)

    for i_pos, feat_idx in enumerate(J):
        phi_i = 0.0
        for mask_int in range(num_coalitions):
            # Only coalitions T that do NOT contain i
            if (mask_int >> i_pos) & 1:
                continue

            size_T = coalition_sizes[mask_int]
            weight = (
                math.factorial(size_T)
                * math.factorial(n - size_T - 1)
                / n_fact
            )

            mask_with_i = mask_int | (1 << i_pos)

            f_T = y_all[mask_int]
            f_T_i = y_all[mask_with_i]

            phi_i += weight * (f_T_i - f_T)

        phi_full[feat_idx] = phi_i

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

    # Construct eval functions
    def eval_fn_teacher(x_batch: torch.Tensor) -> torch.Tensor:
        x_batch = x_batch.to(DEVICE)
        with torch.no_grad():
            return teacher(x_batch)

    def eval_fn_mps(x_batch: torch.Tensor) -> torch.Tensor:
        x_batch = x_batch.to(DEVICE)
        x_aug = augment_with_one(x_batch)
        with torch.no_grad():
            return mps(x_aug).squeeze(-1)

    # Active subset S from the teacher
    S = teacher.S
    print(f"Active subset S (teacher support): {S}")
    k_active = len(S)

    # Some sanity: R2 of MPS vs teacher on test set
    with torch.no_grad():
        y_teacher = eval_fn_teacher(x_test)
        y_mps = eval_fn_mps(x_test)
    r2_teacher = r2_score(y_test, y_teacher)
    r2_mps = r2_score(y_test, y_mps)
    print(f"Teacher vs ground truth: R2 = {r2_teacher:.4f}")
    print(f"MPS vs ground truth:     R2 = {r2_mps:.4f}")

    # Choose a subset of test points for Shapley comparison
    N_TARGETS = 20
    X_targets = x_test[:N_TARGETS]

    phis_teacher_all = []
    phis_mps_all = []
    corrs = []

    for idx in range(N_TARGETS):
        x0 = X_targets[idx]

        phi_teacher = compute_exact_shap_subset(eval_fn_teacher, x0, S, baseline_val=0.0)
        phi_mps = compute_exact_shap_subset(eval_fn_mps, x0, S, baseline_val=0.0)

        phis_teacher_all.append(phi_teacher.unsqueeze(0))
        phis_mps_all.append(phi_mps.unsqueeze(0))

        # Compute correlation only on active subset S
        s_t = phi_teacher[S]
        s_m = phi_mps[S]
        if torch.std(s_t) < 1e-8 or torch.std(s_m) < 1e-8:
            corr = float("nan")
        else:
            c = torch.corrcoef(torch.stack([s_t, s_m]))[0, 1]
            corr = float(c.item())
        corrs.append(corr)

        print(f"[Point {idx:02d}] Shap corr on S = {corr:.4f}")

    phis_teacher_all = torch.cat(phis_teacher_all, dim=0)  # (N_TARGETS, D)
    phis_mps_all = torch.cat(phis_mps_all, dim=0)          # (N_TARGETS, D)

    # Average correlation over targets
    finite_corrs = [c for c in corrs if not math.isnan(c)]
    if finite_corrs:
        mean_corr = sum(finite_corrs) / len(finite_corrs)
    else:
        mean_corr = float("nan")

    print("\n==== Summary over targets ====")
    print(f"Mean Shapley correlation on S: {mean_corr:.4f}")

    # Support recovery: top-k_active by |phi| vs S
    correct_support_teacher = 0
    correct_support_mps = 0

    S_set = set(S)

    for idx in range(N_TARGETS):
        phi_t = phis_teacher_all[idx]
        phi_m = phis_mps_all[idx]

        top_teacher = torch.topk(phi_t.abs(), k_active).indices.tolist()
        top_mps = torch.topk(phi_m.abs(), k_active).indices.tolist()

        if set(top_teacher) == S_set:
            correct_support_teacher += 1
        if set(top_mps) == S_set:
            correct_support_mps += 1

    print(f"Teacher Shap support recovery (top-|S| == S): {correct_support_teacher}/{N_TARGETS}")
    print(f"MPS Shap support recovery (top-|S| == S):     {correct_support_mps}/{N_TARGETS}")

    print("\nDone. You now have:")
    print("- eval_fn_teacher & eval_fn_mps")
    print("- exact Shapley computation on active features S")
    print("- direct comparison between teacher and MPS Shapley vectors.")
    

if __name__ == "__main__":
    main()
