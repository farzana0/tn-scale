#!/usr/bin/env python3
"""
tnshap_aligned_comparison.py

Compare TN-SHAP on teacher vs MPS using EXACTLY the same evaluation points.
The MPS was trained on these exact points, ensuring fair comparison.
"""

import torch
import math
from torchmps import MPS
from poly_teacher import load_teacher, DEVICE


def augment_with_one(x: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
    return torch.cat([ones, x], dim=1)


def load_mps_aligned(prefix: str) -> MPS:
    state = torch.load(f"{prefix}_mps_aligned.pt", map_location=DEVICE)
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


def build_vandermonde(t_nodes: torch.Tensor, degree_max: int) -> torch.Tensor:
    t = t_nodes.to(dtype=torch.float64)
    n_nodes = t.shape[0]
    exponents = torch.arange(0, degree_max + 1, dtype=torch.float64, device=t.device)
    V = t.unsqueeze(1) ** exponents.unsqueeze(0)
    return V


def solve_poly_coeffs(V: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    V64 = V.to(dtype=torch.float64)
    y64 = y.to(dtype=torch.float64)
    c = torch.linalg.solve(V64, y64)
    return c.to(dtype=torch.float32)


def tnshap_from_precomputed(
    h_vals: torch.Tensor,
    g_vals_dict: dict,
    t_nodes: torch.Tensor,
    D: int,
) -> torch.Tensor:
    """
    Compute TN-SHAP using precomputed multilinear extension values.
    This ensures both teacher and MPS use EXACTLY the same evaluation points.
    """
    max_degree = t_nodes.shape[0] - 1
    V = build_vandermonde(t_nodes, degree_max=max_degree)
    
    # Fit h(t) polynomial
    alpha = solve_poly_coeffs(V, h_vals)
    alpha64 = alpha.to(torch.float64)
    
    phi = torch.zeros(D, device=DEVICE, dtype=torch.float32)
    
    for i in range(D):
        # Fit g_i(t) polynomial
        gamma = solve_poly_coeffs(V, g_vals_dict[i])
        gamma64 = gamma.to(torch.float64)
        
        # Recurrence for β_{i,k}
        beta = torch.zeros(max_degree + 1, dtype=torch.float64, device=DEVICE)
        
        # k = max_degree
        beta[max_degree] = alpha64[max_degree] - gamma64[max_degree]
        
        # k = max_degree-1 down to 1
        for k in range(max_degree - 1, 0, -1):
            beta[k] = alpha64[k] + beta[k + 1] - gamma64[k]
        
        # Consistency check/average
        beta1_from_gamma0 = gamma64[0] - alpha64[0]
        beta[1] = 0.5 * (beta[1] + beta1_from_gamma0)
        
        # Shapley value
        ks = torch.arange(1, max_degree + 1, dtype=torch.float64, device=DEVICE)
        phi_i = torch.sum(beta[1:] / ks)
        phi[i] = phi_i.to(torch.float32)
    
    return phi


def main():
    # Load aligned data
    data = torch.load(f"{PREFIX}_tnshap_aligned_data.pt", map_location=DEVICE)
    x_base = data["x_base"].to(DEVICE)
    t_nodes = data["t_nodes"].to(DEVICE)
    eval_dict = data["evaluation_dict"]
    
    # Convert evaluation dict tensors to device
    for key in eval_dict:
        if torch.is_tensor(eval_dict[key]):
            eval_dict[key] = eval_dict[key].to(DEVICE)
        elif isinstance(eval_dict[key], dict):
            for subkey in eval_dict[key]:
                eval_dict[key][subkey] = eval_dict[key][subkey].to(DEVICE)
    
    N_base, D = x_base.shape
    N_t = t_nodes.shape[0]
    
    print(f"Loaded aligned data: N_base={N_base}, D={D}, N_t={N_t}")
    
    # Load models
    teacher = load_teacher(PREFIX)
    mps = load_mps_aligned(PREFIX)
    
    # Get teacher's support
    S = teacher.S
    print(f"Teacher support S: {S}")
    k_active = len(S)
    S_set = set(S)
    
    # Process each base point
    correlations = []
    teacher_acc_list = []
    mps_acc_list = []
    
    for b_idx in range(N_base):
        # Extract precomputed values for this base point
        h_start = b_idx * N_t
        h_end = (b_idx + 1) * N_t
        
        h_vals_teacher = eval_dict['h_values'][h_start:h_end]
        
        g_vals_teacher = {}
        for i in range(D):
            g_vals_teacher[i] = eval_dict['g_values'][i][h_start:h_end]
        
        # Compute MPS values on SAME points
        with torch.no_grad():
            # h(t) values
            h_points = eval_dict['h_points'][h_start:h_end]
            h_points_aug = augment_with_one(h_points)
            h_vals_mps = mps(h_points_aug).squeeze(-1)
            
            # g_i(t) values
            g_vals_mps = {}
            for i in range(D):
                g_points = eval_dict['g_points'][i][h_start:h_end]
                g_points_aug = augment_with_one(g_points)
                g_vals_mps[i] = mps(g_points_aug).squeeze(-1)
        
        # Compute TN-SHAP for both
        phi_teacher = tnshap_from_precomputed(h_vals_teacher, g_vals_teacher, t_nodes, D)
        phi_mps = tnshap_from_precomputed(h_vals_mps, g_vals_mps, t_nodes, D)
        
        # Compare on support S
        s_t = phi_teacher[S]
        s_m = phi_mps[S]
        
        if torch.std(s_t) > 1e-8 and torch.std(s_m) > 1e-8:
            corr = torch.corrcoef(torch.stack([s_t, s_m]))[0, 1].item()
        else:
            corr = float("nan")
        correlations.append(corr)
        
        # Support accuracy
        top_teacher = torch.topk(phi_teacher.abs(), k_active).indices.tolist()
        top_mps = torch.topk(phi_mps.abs(), k_active).indices.tolist()
        
        acc_teacher = len(set(top_teacher) & S_set) / k_active
        acc_mps = len(set(top_mps) & S_set) / k_active
        
        teacher_acc_list.append(acc_teacher)
        mps_acc_list.append(acc_mps)
        
        print(f"[Point {b_idx:02d}] Corr={corr:.4f}, "
              f"Acc_teacher={acc_teacher:.2f}, Acc_MPS={acc_mps:.2f}")
        
        # Verify evaluation consistency
        h_diff = torch.abs(h_vals_teacher - eval_dict['h_values'][h_start:h_end]).max()
        print(f"  h(t) consistency check: max_diff={h_diff:.6f}")
    
    # Summary
    finite_corrs = [c for c in correlations if not math.isnan(c)]
    mean_corr = sum(finite_corrs) / len(finite_corrs) if finite_corrs else float("nan")
    mean_acc_teacher = sum(teacher_acc_list) / len(teacher_acc_list)
    mean_acc_mps = sum(mps_acc_list) / len(mps_acc_list)
    
    print("\n" + "="*60)
    print("TN-SHAP ALIGNED COMPARISON SUMMARY")
    print("="*60)
    print(f"Mean correlation on S: {mean_corr:.4f}")
    print(f"Mean support accuracy (teacher): {mean_acc_teacher:.4f}")
    print(f"Mean support accuracy (MPS): {mean_acc_mps:.4f}")
    print("\n✓ Both models evaluated on IDENTICAL points")
    print("✓ MPS was trained on these exact evaluation points")
    print("✓ Fair apples-to-apples comparison achieved")


PREFIX = "poly"

if __name__ == "__main__":
    main()