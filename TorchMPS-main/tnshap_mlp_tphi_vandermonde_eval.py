#!/usr/bin/env python3
"""
tnshap_mlp_tphi_vandermonde_eval.py

TN-SHAP Shapley comparison for:

  - Teacher f(x) in ORIGINAL space (paths: t*x and g_i masks)
  - Surrogate MPS+MLP(tφ(x)) in FEATURE space:

        h-path:   z_h(t)   = t * φ(x)
        g_i-path: z_g^i(t)[j] = φ_j(x_j) if j == i else t * φ_j(x_j)

We:

  - Check R² of surrogate vs teacher on the same path-aug dataset.
  - Compute TN-SHAP via Vandermonde + recurrence for both.
  - Report Shapley support accuracy and per-point runtimes.
"""

import argparse
import time
import torch
from torch.utils.data import TensorDataset, DataLoader

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


# -----------------------
# Scalar MLP feature map (same as training)
# -----------------------

class ScalarMLPFeatureMap(torch.nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D) -> (B, D, 3)
        """
        B, D = x.shape
        # Use reshape to avoid contiguous issues
        x_flat = x.reshape(B * D, 1)
        h_flat = self.net(x_flat)     # (B*D, 3)
        h = h_flat.reshape(B, D, 3)
        return h


def build_phi_t_features_base(
    x_base: torch.Tensor,
    t: torch.Tensor,
    is_h: torch.Tensor,
    i_idx: torch.Tensor,
    phi_module: ScalarMLPFeatureMap,
) -> torch.Tensor:
    """
    Same t * φ(x_base) semantics as in training:

      h-path:   all coords scaled by t
      g_i-path: coord i unscaled, others scaled by t
    """
    B, D = x_base.shape
    device = x_base.device

    phi_signal = phi_module(x_base)  # (B, D, 3)
    ones = torch.ones(B, D, 1, device=device, dtype=x_base.dtype)
    phi_full = torch.cat([phi_signal, ones], dim=-1)  # (B, D, 4)

    base_scale = t.view(B, 1, 1).expand(B, D, 1).clone()

    gi_mask = (~is_h).nonzero(as_tuple=False).view(-1)
    if gi_mask.numel() > 0:
        b_idx = gi_mask
        i_vals = i_idx[b_idx]
        base_scale[b_idx, i_vals, 0] = 1.0

    phi_full[:, :, :3] = phi_full[:, :, :3] * base_scale
    Z = phi_full.reshape(B, D * 4)
    return Z


# -----------------------
# Vandermonde / interpolation
# -----------------------

def build_vandermonde(t_nodes: torch.Tensor, degree_max: int) -> torch.Tensor:
    t = t_nodes.to(dtype=torch.float64)
    exponents = torch.arange(0, degree_max + 1, dtype=torch.float64, device=t.device)
    V = t.unsqueeze(1) ** exponents.unsqueeze(0)
    return V


def solve_poly_coeffs(V: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    V64 = V.to(dtype=torch.float64)
    y64 = y.to(dtype=torch.float64)
    c = torch.linalg.solve(V64, y64)
    return c.to(dtype=torch.float32)


# -----------------------
# Teacher evals in ORIGINAL space
# -----------------------

def eval_h_on_nodes_teacher(eval_fn, x: torch.Tensor, t_nodes: torch.Tensor) -> torch.Tensor:
    x = x.to(DEVICE)
    t_nodes = t_nodes.to(x.device)
    X_batch = t_nodes.unsqueeze(1) * x.unsqueeze(0)  # (L, D)
    with torch.no_grad():
        y = eval_fn(X_batch)
        if y.ndim > 1:
            y = y.squeeze(-1)
    return y


def eval_g_i_on_nodes_teacher(eval_fn, x: torch.Tensor, i: int, t_nodes: torch.Tensor) -> torch.Tensor:
    x = x.to(DEVICE)
    t_nodes = t_nodes.to(x.device)
    D = x.shape[0]

    X_scaled = t_nodes.unsqueeze(1) * x.unsqueeze(0)
    X_scaled[:, i] = x[i]
    with torch.no_grad():
        y = eval_fn(X_scaled)
        if y.ndim > 1:
            y = y.squeeze(-1)
    return y


# -----------------------
# Surrogate evals: MPS+MLP(tφ(x)) – ORIGINAL x, FEATURE space path
# -----------------------

def eval_h_on_nodes_mps_base(
    mps: MPS,
    phi_module: ScalarMLPFeatureMap,
    x: torch.Tensor,
    t_nodes: torch.Tensor,
) -> torch.Tensor:
    """
    Surrogate h(t) = MPS(t * φ(x)), using base x (no t inside MLP).
    """
    x = x.to(DEVICE)
    t_nodes = t_nodes.to(DEVICE)
    L = t_nodes.shape[0]
    D = x.shape[0]

    X_base_batch = x.unsqueeze(0).expand(L, D)     # (L, D)
    t_batch = t_nodes.clone()                      # (L,)
    is_h_batch = torch.ones(L, device=DEVICE, dtype=torch.bool)
    i_batch = torch.full((L,), -1, device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        Z_batch = build_phi_t_features_base(X_base_batch, t_batch, is_h_batch, i_batch, phi_module)
        y = mps(Z_batch).squeeze(-1)
    return y


def eval_g_i_on_nodes_mps_base(
    mps: MPS,
    phi_module: ScalarMLPFeatureMap,
    x: torch.Tensor,
    i: int,
    t_nodes: torch.Tensor,
) -> torch.Tensor:
    """
    Surrogate g_i(t) = MPS(z_g^i(t)) where:

      z_g^i(t)[j] = φ_j(x_j) if j == i else t * φ_j(x_j)
    """
    x = x.to(DEVICE)
    t_nodes = t_nodes.to(DEVICE)
    L = t_nodes.shape[0]
    D = x.shape[0]

    X_base_batch = x.unsqueeze(0).expand(L, D)   # base x replicated
    t_batch = t_nodes.clone()
    is_h_batch = torch.zeros(L, device=DEVICE, dtype=torch.bool)
    i_batch = torch.full((L,), i, device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        Z_batch = build_phi_t_features_base(X_base_batch, t_batch, is_h_batch, i_batch, phi_module)
        y = mps(Z_batch).squeeze(-1)
    return y


# -----------------------
# Load surrogate
# -----------------------

def load_mps_and_phi(prefix: str):
    state = torch.load(f"{prefix}_mps_mlp_tphi.pt", map_location=DEVICE)
    D_aug = state["D_aug"]
    bond_dim = state["bond_dim"]
    mlp_hidden = state.get("mlp_hidden", 32)

    phi_module = ScalarMLPFeatureMap(hidden_dim=mlp_hidden).to(DEVICE)
    mps = MPS(
        input_dim=D_aug,
        output_dim=1,
        bond_dim=bond_dim,
        adaptive_mode=False,
        periodic_bc=False,
        parallel_eval=True,
    ).to(DEVICE)

    phi_module.load_state_dict(state["phi_state_dict"])
    mps.load_state_dict(state["mps_state_dict"])

    phi_module.eval()
    mps.eval()
    return mps, phi_module


# -----------------------
# TN-SHAP via Vandermonde & recurrence
# -----------------------

def tnshap_vandermonde_teacher(
    eval_fn,
    x: torch.Tensor,
    max_degree: int,
    t_nodes: torch.Tensor,
) -> torch.Tensor:
    """
    TN-SHAP for TEACHER (original x-space paths).
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
        beta[max_degree] = alpha64[max_degree] - gamma64[max_degree]

        for k in range(max_degree - 1, 0, -1):
            beta[k] = alpha64[k] + beta[k + 1] - gamma64[k]

        beta1_from_gamma0 = gamma64[0] - alpha64[0]
        beta[1] = 0.5 * (beta[1] + beta1_from_gamma0)

        ks = torch.arange(1, max_degree + 1, dtype=torch.float64, device=x.device)
        phi_i = torch.sum(beta[1:] / ks)
        phi_full[i] = phi_i.to(torch.float32)

    return phi_full


def tnshap_vandermonde_mps(
    mps: MPS,
    phi_module: ScalarMLPFeatureMap,
    x: torch.Tensor,
    max_degree: int,
    t_nodes: torch.Tensor,
) -> torch.Tensor:
    """
    TN-SHAP for MPS+MLP(tφ(x)) surrogate (feature-space paths).
    """
    x = x.to(DEVICE)
    D = x.shape[0]
    assert t_nodes.shape[0] >= max_degree + 1
    t_nodes_sq = t_nodes[: max_degree + 1]
    V = build_vandermonde(t_nodes_sq, degree_max=max_degree)

    # h(t) in feature space
    h_vals = eval_h_on_nodes_mps_base(mps, phi_module, x, t_nodes_sq)
    alpha = solve_poly_coeffs(V, h_vals)
    alpha64 = alpha.to(torch.float64)

    phi_full = torch.zeros(D, device=DEVICE, dtype=torch.float32)
    for i in range(D):
        g_vals = eval_g_i_on_nodes_mps_base(mps, phi_module, x, i, t_nodes_sq)
        gamma = solve_poly_coeffs(V, g_vals)
        gamma64 = gamma.to(torch.float64)

        beta = torch.zeros(max_degree + 1, dtype=torch.float64, device=x.device)
        beta[max_degree] = alpha64[max_degree] - gamma64[max_degree]

        for k in range(max_degree - 1, 0, -1):
            beta[k] = alpha64[k] + beta[k + 1] - gamma64[k]

        beta1_from_gamma0 = gamma64[0] - alpha64[0]
        beta[1] = 0.5 * (beta[1] + beta1_from_gamma0)

        ks = torch.arange(1, max_degree + 1, dtype=torch.float64, device=x.device)
        phi_i = torch.sum(beta[1:] / ks)
        phi_full[i] = phi_i.to(torch.float32)

    return phi_full


# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="sqexp_D50",
                        help="Prefix for *_teacher.pt, *_tnshap_targets.pt, *_mps_mlp_tphi.pt")
    parser.add_argument("--max-degree", type=int, default=None,
                        help="Override polynomial degree in t; default=stored value.")
    parser.add_argument("--n-targets", type=int, default=None,
                        help="Number of base points to evaluate TN-SHAP on.")
    parser.add_argument("--eval-batch-size", type=int, default=4096,
                        help="Batch size for R² sanity check.")
    args = parser.parse_args()

    prefix = args.prefix
    print(f"=== TN-SHAP eval for PREFIX={prefix} with MLP(tφ(x)) surrogate ===")

    # Load targets + metadata
    targets = torch.load(f"{prefix}_tnshap_targets.pt", map_location=DEVICE)
    X_targets = targets["x_base"].to(DEVICE)
    t_nodes   = targets["t_nodes"].to(DEVICE)
    X_all_path = targets["X_all_path"].to(DEVICE)
    Y_all     = targets["Y_all"].to(DEVICE)
    X_base_all = targets["X_base_all"].to(DEVICE)
    t_all     = targets["t_all"].to(DEVICE)
    is_h_all  = targets["is_h_all"].to(DEVICE)
    i_all     = targets["i_all"].to(DEVICE)
    max_degree_saved = int(targets["max_degree"])
    mlp_hidden = int(targets.get("mlp_hidden", 32))
    fmap_name = targets.get("feature_map", "scalar_mlp3_plus_one_tphi")

    if args.n_targets is not None:
        X_targets = X_targets[: args.n_targets]

    N_TARGETS = X_targets.shape[0]
    D         = X_targets.shape[1]

    max_degree = args.max_degree if args.max_degree is not None else max_degree_saved
    max_degree = int(max_degree)
    assert max_degree + 1 <= t_nodes.shape[0]

    print(f"N_TARGETS={N_TARGETS}, D={D}, max_degree={max_degree}")
    print(f"feature_map={fmap_name}, mlp_hidden={mlp_hidden}\n")

    # Load teacher and surrogate
    teacher = load_teacher(prefix)
    if isinstance(teacher, torch.nn.Module):
        teacher.to(DEVICE)

    mps, phi_module = load_mps_and_phi(prefix)

    def eval_fn_teacher(x_batch: torch.Tensor) -> torch.Tensor:
        x_batch = x_batch.to(DEVICE)
        with torch.no_grad():
            y = teacher(x_batch)
            if y.ndim > 1:
                y = y.squeeze(-1)
        return y

    # ---- R² sanity check on path-aug dataset ----
    teacher_pred_all = eval_fn_teacher(X_all_path)
    r2_teacher = r2_score(Y_all, teacher_pred_all)

    D_aug = D * 4
    eval_ds = TensorDataset(X_base_all, Y_all, t_all, is_h_all, i_all)
    eval_loader = DataLoader(eval_ds, batch_size=args.eval_batch_size,
                             shuffle=False, drop_last=False)

    y_true_list = []
    y_pred_list = []

    mps.eval()
    phi_module.eval()
    with torch.no_grad():
        for xb_base, yb, tb, is_hb, ib in eval_loader:
            xb_base = xb_base.to(DEVICE)
            yb = yb.to(DEVICE)
            tb = tb.to(DEVICE)
            is_hb = is_hb.to(DEVICE)
            ib = ib.to(DEVICE)

            Zb = build_phi_t_features_base(xb_base, tb, is_hb, ib, phi_module)
            y_m = mps(Zb).squeeze(-1)

            y_true_list.append(yb)
            y_pred_list.append(y_m)

    Y_true_all = torch.cat(y_true_list, dim=0)
    mps_all_pred = torch.cat(y_pred_list, dim=0)

    print(f"Teacher vs path-aug ground truth: R2 = {r2_teacher:.4f}")
    r2_mps = r2_score(Y_true_all, mps_all_pred)
    print(f"MPS+MLP(tφ) vs path-aug truth:    R2 = {r2_mps:.4f}\n")

    # ---- TN-SHAP comparison ----
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

        # Teacher TN-SHAP
        t0 = time.time()
        phi_teacher = tnshap_vandermonde_teacher(
            eval_fn_teacher, x0, max_degree=max_degree, t_nodes=t_nodes
        )
        t1 = time.time()
        t_teacher = t1 - t0

        # Surrogate TN-SHAP
        t2 = time.time()
        phi_mps = tnshap_vandermonde_mps(
            mps, phi_module, x0, max_degree=max_degree, t_nodes=t_nodes
        )
        t3 = time.time()
        t_mps = t3 - t2

        t_teacher_list.append(t_teacher)
        t_mps_list.append(t_mps)

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

    print("\n==== Shapley Support Accuracy on S ====")
    print(f"Teacher Mean Acc: {mean_acc_teacher:.4f} ± {std_acc_teacher:.4f}")
    print(f"MPS Mean Acc:     {mean_acc_mps:.4f} ± {std_acc_mps:.4f}")

    t_teacher_tensor = torch.tensor(t_teacher_list, device=DEVICE)
    t_mps_tensor     = torch.tensor(t_mps_list, device=DEVICE)

    mean_t_teacher = t_teacher_tensor.mean().item()
    std_t_teacher  = t_teacher_tensor.std(unbiased=True).item() if len(t_teacher_tensor) > 1 else 0.0
    mean_t_mps = t_mps_tensor.mean().item()
    std_t_mps  = t_mps_tensor.std(unbiased=True).item() if len(t_mps_tensor) > 1 else 0.0

    print("\n==== Per-point TN-SHAP evaluation time ====")
    print(f"Teacher: mean = {mean_t_teacher*1000:.2f} ms, std = {std_t_teacher*1000:.2f} ms")
    print(f"MPS+MLP: mean = {mean_t_mps*1000:.2f} ms, std = {std_t_mps*1000:.2f} ms")


if __name__ == "__main__":
    main()
