#!/usr/bin/env python3
import time
from typing import Optional, Tuple

import numpy as np
import torch
from torchmps import MPS

CKPT_PATH = "mps_with_featuremap.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------- Chebyshev nodes -----------------

def chebyshev_nodes_01(m: int) -> np.ndarray:
    if m <= 0:
        return np.zeros((0,), dtype=np.float32)
    k = np.arange(m, dtype=np.float64)
    nodes = np.cos((2 * k + 1) * np.pi / (2 * m))
    t = (nodes + 1.0) * 0.5
    return t.astype(np.float32)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ----------------- TN-Shap k=1 on MPS with appended 1 -----------------

@torch.no_grad()
def tn_selector_k1_sharedgrid_mps(
    model: torch.nn.Module,
    x_base: np.ndarray,
    t_nodes: np.ndarray,
    *,
    D_orig: int,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, dict]:
    """
    TN-Shap style selector-path estimator (k=1) for MPS with bias-apppended inputs.

    - x_base: original features, shape [D_orig]
    - model input_dim = D_orig + 1
    - last dimension is a constant 1 and is NEVER masked or scaled.

    Path:
      x(t) = t * x_base  on the first D_orig dims, last dim = 1

    For each feature i < D_orig:
      - compute polynomial coefficients for:
          c_empty: base path (no masking)
          c_i:     path with feature i masked to 0 along entire path
      - c^{ {i} } = c_empty - c_i (inclusion–exclusion)
      - φ_i = sum_{r>=1} c^{ {i} }[r] / r

    Returns:
      phi: [D_orig]  (Shapley estimate per original feature)
      timing: dict
    """
    device = device or DEVICE
    x_base = np.asarray(x_base, np.float32).ravel()
    assert x_base.shape[0] == D_orig
    d = D_orig
    k = 1

    # Grid + Vandermonde inverse
    t = torch.tensor(np.asarray(t_nodes, np.float32), device=device)   # [m]
    m = int(t.numel())
    V = torch.vander(t, N=m, increasing=True)  # [m, m]

    _sync(); t0 = time.perf_counter()
    Vinv = torch.linalg.inv(V)                # [m, m]
    _sync(); t_solve = time.perf_counter() - t0

    x_t = torch.tensor(x_base, dtype=torch.float32, device=device)  # [d]

    # ----- Base path with appended 1 -----
    X_path = t.unsqueeze(1) * x_t.unsqueeze(0)    # [m, d]
    ones = torch.ones(m, 1, device=device)
    Xg = torch.cat([X_path, ones], dim=1)        # [m, d+1]

    _sync(); t0 = time.perf_counter()
    yg = model(Xg)
    if yg.ndim > 1: yg = yg.squeeze(-1)
    _sync(); t_eval = time.perf_counter() - t0

    c_empty = (Vinv @ yg.unsqueeze(1)).squeeze(1)   # [m]
    coeffs = {(): c_empty.detach().cpu().numpy().astype(np.float64)}

    # ----- Masked paths for each feature i < D_orig -----
    subs = [(i,) for i in range(d)]
    if subs:
        Xh = Xg.repeat(len(subs), 1)               # [d*m, d+1]
        for r, S in enumerate(subs):
            i = S[0]
            Xh[r * m:(r + 1) * m, i] = 0.0         # mask feature i

        _sync(); t0 = time.perf_counter()
        yh = model(Xh)
        if yh.ndim > 1: yh = yh.squeeze(-1)
        yh = yh.view(len(subs), m)                 # [d, m]
        _sync(); t_eval += time.perf_counter() - t0

        _sync(); t0 = time.perf_counter()
        cS = (yh @ Vinv.T)                         # [d, m]
        _sync(); t_solve += time.perf_counter() - t0

        for S, c in zip(subs, cS):
            coeffs[S] = c.detach().cpu().numpy().astype(np.float64)

    # ----- Compute φ_i -----
    phi = np.zeros(d, dtype=np.float64)

    # weights[r] = 1 / r for r >= 1
    weights = np.zeros(m, dtype=np.float64)
    for r in range(k, m):
        weights[r] = 1.0 / max(r, 1)

    for i in range(d):
        cT = np.zeros(m, dtype=np.float64)
        cT += coeffs[()]         # empty
        cT -= coeffs[(i,)]       # masked i
        phi[i] = float(np.dot(cT[k:], weights[k:]))

    timing = {
        "t_eval_s": float(t_eval),
        "t_solve_s": float(t_solve),
        "t_total_s": float(t_eval + t_solve),
    }
    return phi, timing


def tnshap_order1_path_mps(
    model: torch.nn.Module,
    x_base: np.ndarray,
    m_nodes: int,
    D_orig: int,
    *,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, dict]:
    t_nodes = chebyshev_nodes_01(m_nodes)
    return tn_selector_k1_sharedgrid_mps(model, x_base, t_nodes, D_orig=D_orig, device=device)


# ----------------- Accuracy over the 100 target points -----------------

def evaluate_tnshap_support_recovery(
    model: torch.nn.Module,
    X_targets: torch.Tensor,   # [K, D_orig]
    S_true,
    m_nodes: int,
    device: torch.device,
) -> Tuple[float, float]:
    """
    For each target x in X_targets:
      - compute TN-Shap φ_i
      - rank |φ_i|, take top-|S_true|
      - accuracy = 1 if top-k set == S_true, else 0

    Returns:
      accuracy (mean over K),
      avg_tnshap_time_per_point (seconds)
    """
    D_orig = X_targets.shape[1]
    S_set = set(S_true)
    k = len(S_true)

    accuracies = []
    times = []

    for r in range(X_targets.shape[0]):
        x = X_targets[r].numpy()
        phi, timing = tnshap_order1_path_mps(model, x, m_nodes, D_orig=D_orig, device=device)

        scores = np.abs(phi)
        topk = np.argsort(-scores)[:k]
        topk_set = set(topk.tolist())

        acc = 1.0 if topk_set == S_set else 0.0
        accuracies.append(acc)
        times.append(timing["t_total_s"])

    accuracy = float(np.mean(accuracies)) if accuracies else 0.0
    avg_time = float(np.mean(times)) if times else 0.0
    return accuracy, avg_time


# ----------------- Main -----------------

def main():
    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint from {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    cfg = ckpt["config"]
    D = cfg["D"]
    D_ext = cfg["D_ext"]
    bond_dim = cfg["bond_dim"]
    M_NODES = cfg["M_NODES"]
    K_POINTS = cfg["K_POINTS"]

    S_true = ckpt["active_set_S"]        # list[int]
    X_targets = ckpt["X_targets"]        # [K, D]
    assert X_targets.shape[0] == K_POINTS
    assert X_targets.shape[1] == D

    print(f"D = {D}, D_ext = {D_ext}, bond_dim = {bond_dim}")
    print(f"M_NODES = {M_NODES}, K_POINTS = {K_POINTS}")
    print(f"True active set S: {S_true}")

    # Rebuild MPS
    mps = MPS(
        input_dim=D_ext,
        output_dim=1,
        bond_dim=bond_dim,
        adaptive_mode=False,
        periodic_bc=False,
    ).to(DEVICE)
    mps.load_state_dict(ckpt["state_dict"])
    mps.eval()

    print("\nEvaluating TN-Shap (k=1 path integral) support recovery on the 100 target points...\n")
    t0 = time.time()
    acc, avg_time = evaluate_tnshap_support_recovery(
        mps, X_targets, S_true, M_NODES, device=DEVICE
    )
    elapsed = time.time() - t0

    print(f"Exact support recovery accuracy (top-|S| == S): {acc:.3f}")
    print(f"Average TN-Shap eval time per point: {avg_time*1000:.2f} ms")
    print(f"Total evaluation time over {K_POINTS} points: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
