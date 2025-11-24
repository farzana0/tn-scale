#!/usr/bin/env python3
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchmps import MPS

CKPT_PATH = "mps_with_featuremap.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- CONFIG (must match training) ----------------
D = 50
M_NODES = 20
K_POINTS = 10
FEATURE_DIM = 3   # [1, x, x^2] inside MPS


# ---------------- Chebyshev nodes (for TN-Shap) ----------------

def chebyshev_nodes_01(m: int) -> np.ndarray:
    """
    Chebyshev nodes mapped from [-1,1] to [0,1], as in training.
    Returns np.ndarray for TN-Shap Vandermonde.
    """
    if m <= 0:
        return np.zeros((0,), dtype=np.float32)
    k = np.arange(m, dtype=np.float64)
    nodes = np.cos((2 * k + 1) * np.pi / (2 * m))
    t = (nodes + 1.0) * 0.5
    return t.astype(np.float32)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------- Sparse polynomial teacher & dataset (same as training) ----------------

def sparse_poly_f(x, S, a, b):
    """
    x: [N, D]
    S: list of indices
    a: [|S|]
    b: [|S|, |S|] with b[i, :i+1] = 0 used in training
    """
    xS = x[:, S]
    lin = (xS * a).sum(dim=1)
    quad = torch.zeros_like(lin)
    k = len(S)
    for i in range(k):
        for j in range(i + 1, k):
            quad += b[i, j] * xS[:, i] * xS[:, j]
    return lin + quad


def build_masked_dataset(X_targets, S, a, b):
    """
    Mirror of the training builder, but using the checkpointed X_targets, S, a, b.
    X_targets: [K_POINTS, D]  (from ckpt)
    """
    # Chebyshev nodes in [0,1]
    t = torch.from_numpy(chebyshev_nodes_01(M_NODES)).to(DEVICE)      # [M_NODES]
    K = X_targets.shape[0]

    # Base path: x(t) = t * x_base, flattened over all targets
    base = (t[:, None] * X_targets[:, None, :]).reshape(-1, D)        # [M_NODES * K, D]

    X_list = []
    y_list = []

    # Unmasked
    X_list.append(base)
    y_list.append(sparse_poly_f(base, S, a, b))

    # Masked per feature i
    for i in range(D):
        xm = base.clone()
        xm[:, i] = 0.0
        X_list.append(xm)
        y_list.append(sparse_poly_f(xm, S, a, b))

    X = torch.cat(X_list, 0)
    y = torch.cat(y_list, 0)

    return X, y


def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float((1 - ss_res / (ss_tot + 1e-12)).item())


def evaluate_mps_r2_on_masked_grid(mps, X_targets, S, a, b):
    """
    Uses the *same* X_targets and masking logic as in training
    to compute R² of the MPS surrogate against the true sparse polynomial.
    """
    X_targets = X_targets.to(DEVICE)
    a = a.to(DEVICE)
    b = b.to(DEVICE)

    X, y = build_masked_dataset(X_targets, S, a, b)
    mps.eval()
    with torch.no_grad():
        y_pred = mps(X).squeeze(-1)
    r2 = r2_score(y, y_pred)
    print(f"MPS R² on masked Chebyshev grid (train-style dataset): {r2:.4f}")
    return r2


# ---------------- Path builder (for TN-Shap) ----------------

def _build_mps_input_path(
    x_base: np.ndarray,
    t: torch.Tensor,
    D_orig: int,
    D_ext: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build input X(t) for the MPS along the path x(t) = t * x_base.

    Training used:
        input_dim = D_orig
        feature_dim = FEATURE_DIM
        feature_map: x -> [1, x, x^2] (inside MPS)

    So here:
        D_ext == D_orig, and we just feed x(t) of dimension D_orig.

    (The function is written generically to also handle explicit [1,x,x^2] cases,
    but in this experiment we are in the D_ext == D_orig branch.)
    """
    x_base_t = torch.tensor(x_base, dtype=torch.float32, device=device)  # [D_orig]
    t = t.to(device).float()                                             # [m]
    m = int(t.numel())

    # Original path in feature space: [m, D_orig]
    x_path = t.unsqueeze(1) * x_base_t.unsqueeze(0)                      # [m, D_orig]

    # Case 1: model expects only original features (internal feature map)
    if D_ext == D_orig:
        return x_path

    # Case 1b: original features + global bias 1
    if D_ext == D_orig + 1:
        ones_tail = torch.ones(m, 1, device=device, dtype=torch.float32)
        return torch.cat([x_path, ones_tail], dim=1)

    # Case 2: explicit [1, x, x^2] per feature
    if D_ext == 3 * D_orig or D_ext == 3 * D_orig + 1:
        x_path_sq = x_path * x_path
        X_poly = torch.empty(m, 3 * D_orig, device=device, dtype=torch.float32)
        for i in range(D_orig):
            j0 = 3 * i
            j1 = j0 + 1
            j2 = j0 + 2
            X_poly[:, j0] = 1.0
            X_poly[:, j1] = x_path[:, i]
            X_poly[:, j2] = x_path_sq[:, i]

        if D_ext == 3 * D_orig + 1:
            ones_tail = torch.ones(m, 1, device=device, dtype=torch.float32)
            Xg = torch.cat([X_poly, ones_tail], dim=1)
        else:
            Xg = X_poly
        return Xg

    # Fallback: mismatch between checkpoint and assumptions
    raise ValueError(
        f"Unsupported (D_orig={D_orig}, D_ext={D_ext}) combination in "
        f"_build_mps_input_path. Expected one of "
        f"D_ext in {{D, D+1, 3D, 3D+1}}."
    )


# ---------------- TN-Shap k=1 on MPS (degrees in t) ----------------

@torch.no_grad()
def tn_selector_k1_sharedgrid_mps_poly(
    model: torch.nn.Module,
    x_base: np.ndarray,
    t_nodes: np.ndarray,
    *,
    D_orig: int,
    D_ext: int,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, dict]:
    """
    TN-Shap style selector-path estimator (k=1) for an MPS whose effective
    feature map is polynomial [1, x, x^2] in each coordinate.

    We parametrize the path in original feature space:
        x(t) = t * x_base

    X(t) for the MPS is built by `_build_mps_input_path`.

    For each feature i:
      - c_empty(t): polynomial coefficients of f(x(t)) in t
      - c_i(t)    : polynomial of f(x(t)) with feature i masked (x_i(t) -> 0)
      - c^{ {i} } = c_empty - c_i
      - φ_i = sum_{r>=1} c^{ {i} }[r] / r
    """
    device = device or DEVICE
    x_base = np.asarray(x_base, np.float32).ravel()
    assert x_base.shape[0] == D_orig
    d = D_orig
    k = 1

    # Grid + Vandermonde inverse over t
    t = torch.tensor(np.asarray(t_nodes, np.float32), device=device)   # [m]
    m = int(t.numel())
    V = torch.vander(t, N=m, increasing=True)                          # [m, m]

    _sync()
    t0 = time.perf_counter()
    Vinv = torch.linalg.inv(V)                                        # [m, m]
    _sync()
    t_solve = time.perf_counter() - t0

    # ----- Base path -----
    Xg = _build_mps_input_path(x_base, t, D_orig, D_ext, device)      # [m, D_ext]

    _sync()
    t0 = time.perf_counter()
    yg = model(Xg)
    if yg.ndim > 1:
        yg = yg.squeeze(-1)
    _sync()
    t_eval = time.perf_counter() - t0

    c_empty = (Vinv @ yg.unsqueeze(1)).squeeze(1)                      # [m]
    coeffs = {(): c_empty.detach().cpu().numpy().astype(np.float64)}

    # ----- Masked paths for each feature i -----
    subs = [(i,) for i in range(d)]
    if subs:
        Xh = Xg.repeat(len(subs), 1)                                   # [d*m, D_ext]

        for r, S in enumerate(subs):
            i = S[0]

            if D_ext == D_orig or D_ext == D_orig + 1:
                # Mask feature i by zeroing its original coordinate in all t
                Xh[r * m:(r + 1) * m, i] = 0.0

            elif D_ext == 3 * D_orig or D_ext == 3 * D_orig + 1:
                # Explicit [1, x, x^2] blocks: masked feature i -> [1, 0, 0]
                j0 = 3 * i
                j1 = j0 + 1
                j2 = j0 + 2
                Xh[r * m:(r + 1) * m, j0] = 1.0
                Xh[r * m:(r + 1) * m, j1] = 0.0
                Xh[r * m:(r + 1) * m, j2] = 0.0

            else:
                raise ValueError(
                    f"Unsupported (D_orig={D_orig}, D_ext={D_ext}) in masking logic."
                )

        _sync()
        t0 = time.perf_counter()
        yh = model(Xh)
        if yh.ndim > 1:
            yh = yh.squeeze(-1)
        yh = yh.view(len(subs), m)                                     # [d, m]
        _sync()
        t_eval += time.perf_counter() - t0

        _sync()
        t0 = time.perf_counter()
        cS = (yh @ Vinv.T)                                             # [d, m]
        _sync()
        t_solve += time.perf_counter() - t0

        for S, c in zip(subs, cS):
            coeffs[S] = c.detach().cpu().numpy().astype(np.float64)

    # ----- Compute φ_i -----
    phi = np.zeros(d, dtype=np.float64)

    # weights[r] = 1 / r for r >= 1 (k=1)
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


def tnshap_order1_path_mps_poly(
    model: torch.nn.Module,
    x_base: np.ndarray,
    m_nodes: int,
    D_orig: int,
    D_ext: int,
    *,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, dict]:
    t_nodes = chebyshev_nodes_01(m_nodes)
    return tn_selector_k1_sharedgrid_mps_poly(
        model,
        x_base,
        t_nodes,
        D_orig=D_orig,
        D_ext=D_ext,
        device=device,
    )


# ---------------- TN-Shap support recovery ----------------

def evaluate_tnshap_support_recovery_poly(
    model: torch.nn.Module,
    X_targets: torch.Tensor,   # [K, D_orig]
    S_true,
    m_nodes: int,
    D_ext: int,
    device: torch.device,
) -> Tuple[float, float]:
    """
    For each target x in X_targets:
      - compute TN-Shap φ_i (polynomial path in t, with feature-mask)
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
        # x = X_targets[r].numpy()
        x = X_targets[r].detach().cpu().numpy()

        phi, timing = tnshap_order1_path_mps_poly(
            model,
            x,
            m_nodes,
            D_orig=D_orig,
            D_ext=D_ext,
            device=device,
        )

        scores = np.abs(phi)
        topk = np.argsort(-scores)[:k]
        topk_set = set(topk.tolist())

        acc = 1.0 if topk_set == S_set else 0.0
        accuracies.append(acc)
        times.append(timing["t_total_s"])

    accuracy = float(np.mean(accuracies)) if accuracies else 0.0
    avg_time = float(np.mean(times)) if times else 0.0
    return accuracy, avg_time


# ---------------- Main ----------------

def main():
    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint from {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    cfg = ckpt["config"]
    D_ckpt = cfg["D"]
    feature_dim_ckpt = cfg["feature_dim"]
    bond_dim = cfg["bond_dim"]

    assert D_ckpt == D, f"Config D={D_ckpt} does not match eval D={D}"
    assert feature_dim_ckpt == FEATURE_DIM, (
        f"Config feature_dim={feature_dim_ckpt} does not match eval FEATURE_DIM={FEATURE_DIM}"
    )

    D_ext = D_ckpt   # MPS input_dim (original feature space)

    S_true = ckpt["S"]                       # list[int]
    a = ckpt["a"]                            # [|S|]
    b = ckpt["b"]                            # [|S|, |S|]
    X_targets = ckpt["X_targets"]            # [K_POINTS, D]
    X_targets = X_targets.to(DEVICE)
    assert X_targets.shape[0] == K_POINTS
    assert X_targets.shape[1] == D

    print(f"D = {D}, input_dim (D_ext) = {D_ext}, bond_dim = {bond_dim}")
    print(f"M_NODES = {M_NODES}, K_POINTS = {K_POINTS}")
    print(f"True active set S: {S_true}")
    print("MPS uses internal polynomial feature map [1, x, x^2].")

    # Rebuild MPS exactly as in training
    mps = MPS(
        input_dim=D_ext,
        output_dim=1,
        bond_dim=bond_dim,
        adaptive_mode=False,
        periodic_bc=False,
        feature_dim=feature_dim_ckpt,
    ).to(DEVICE)

    # Re-register the same feature map
    def poly_feature_map(x: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.ones_like(x), x, x ** 2], dim=-1)

    mps.register_feature_map(poly_feature_map)
    mps.parallel_eval = True

    mps.load_state_dict(ckpt["state_dict"])
    mps.eval()

    # ---- Surrogate R² on the masked Chebyshev grid ----
    print("\nEvaluating MPS surrogate quality on the masked Chebyshev grid...\n")
    evaluate_mps_r2_on_masked_grid(mps, X_targets, S_true, a, b)

    # ---- TN-Shap support recovery on the K target points ----
    print("\nEvaluating TN-Shap (k=1 path integral with polynomial map) "
          "support recovery on the target points...\n")
    t0 = time.time()
    acc, avg_time = evaluate_tnshap_support_recovery_poly(
        mps, X_targets, S_true, M_NODES, D_ext=D_ext, device=DEVICE
    )
    elapsed = time.time() - t0

    print(f"Exact support recovery accuracy (top-|S| == S): {acc:.3f}")
    print(f"Average TN-Shap eval time per point: {avg_time*1000:.2f} ms")
    print(f"Total evaluation time over {K_POINTS} points: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
