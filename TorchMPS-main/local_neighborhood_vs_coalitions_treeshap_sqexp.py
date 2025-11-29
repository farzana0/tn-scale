#!/usr/bin/env python3
"""
local_neighborhood_vs_coalitions_treeshap_sqexp.py

Compare 3 local sampling schemes around each evaluation point x0:

  1) PATH       : x(t)      = t * x0
  2) COAL       : x(m)      = m ⊙ x0, m ∈ {0,1}^D
  3) TNSHAP_GI  : clean TN-SHAP-style g_i(0,t), g_i(q,t):
                     - Pick t_i ~ Uniform[t_min, t_max]
                     - base = t_i * x0
                     - x_i^(0)(t_i) = base with coord i = 0
                     - x_i^(q)(t_i) = base with coord i = x0[i]

For each scheme and each eval point x0:
  - Build 2D local samples (2 per feature, D features)
  - Train Tree, RF, XGB
  - Compute TreeSHAP at x0
  - Take top-K features (K = support_size) and compute hit accuracy
  - Record train R^2 and hit accuracy

At the end, print mean ± std over all eval points.

Supports:
  --eval-indices path/to/file.txt
    to fix the evaluation points (one index per line).

This version is tailored for sqexp_* experiments (no RANDOM scheme,
so no NaN issues from exploding teacher values).
"""

import argparse
import numpy as np
import torch
import random

from poly_teacher import load_teacher, load_data, DEVICE

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from xgboost import XGBRegressor
import shap


# -----------------------
# Global seeding
# -----------------------

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------
# Sampling helpers
# -----------------------

def build_path_dataset(x0, teacher, n_points, t_min, t_max):
    """
    PATH scheme:
      - Sample t ~ Uniform[t_min, t_max]
      - X(t) = t * x0
    """
    t = torch.empty(n_points, device=x0.device).uniform_(t_min, t_max)
    X = t.unsqueeze(1) * x0.unsqueeze(0)
    with torch.no_grad():
        y = teacher(X).view(-1)
    return X.cpu().numpy(), y.cpu().numpy()


def build_coalition_dataset(x0, teacher, n_points):
    """
    COAL scheme:
      - Sample masks m ∈ {0,1}^D
      - X(m) = m ⊙ x0
    """
    D = x0.shape[0]
    m = torch.randint(low=0, high=2, size=(n_points, D),
                      device=x0.device, dtype=torch.float32)
    X = m * x0.unsqueeze(0)
    with torch.no_grad():
        y = teacher(X).view(-1)
    return X.cpu().numpy(), y.cpu().numpy()


def build_tnshap_gi_dataset(x0, teacher, t_min, t_max):
    """
    TNSHAP_GI scheme:
      For each feature i:
        - sample t_i ~ Uniform[t_min, t_max]
        - base = t_i * x0
        - x_i^(0)(t_i) = base, with coord i = 0
        - x_i^(q)(t_i) = base, with coord i = x0[i]

      Total points = 2D.
    """
    D = x0.shape[0]
    t = torch.empty(D, device=x0.device).uniform_(t_min, t_max)  # one t per feature
    base = t.unsqueeze(1) * x0.unsqueeze(0)  # shape: (D, D)

    X_list = []

    for i in range(D):
        row = base[i].clone()

        # g_i(0,t): feature i at baseline 0
        x0_version = row.clone()
        x0_version[i] = 0.0
        X_list.append(x0_version)

        # g_i(q,t): feature i at original value x0[i]
        xq_version = row.clone()
        xq_version[i] = x0[i]
        X_list.append(xq_version)

    X = torch.stack(X_list, dim=0)  # shape (2D, D)
    with torch.no_grad():
        y = teacher(X).view(-1)
    return X.cpu().numpy(), y.cpu().numpy()


# -----------------------
# Attribution helpers
# -----------------------

def get_true_support(D, support_size):
    """
    True support = last `support_size` features.
    """
    return np.arange(D - support_size, D)


def shap_hit_accuracy(model, x0_np, support_true, support_size, background_X):
    """
    Compute TreeSHAP values at x0, get top `support_size` features,
    and compute hit accuracy wrt `support_true`.
    """
    explainer = shap.TreeExplainer(model, data=background_X)
    phi = explainer.shap_values(x0_np.reshape(1, -1))
    phi = np.array(phi)

    # Handle possible shapes:
    #  (D,), (1,D), (n_outputs, 1, D)
    if phi.ndim == 1:
        pass
    elif phi.ndim == 2:
        phi = phi[0]
    elif phi.ndim == 3:
        phi = phi.sum(axis=0)[0]

    idx_sorted = np.argsort(-np.abs(phi))
    top_k = idx_sorted[:support_size]
    hits = len(np.intersect1d(top_k, support_true))
    return hits / len(support_true)


def train_and_eval_models(X_train, y_train, x0_np, support_true, support_size):
    """
    Train Tree, RF, XGB on (X_train, y_train) and compute:
      - R^2 on training set
      - SHAP hit accuracy at x0

    Returns dict: { 'tree': (R2, acc), 'rf': (R2, acc), 'xgb': (R2, acc) }
    """
    results = {}

    # Decision Tree
    tree = DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=0)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_train)
    r2 = r2_score(y_train, y_pred)
    acc = shap_hit_accuracy(tree, x0_np, support_true, support_size, X_train)
    results['tree'] = (r2, acc)

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=1,
        random_state=0,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_train)
    r2 = r2_score(y_train, y_pred)
    acc = shap_hit_accuracy(rf, x0_np, support_true, support_size, X_train)
    results['rf'] = (r2, acc)

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        objective='reg:squarederror',
        tree_method='hist',
        random_state=0,
        n_jobs=1,
        subsample=1.0,
        colsample_bytree=1.0,
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_train)
    r2 = r2_score(y_train, y_pred)
    acc = shap_hit_accuracy(xgb, x0_np, support_true, support_size, X_train)
    results['xgb'] = (r2, acc)

    return results


# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, required=True,
                        help="Task prefix, e.g. sqexp_D50, sqexp_D100.")
    parser.add_argument("--support-size", type=int, required=True,
                        help="True support size (e.g., 17 for D=50, 33 for D=100).")
    parser.add_argument("--t-min", type=float, default=0.2,
                        help="Minimum t for PATH / TNSHAP_GI sampling.")
    parser.add_argument("--t-max", type=float, default=1.0,
                        help="Maximum t for PATH / TNSHAP_GI sampling.")
    parser.add_argument("--n-eval", type=int, default=100,
                        help="Maximum number of evaluation points to use.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Global random seed.")
    parser.add_argument("--eval-indices", type=str, default=None,
                        help="Optional: file with fixed evaluation indices, one per line.")
    args = parser.parse_args()

    set_global_seed(args.seed)

    print("==== Config ====")
    print(args)

    # Load teacher and data
    teacher = load_teacher(args.prefix)
    data = load_data(args.prefix)

    # Robust handling of data format
    if isinstance(data, dict):
        if "X_base" in data and "y_base" in data:
            X_all = data["X_base"]
            y_all = data["y_base"]
        elif "X" in data and "y" in data:
            X_all = data["X"]
            y_all = data["y"]
        else:
            keys = list(data.keys())
            X_all = data[keys[0]]
            y_all = data[keys[1]]
    else:
        X_all = data[0]
        y_all = data[1]

    X_all = X_all.to(DEVICE)
    y_all = y_all.to(DEVICE).view(-1)

    N, D = X_all.shape
    print(f"Data: N={N}, D={D}")
    print(f"True support: last {args.support_size} dims")

    # Decide which eval indices to use
    if args.eval_indices is not None:
        idx_list = []
        with open(args.eval_indices, "r") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                idx_list.append(int(line))
        idx_tensor = torch.tensor(idx_list, device=X_all.device, dtype=torch.long)
        idx_tensor = idx_tensor[idx_tensor < N]
        idx_eval = idx_tensor
        N_eval = min(len(idx_eval), args.n_eval)
        idx_eval = idx_eval[:N_eval]
        print(f"Using fixed eval indices from {args.eval_indices}: {idx_eval.cpu().numpy().tolist()}")
    else:
        N_eval = min(args.n_eval, N)
        idx_eval = torch.arange(N_eval, device=X_all.device, dtype=torch.long)
        print(f"Using first {N_eval} points as evaluation set.")

    # True support indices
    support_true = get_true_support(D, args.support_size)

    schemes = ["PATH", "COAL", "TNSHAP_GI"]
    models = ["tree", "rf", "xgb"]

    r2_scores = {scheme: {m: [] for m in models} for scheme in schemes}
    acc_scores = {scheme: {m: [] for m in models} for scheme in schemes}

    for it, idx in enumerate(idx_eval, start=1):
        x0 = X_all[idx]
        x0_np = x0.detach().cpu().numpy()

        print(f"\nEval point {it}/{N_eval} (index {int(idx)})")
        n_train = 2 * D  # total samples for PATH / COAL; TNSHAP_GI naturally has 2D

        # -------- PATH --------
        X_path, y_path = build_path_dataset(
            x0=x0, teacher=teacher,
            n_points=n_train,
            t_min=args.t_min, t_max=args.t_max,
        )
        res_path = train_and_eval_models(X_path, y_path, x0_np, support_true, args.support_size)
        for m in models:
            r2_scores["PATH"][m].append(res_path[m][0])
            acc_scores["PATH"][m].append(res_path[m][1])

        # -------- COAL --------
        X_coal, y_coal = build_coalition_dataset(
            x0=x0, teacher=teacher,
            n_points=n_train,
        )
        res_coal = train_and_eval_models(X_coal, y_coal, x0_np, support_true, args.support_size)
        for m in models:
            r2_scores["COAL"][m].append(res_coal[m][0])
            acc_scores["COAL"][m].append(res_coal[m][1])

        # -------- TNSHAP_GI --------
        X_gi, y_gi = build_tnshap_gi_dataset(
            x0=x0, teacher=teacher,
            t_min=args.t_min, t_max=args.t_max,
        )
        res_gi = train_and_eval_models(X_gi, y_gi, x0_np, support_true, args.support_size)
        for m in models:
            r2_scores["TNSHAP_GI"][m].append(res_gi[m][0])
            acc_scores["TNSHAP_GI"][m].append(res_gi[m][1])

        # Quick per-point summary
        print("  PATH      : " + ", ".join([f"{m}: R2={res_path[m][0]:.3f}, acc={res_path[m][1]:.3f}" for m in models]))
        print("  COAL      : " + ", ".join([f"{m}: R2={res_coal[m][0]:.3f}, acc={res_coal[m][1]:.3f}" for m in models]))
        print("  TNSHAP_GI : " + ", ".join([f"{m}: R2={res_gi[m][0]:.3f}, acc={res_gi[m][1]:.3f}" for m in models]))

    # -------- Final summary --------
    print("\n===== SUMMARY (mean ± std over eval points) =====\n")
    for scheme in schemes:
        print(f"Scheme: {scheme}")
        for m in models:
            r2_arr = np.array(r2_scores[scheme][m])
            acc_arr = np.array(acc_scores[scheme][m])
            print(f"  Model: {m}")
            print(f"    R2  : mean={r2_arr.mean():.4f}, std={r2_arr.std():.4f}")
            print(f"    ACC : mean={acc_arr.mean():.4f}, std={acc_arr.std():.4f}")
        print()


if __name__ == "__main__":
    main()
