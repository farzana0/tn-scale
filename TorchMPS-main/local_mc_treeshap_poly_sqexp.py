#!/usr/bin/env python3
"""
local_mc_treeshap_poly_sqexp.py

End-to-end script.

For each of n_samples iterations:

  1) Sample ONE base point x0.
  2) Construct local TN-SHAP-style path-augmented dataset in ORIGINAL x-space:
       - h(t) = f(t * x0)
       - g_on_i(t)
       - g_off_i(t)
  3) Train local surrogate models:
       - DecisionTreeRegressor
       - RandomForestRegressor
       - XGBRegressor (optional)
  4) Measure local R²
  5) Compute TreeSHAP values at x0
  6) Report top-K overlap with last K features (true support)

Finally print mean ± std over all iterations.

Works for:
  - poly5_D50, poly10_D50, poly5_D100, poly10_D100
  - sqexp_D50, sqexp_D100
"""

import argparse
import time
import numpy as np
import torch

from poly_teacher import load_teacher, load_data, DEVICE

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Optional deps
try:
    import shap
    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False

try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def chebyshev_nodes_scaled(n_nodes, t_min=0.0, t_max=1.0, device=None, dtype=torch.float32):
    k = torch.arange(1, n_nodes + 1, dtype=torch.float64, device=device)
    u = torch.cos((2.0 * k - 1.0) / (2.0 * n_nodes) * torch.pi)
    u01 = (u + 1.0) / 2.0
    t = t_min + (t_max - t_min) * u01
    return t.to(dtype).to(device)


def get_shap_values(explainer, x_np):
    vals = explainer.shap_values(x_np)
    vals = np.array(vals)
    if vals.ndim == 3:
        vals = vals[0]
    return vals


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prefix", type=str, default="sqexp_D50")
    parser.add_argument("--max-degree", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--support-size", type=int, default=17)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--t-min", type=float, default=0.2)
    parser.add_argument("--t-max", type=float, default=1.0)

    # Tree
    parser.add_argument("--tree-max-depth", type=int, default=6)
    parser.add_argument("--tree-min-samples-leaf", type=int, default=5)

    # Random Forest
    parser.add_argument("--rf-n-estimators", type=int, default=200)
    parser.add_argument("--rf-max-depth", type=int, default=8)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)

    # XGBoost
    parser.add_argument("--xgb-n-estimators", type=int, default=200)
    parser.add_argument("--xgb-max-depth", type=int, default=4)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("==== Config ====")
    print(args)

    if not HAVE_SHAP:
        raise RuntimeError("shap is required: pip install shap")
    if not HAVE_XGB:
        print("xgboost not installed; skipping XGB model.")

    # -------------------------------------------------
    # Load data + teacher
    # -------------------------------------------------
    teacher = load_teacher(args.prefix)
    if isinstance(teacher, torch.nn.Module):
        teacher.to(DEVICE).eval()

    x_train, _, _, _ = load_data(args.prefix)
    x_train = x_train.to(DEVICE)

    N_total, D = x_train.shape
    print(f"Data: N={N_total}, D={D}")

    # Path nodes
    t_nodes = chebyshev_nodes_scaled(
        n_nodes=args.max_degree + 1,
        t_min=args.t_min,
        t_max=args.t_max,
        device=DEVICE,
        dtype=x_train.dtype,
    )

    K = min(args.support_size, D)
    S = np.arange(D - K, D)

    print(f"True support: last {K} dims")

    # Storage
    stats = {
        "tree": {"r2": [], "acc": []},
        "rf": {"r2": [], "acc": []},
        "xgb": {"r2": [], "acc": []},
    }

    # -------------------------------------------------
    # Main loop
    # -------------------------------------------------
    for it in range(args.n_samples):
        idx0 = np.random.randint(0, N_total)
        x0 = x_train[idx0]

        N_local = (1 + 2 * D) * len(t_nodes)
        X = torch.zeros((N_local, D), device=DEVICE)
        Y = torch.zeros((N_local,), device=DEVICE)

        j = 0
        with torch.no_grad():
            for t in t_nodes:
                base = t * x0

                # h
                X[j] = base
                Y[j] = teacher(base.unsqueeze(0)).squeeze()
                j += 1

                # g_on
                tmp = base.repeat(D, 1)
                tmp[torch.arange(D), torch.arange(D)] = x0
                X[j:j + D] = tmp
                Y[j:j + D] = teacher(tmp).squeeze()
                j += D

                # g_off
                tmp = base.repeat(D, 1)
                tmp[torch.arange(D), torch.arange(D)] = 0.0
                X[j:j + D] = tmp
                Y[j:j + D] = teacher(tmp).squeeze()
                j += D

        X_np = X.cpu().numpy()
        Y_np = Y.cpu().numpy()
        x0_np = x0.cpu().numpy().reshape(1, -1)

        # --- Models ---
        models = {
            "tree": DecisionTreeRegressor(
                max_depth=args.tree_max_depth,
                min_samples_leaf=args.tree_min_samples_leaf,
                random_state=args.seed,
            ),
            "rf": RandomForestRegressor(
                n_estimators=args.rf_n_estimators,
                max_depth=args.rf_max_depth,
                min_samples_leaf=args.rf_min_samples_leaf,
                n_jobs=-1,
                random_state=args.seed,
            ),
        }

        if HAVE_XGB:
            models["xgb"] = XGBRegressor(
                n_estimators=args.xgb_n_estimators,
                max_depth=args.xgb_max_depth,
                learning_rate=args.xgb_learning_rate,
                objective="reg:squarederror",
                n_jobs=-1,
                random_state=args.seed,
            )

        for name, model in models.items():
            model.fit(X_np, Y_np)
            y_pred = model.predict(X_np)
            r2 = r2_score(Y_np, y_pred)

            expl = shap.TreeExplainer(model)
            phi = get_shap_values(expl, x0_np)[0]

            order = np.argsort(-np.abs(phi))
            topK = order[:K]
            hit = np.isin(topK, S).sum() / K

            stats[name]["r2"].append(r2)
            stats[name]["acc"].append(hit)

        if it < 3:
            print(f"\nIter {it+1}/{args.n_samples}")
            for name in models.keys():
                print(
                    f"  {name}: "
                    f"R2={stats[name]['r2'][-1]:.3f}, "
                    f"acc={stats[name]['acc'][-1]:.3f}"
                )

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------
    print("\n===== SUMMARY =====")
    for name in stats.keys():
        if len(stats[name]["r2"]) == 0:
            continue
        r2s = np.array(stats[name]["r2"])
        accs = np.array(stats[name]["acc"])
        print(f"\nModel: {name}")
        print(f"  R2  : mean={r2s.mean():.4f}, std={r2s.std():.4f}")
        print(f"  ACC : mean={accs.mean():.4f}, std={accs.std():.4f}")


if __name__ == "__main__":
    main()
