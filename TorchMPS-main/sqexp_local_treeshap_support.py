#!/usr/bin/env python3
"""
sqexp_singlepoint_mc_treeshap.py

For the sqexp teacher:

We run N_samples iterations. In each iteration:

  1. Sample ONE base point x0 from the training data.
  2. Build a local TN-SHAP-style path-augmented dataset in ORIGINAL x-space:

        For Chebyshev nodes t in [0.2, 1.0]:

          - h(t)        = f(t * x0)
          - g_on_i(t)   = f(x_on_i(t)),
                          x_on_i(t)[j]  = x0[j] if j == i else t * x0[j]
          - g_off_i(t)  = f(x_off_i(t)),
                          x_off_i(t)[j] = 0     if j == i else t * x0[j]

  3. Train a DecisionTreeRegressor on (X_local, Y_local).
  4. Compute decision-tree R² on this local dataset.
  5. Compute TreeSHAP values on the single base point x0.
  6. Assume "true support" is the LAST 17 features: {D-17, ..., D-1}.
     Rank features by |phi|, take top-17, and compute:
         hit = |top-17 ∩ S| / 17

We then report:

  - Mean and std of decision-tree R² over all iterations.
  - Mean and std of TreeSHAP top-17 support accuracy over all iterations.
"""

import argparse
import time

import numpy as np
import torch

from poly_teacher import load_teacher, load_data, DEVICE

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score as sk_r2_score

try:
    import shap
    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False


# -----------------------
# Utilities
# -----------------------

def chebyshev_nodes_scaled(n_nodes, t_min=0.2, t_max=1.0, device=None, dtype=torch.float32):
    """
    Chebyshev nodes in [t_min, t_max] on given device / dtype.
    """
    k = torch.arange(1, n_nodes + 1, dtype=torch.float64, device=device)
    u = torch.cos((2.0 * k - 1.0) / (2.0 * n_nodes) * torch.pi)
    u01 = (u + 1.0) / 2.0
    t = t_min + (t_max - t_min) * u01
    return t.to(dtype).to(device)


# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix", type=str, default="sqexp_D50",
        help="Prefix to load sqexp teacher/data (e.g. 'sqexp_D50')."
    )
    parser.add_argument(
        "--max-degree", type=int, default=10,
        help="Polynomial degree in t (=> N_T_NODES = max_degree + 1)."
    )
    parser.add_argument(
        "--n-samples", type=int, default=100,
        help="Number of iterations (one base point per iteration)."
    )
    parser.add_argument("--seed", type=int, default=None)

    # Decision tree options
    parser.add_argument("--tree-max-depth", type=int, default=6,
                        help="max_depth for local DecisionTreeRegressor.")
    parser.add_argument("--tree-min-samples-leaf", type=int, default=5,
                        help="min_samples_leaf for local DecisionTreeRegressor.")

    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    print(f"Device: {DEVICE}")
    print(f"Prefix: {args.prefix}")
    print(f"N_samples (iterations)={args.n_samples}, max_degree={args.max_degree}")
    print(f"Decision Tree: max_depth={args.tree_max_depth}, "
          f"min_samples_leaf={args.tree_min_samples_leaf}")
    if not HAVE_SHAP:
        print("⚠️ shap library not found. TreeSHAP attributions will be skipped.")
        return

    # ---------------------------------------------------------
    # 1) Load teacher and original data
    # ---------------------------------------------------------
    teacher = load_teacher(args.prefix)
    if isinstance(teacher, torch.nn.Module):
        teacher.to(DEVICE)

    x_train, y_train, x_test, y_test = load_data(args.prefix)
    x_train = x_train.to(DEVICE)

    N_total, D = x_train.shape
    print(f"Training data: N_total={N_total}, D={D}")

    # Chebyshev nodes for paths
    N_T_NODES = args.max_degree + 1
    t_nodes = chebyshev_nodes_scaled(
        n_nodes=N_T_NODES,
        t_min=0.2,
        t_max=1.0,
        device=DEVICE,
        dtype=x_train.dtype,
    )
    print(f"Using N_T_NODES={N_T_NODES} Chebyshev nodes in t for path augmentation.")

    # True support = last 17 features (or fewer if D < 17)
    K = min(17, D)
    S = np.arange(D - K, D, dtype=int)
    print(f"\nAssuming true support S is the last {K} features: {S.tolist()}")

    # Storage for metrics over iterations
    r2_list = []
    acc_list = []

    t_all_start = time.time()

    # ---------------------------------------------------------
    # 2) Loop over n-samples iterations
    # ---------------------------------------------------------
    for it in range(args.n_samples):
        it_start = time.time()

        # 2.1 Sample ONE base point x0
        idx0 = torch.randint(low=0, high=N_total, size=(1,), device=DEVICE).item()
        x0 = x_train[idx0]  # (D,)

        # 2.2 Build local path-augmented dataset for this x0
        # Size: (1 + 2D) * N_T_NODES
        N_local = (1 + 2 * D) * N_T_NODES
        X_local = torch.zeros(N_local, D, device=DEVICE, dtype=x0.dtype)
        Y_local = torch.zeros(N_local, device=DEVICE, dtype=x0.dtype)

        teacher.eval()
        idx = 0

        with torch.no_grad():
            for t in t_nodes:
                base = t * x0  # (D,)

                # h-path
                x_h = base
                y_h = teacher(x_h.unsqueeze(0)).squeeze(0)
                if y_h.ndim > 0:
                    y_h = y_h.squeeze(-1)
                X_local[idx] = x_h
                Y_local[idx] = y_h
                idx += 1

                # g_on_i paths
                x_on_batch = base.unsqueeze(0).expand(D, -1).clone()  # (D, D)
                x_on_batch[torch.arange(D), torch.arange(D)] = x0
                y_on_batch = teacher(x_on_batch)
                if y_on_batch.ndim > 1:
                    y_on_batch = y_on_batch.squeeze(-1)
                X_local[idx:idx + D] = x_on_batch
                Y_local[idx:idx + D] = y_on_batch
                idx += D

                # g_off_i paths
                x_off_batch = base.unsqueeze(0).expand(D, -1).clone()  # (D, D)
                x_off_batch[torch.arange(D), torch.arange(D)] = 0.0
                y_off_batch = teacher(x_off_batch)
                if y_off_batch.ndim > 1:
                    y_off_batch = y_off_batch.squeeze(-1)
                X_local[idx:idx + D] = x_off_batch
                Y_local[idx:idx + D] = y_off_batch
                idx += D

        assert idx == N_local, "Index mismatch when building local path-aug dataset."

        # 2.3 Train local decision tree
        X_np = X_local.detach().cpu().numpy()
        y_np = Y_local.detach().cpu().numpy()

        tree = DecisionTreeRegressor(
            max_depth=args.tree_max_depth,
            min_samples_leaf=args.tree_min_samples_leaf,
            random_state=args.seed,
        )
        tree.fit(X_np, y_np)

        # R² on local dataset
        y_pred = tree.predict(X_np)
        r2_it = sk_r2_score(y_np, y_pred)
        r2_list.append(r2_it)

        # 2.4 TreeSHAP on x0
        explainer = shap.TreeExplainer(tree)
        x0_np = x0.detach().cpu().unsqueeze(0).numpy()  # (1, D)
        shap_values = explainer.shap_values(x0_np)
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[0]  # (1, D)
        phi = shap_values[0]  # (D,)

        # 2.5 Top-17 overlap with last 17 features
        order = np.argsort(-np.abs(phi))
        topK = order[:K]
        hit_frac = np.isin(topK, S).sum() / float(K)
        acc_list.append(hit_frac)

        it_time = time.time() - it_start

        if it < 5:
            print(f"\n[Iteration {it+1}/{args.n_samples}]")
            print(f"  Local R²: {r2_it:.4f}")
            print(f"  Top-{K} SHAP indices: {topK.tolist()}")
            print(f"  Top-{K} hit fraction (|top∩S|/|S|): {hit_frac:.3f}")
            print(f"  Iteration time: {it_time:.2f} sec")

    r2_arr = np.array(r2_list)
    acc_arr = np.array(acc_list)

    total_time = time.time() - t_all_start

    # ---------------------------------------------------------
    # 3) Summary
    # ---------------------------------------------------------
    print("\n=== Summary over iterations ===")
    print(f"Number of iterations (samples): {args.n_samples}")
    print(f"\nDecisionTree R² on local path-aug data:")
    print(f"  Mean: {r2_arr.mean():.4f}")
    print(f"  Std : {r2_arr.std():.4f}")
    print(f"\nTreeSHAP top-{K} support hit (last {K} features):")
    print(f"  Mean accuracy: {acc_arr.mean():.4f}")
    print(f"  Std  accuracy: {acc_arr.std():.4f}")
    print(f"\nTotal runtime: {total_time:.2f} sec")


if __name__ == "__main__":
    main()
