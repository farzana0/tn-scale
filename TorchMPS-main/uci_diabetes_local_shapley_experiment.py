#!/usr/bin/env python3
"""
UCI Diabetes: local SHAP comparison with different sampling schemes.

Pipeline:
  1. Load UCI Diabetes regression dataset (10 features).
  2. Standardize features, split into train/test.
  3. Train an MLP regressor on the training set.
  4. Select 100 test points (deterministically: first 100).
  5. For each test point x0, build local datasets using 4 schemes:
       - PATH      : baseline + t * (x0 - baseline), t ~ Uniform[t_min, t_max]
       - COAL      : m ⊙ x0 + (1-m) ⊙ baseline,  m ∈ {0,1}^d
       - RANDOM    : x0 + ε,  ε ~ N(0, σ^2 I)
       - TNSHAP_GI : TN-style g_i(0,t), g_i(q,t) paths
  6. For each local dataset:
       - Train DecisionTreeRegressor, RandomForestRegressor, XGBRegressor.
       - Compute TreeSHAP φ_sur(x0).
  7. For each x0:
       - Compute ground-truth Shapley φ_true(x0) for the MLP by exhaustive
         coalition enumeration (2^d coalitions) with a fixed baseline.
  8. Compare φ_sur vs φ_true via:
       - Pearson correlation over features.
       - Top-k overlap of |φ| (k = ceil(d/3)).
  9. Print mean ± std over the 100 test points for each
     (scheme, model) combination.

Requires:
  - scikit-learn
  - xgboost
  - shap
"""

import argparse
import numpy as np
from math import factorial, ceil

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from xgboost import XGBRegressor
import shap


# -----------------------
# Shapley enumeration for MLP
# -----------------------

def shapley_enumeration(f, x0, baseline):
    """
    Exact Shapley values for a single point x0 via full coalition enumeration.

    f        : callable, f(X) -> y, where X shape is (n_samples, n_features)
    x0       : (d,) np.array, standardized point
    baseline : (d,) np.array, baseline point (e.g., zeros)

    Returns:
      phi : (d,) np.array, exact Shapley values.
    """
    x0 = np.asarray(x0, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    d = x0.shape[0]
    n_subsets = 1 << d  # 2^d

    # Precompute v(S) for all subsets S
    v = np.zeros(n_subsets, dtype=float)

    for mask in range(n_subsets):
        x = baseline.copy()
        # set features in S to x0
        m = mask
        j = 0
        while m:
            if m & 1:
                x[j] = x0[j]
            m >>= 1
            j += 1
        v[mask] = float(f(x.reshape(1, -1))[0])

    # Precompute weights w_k = k!(d-k-1)! / d!
    denom = factorial(d)
    w_k = np.zeros(d, dtype=float)
    for k in range(d):
        w_k[k] = factorial(k) * factorial(d - k - 1) / denom

    # Precompute popcount for all masks
    popcount = np.zeros(n_subsets, dtype=int)
    for mask in range(1, n_subsets):
        popcount[mask] = popcount[mask >> 1] + (mask & 1)

    # Compute Shapley values
    phi = np.zeros(d, dtype=float)
    for j in range(d):
        bit_j = 1 << j
        for mask in range(n_subsets):
            if mask & bit_j:
                continue  # S must NOT contain j
            k = popcount[mask]
            S = mask
            Sj = mask | bit_j
            phi[j] += w_k[k] * (v[Sj] - v[S])

    return phi


# -----------------------
# Sampling schemes
# -----------------------

def build_path_dataset(x0, f, n_points, t_min, t_max, baseline):
    """
    PATH:
      x(t) = baseline + t * (x0 - baseline),  t ~ U[t_min, t_max]
    """
    d = x0.shape[0]
    t = np.random.uniform(t_min, t_max, size=n_points)
    X = baseline[None, :] + t[:, None] * (x0[None, :] - baseline[None, :])
    y = f(X)
    return X, y


def build_coalition_dataset(x0, f, n_points, baseline):
    """
    COAL:
      m ∈ {0,1}^d, sampled uniformly
      x(m) = m ⊙ x0 + (1-m) ⊙ baseline
    """
    d = x0.shape[0]
    m = np.random.randint(0, 2, size=(n_points, d))
    X = m * x0[None, :] + (1 - m) * baseline[None, :]
    y = f(X)
    return X, y


def build_random_dataset(x0, f, n_points, sigma):
    """
    RANDOM:
      x = x0 + ε,  ε ~ N(0, diag(sigma^2))
    sigma: (d,) std for each coordinate.
    """
    d = x0.shape[0]
    noise = np.random.randn(n_points, d) * sigma[None, :]
    X = x0[None, :] + noise
    y = f(X)
    return X, y


def build_tnshap_gi_dataset(x0, f, t_min, t_max, baseline):
    """
    TNSHAP_GI:
      For each feature j:
        - sample t_j ~ U[t_min, t_max]
        - base_j = baseline + t_j * (x0 - baseline)
        - g_j(0,t_j): base_j with coord j = baseline_j
        - g_j(q,t_j): base_j with coord j = x0_j

      Total points = 2d.
    """
    d = x0.shape[0]
    t = np.random.uniform(t_min, t_max, size=d)
    base = baseline[None, :] + t[:, None] * (x0[None, :] - baseline[None, :])  # (d, d)

    X_list = []
    for j in range(d):
        row = base[j].copy()

        # g_j(0,t): feature j at baseline
        x0_version = row.copy()
        x0_version[j] = baseline[j]
        X_list.append(x0_version)

        # g_j(q,t): feature j at original x0
        xq_version = row.copy()
        xq_version[j] = x0[j]
        X_list.append(xq_version)

    X = np.stack(X_list, axis=0)  # (2d, d)
    y = f(X)
    return X, y


# -----------------------
# Comparing surrogate SHAP vs exact SHAP
# -----------------------

def tree_shap_values(model, x0, background_X):
    """
    Compute SHAP values for a single x0 using TreeSHAP.
    """
    explainer = shap.TreeExplainer(model, data=background_X)
    phi = explainer.shap_values(x0.reshape(1, -1))
    phi = np.array(phi)
    # shapes: (d,), (1,d), or (n_outputs, 1, d)
    if phi.ndim == 1:
        return phi
    if phi.ndim == 2:
        return phi[0]
    if phi.ndim == 3:
        return phi.sum(axis=0)[0]
    raise ValueError(f"Unexpected SHAP value shape: {phi.shape}")


def topk_overlap(phi_true, phi_sur, k):
    """Top-k support overlap between |phi_true| and |phi_sur|."""
    idx_true = np.argsort(-np.abs(phi_true))[:k]
    idx_sur = np.argsort(-np.abs(phi_sur))[:k]
    return len(np.intersect1d(idx_true, idx_sur)) / float(k)


# -----------------------
# Main experiment
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-eval", type=int, default=100,
                        help="Number of test points to explain.")
    parser.add_argument("--n-local", type=int, default=200,
                        help="Number of local samples per scheme (except TNSHAP_GI which uses 2d).")
    parser.add_argument("--t-min", type=float, default=0.0,
                        help="Min t for PATH/TNSHAP_GI.")
    parser.add_argument("--t-max", type=float, default=1.0,
                        help="Max t for PATH/TNSHAP_GI.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed.")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # 1. Load and preprocess Diabetes
    data = load_diabetes()
    X = data.data    # already standardized-ish, but we standardize again
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_train, d = X_train.shape
    print(f"Dataset: Diabetes, n_train={n_train}, n_test={X_test.shape[0]}, d={d}")

    # Baseline = zero vector in standardized space
    baseline = np.zeros(d, dtype=float)

    # 2. Train MLP
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=3000,
        random_state=0,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.1,
    )
    mlp.fit(X_train, y_train)
    y_pred_test = mlp.predict(X_test)
    r2_test = r2_score(y_test, y_pred_test)
    print(f"MLP Test R^2: {r2_test:.4f}")

    # Black-box function f(X) that returns predictions
    def f(X_in):
        return mlp.predict(X_in)

    # 3. Pick 100 test points deterministically
    n_eval = min(args.n_eval, X_test.shape[0])
    idx_eval = np.arange(n_eval)
    X_eval = X_test[idx_eval]
    y_eval = y_test[idx_eval]
    print(f"Using {n_eval} test points for explanation.")

    # 4. Prepare metric containers
    schemes = ["PATH", "COAL", "RANDOM", "TNSHAP_GI"]
    models = ["tree", "rf", "xgb"]

    # For each: dict[scheme][model] -> list of metrics
    r2_local = {s: {m: [] for m in models} for s in schemes}
    corr_local = {s: {m: [] for m in models} for s in schemes}
    acc_local = {s: {m: [] for m in models} for s in schemes}

    k_top = ceil(d / 3)

    # For RANDOM scheme: per-feature std from training data
    sigma = 0.1 * np.std(X_train, axis=0, ddof=1)
    sigma[sigma == 0.0] = 1e-6

    # 5. Loop over test points
    for t_idx, x0 in enumerate(X_eval):
        print(f"\n=== Point {t_idx+1}/{n_eval} ===")

        # Exact Shapley for MLP at x0
        phi_true = shapley_enumeration(f, x0, baseline)

        # Sampling schemes
        local_data = {}

        # PATH
        X_path, y_path = build_path_dataset(
            x0, f, n_points=args.n_local,
            t_min=args.t_min, t_max=args.t_max,
            baseline=baseline,
        )
        local_data["PATH"] = (X_path, y_path)

        # COAL
        X_coal, y_coal = build_coalition_dataset(
            x0, f, n_points=args.n_local,
            baseline=baseline,
        )
        local_data["COAL"] = (X_coal, y_coal)

        # RANDOM
        X_rand, y_rand = build_random_dataset(
            x0, f, n_points=args.n_local,
            sigma=sigma,
        )
        local_data["RANDOM"] = (X_rand, y_rand)

        # TNSHAP_GI (2d points)
        X_gi, y_gi = build_tnshap_gi_dataset(
            x0, f, t_min=args.t_min, t_max=args.t_max,
            baseline=baseline,
        )
        local_data["TNSHAP_GI"] = (X_gi, y_gi)

        # For each scheme, fit surrogates and compare SHAP
        for scheme in schemes:
            X_loc, y_loc = local_data[scheme]

            # Tree
            tree = DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=0)
            tree.fit(X_loc, y_loc)
            y_hat = tree.predict(X_loc)
            r2 = r2_score(y_loc, y_hat)
            phi_sur = tree_shap_values(tree, x0, X_loc)
            corr = pearsonr(phi_true, phi_sur)[0] if np.std(phi_true) > 0 and np.std(phi_sur) > 0 else np.nan
            acc = topk_overlap(phi_true, phi_sur, k_top)

            r2_local[scheme]["tree"].append(r2)
            corr_local[scheme]["tree"].append(corr)
            acc_local[scheme]["tree"].append(acc)

            # RF
            rf = RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=1,
                random_state=0,
                n_jobs=-1,
            )
            rf.fit(X_loc, y_loc)
            y_hat = rf.predict(X_loc)
            r2 = r2_score(y_loc, y_hat)
            phi_sur = tree_shap_values(rf, x0, X_loc)
            corr = pearsonr(phi_true, phi_sur)[0] if np.std(phi_true) > 0 and np.std(phi_sur) > 0 else np.nan
            acc = topk_overlap(phi_true, phi_sur, k_top)

            r2_local[scheme]["rf"].append(r2)
            corr_local[scheme]["rf"].append(corr)
            acc_local[scheme]["rf"].append(acc)

            # XGB
            xgb = XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=0,
                n_jobs=1,
                subsample=1.0,
                colsample_bytree=1.0,
            )
            xgb.fit(X_loc, y_loc)
            y_hat = xgb.predict(X_loc)
            r2 = r2_score(y_loc, y_hat)
            phi_sur = tree_shap_values(xgb, x0, X_loc)
            corr = pearsonr(phi_true, phi_sur)[0] if np.std(phi_true) > 0 and np.std(phi_sur) > 0 else np.nan
            acc = topk_overlap(phi_true, phi_sur, k_top)

            r2_local[scheme]["xgb"].append(r2)
            corr_local[scheme]["xgb"].append(corr)
            acc_local[scheme]["xgb"].append(acc)

    # 6. Final summary
    print("\n===== SUMMARY over {} points =====".format(n_eval))
    print("Top-k overlap uses k = {}, features = {}".format(k_top, d))
    for scheme in schemes:
        print(f"\nScheme: {scheme}")
        for m in models:
            r2_arr = np.array(r2_local[scheme][m])
            corr_arr = np.array(corr_local[scheme][m])
            acc_arr = np.array(acc_local[scheme][m])

            print(f"  Model: {m}")
            print(f"    R2(local)      : mean={np.nanmean(r2_arr):.4f}, std={np.nanstd(r2_arr):.4f}")
            print(f"    Corr(φ_true,φ) : mean={np.nanmean(corr_arr):.44f}, std={np.nanstd(corr_arr):.4f}")
            print(f"    Top-{k_top} acc: mean={np.nanmean(acc_arr):.4f}, std={np.nanstd(acc_arr):.4f}")


if __name__ == "__main__":
    main()
