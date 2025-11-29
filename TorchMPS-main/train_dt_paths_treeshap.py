#!/usr/bin/env python3
"""
train_dt_paths_treeshap.py

Train a Decision Tree on the EXACT SAME path-augmented dataset used in train_mps_sqexp_paths.py
and compute Shapley values using TreeSHAP.

This script:
  - Uses the exact same path construction as train_mps_sqexp_paths.py:
      * h(t) = f(t * x) paths
      * g_on_i(t) paths where x_i is fixed
      * g_off_i(t) paths where x_i is set to 0
  - Trains a sklearn DecisionTreeRegressor on this path-augmented data
  - Computes TreeSHAP values on the base points
"""

import argparse
import os
import time

import torch
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score as sklearn_r2_score

from poly_teacher import load_teacher, load_data, DEVICE

# Try to import shap for TreeSHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    raise ImportError("TreeSHAP requires the 'shap' library. Install with: pip install shap")


# -----------------------
# Utilities
# -----------------------

def chebyshev_nodes_scaled(n_nodes, t_min=0.2, t_max=1.0, device=None, dtype=torch.float32):
    """Same as in train_mps_sqexp_paths.py"""
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
        "--prefix", type=str, default="poly5_D50",
        help="Prefix to load teacher/data (e.g. 'poly5_D50', 'sqexp_D50')."
    )
    parser.add_argument(
        "--max-degree", type=int, default=10,
        help="Polynomial degree in t for path augmentation (determines number of t-nodes)."
    )
    parser.add_argument(
        "--n-targets", type=int, default=20,
        help="Number of base points whose paths we include."
    )
    parser.add_argument(
        "--max-depth", type=int, default=10,
        help="Maximum depth of the Decision Tree."
    )
    parser.add_argument(
        "--min-samples-split", type=int, default=2,
        help="Minimum samples required to split an internal node."
    )
    parser.add_argument(
        "--min-samples-leaf", type=int, default=1,
        help="Minimum samples required to be at a leaf node."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for Decision Tree."
    )
    args = parser.parse_args()

    print(f"=== Training Decision Tree on Path-Augmented Data + TreeSHAP ===")
    print(f"Device: {DEVICE}")
    print(f"Prefix: {args.prefix}")
    print(f"Path augmentation: max_degree={args.max_degree}, n_targets={args.n_targets}")
    print(f"Decision Tree params: max_depth={args.max_depth}, "
          f"min_samples_split={args.min_samples_split}, "
          f"min_samples_leaf={args.min_samples_leaf}, seed={args.seed}")

    # ---------------------------------------------------------
    # 1) Load teacher and original data
    # ---------------------------------------------------------
    print("\n--- Loading teacher and data ---")
    teacher = load_teacher(args.prefix)
    if isinstance(teacher, torch.nn.Module):
        teacher.to(DEVICE)

    x_train, y_train, x_test, y_test = load_data(args.prefix)
    x_train = x_train.to(DEVICE)

    N_total, D = x_train.shape
    N_base = min(args.n_targets, N_total)
    perm = torch.randperm(N_total, device=DEVICE)
    idxs = perm[:N_base]
    x_base = x_train[idxs]  # (N_base, D)

    print(f"Using N_base={N_base} base points out of {N_total}, D={D}")

    # Check if teacher has support S
    has_support = hasattr(teacher, "S")
    if has_support:
        S = list(teacher.S)
        k_active = len(S)
        print(f"Ground truth active set S (size {k_active}): {S}")
    else:
        print("Teacher has no attribute 'S'")

    # ---------------------------------------------------------
    # 2) Build Chebyshev t in [0.2, 1.0] for path augmentation
    #    (EXACT SAME as train_mps_sqexp_paths.py)
    # ---------------------------------------------------------
    N_T_NODES = args.max_degree + 1
    t_nodes = chebyshev_nodes_scaled(
        n_nodes=N_T_NODES,
        t_min=0.2,
        t_max=1.0,
        device=DEVICE,
        dtype=x_base.dtype,
    )
    print(f"Using N_T_NODES={N_T_NODES} Chebyshev nodes in t=[0.2, 1.0] for path augmentation")

    # ---------------------------------------------------------
    # 3) Construct path-augmented dataset
    #    (EXACT SAME as train_mps_sqexp_paths.py)
    #    for each x0, each t:
    #       - 1 h-path       : t * x0
    #       - D g_on_i paths : x_on_i(t)
    #       - D g_off_i paths: x_off_i(t)
    # ---------------------------------------------------------
    print("\n--- Constructing path-augmented dataset ---")
    total_samples = (1 + 2 * D) * N_base * N_T_NODES
    X_all = torch.zeros(total_samples, D, device=DEVICE, dtype=x_base.dtype)
    Y_all = torch.zeros(total_samples, device=DEVICE, dtype=x_base.dtype)

    teacher.eval()
    idx = 0

    print("Generating path-augmented dataset (h + g_on_i + g_off_i paths)...")
    with torch.no_grad():
        for b in range(N_base):
            x0 = x_base[b]  # (D,)

            for t in t_nodes:
                # Base scaled vector for this t
                base = t * x0  # (D,)

                # -------------------
                # h-path: x_h(t) = t * x0
                # -------------------
                x_h = base
                y_h = teacher(x_h.unsqueeze(0)).squeeze(0)
                if y_h.ndim > 0:
                    y_h = y_h.squeeze(-1)
                X_all[idx] = x_h
                Y_all[idx] = y_h
                idx += 1

                # -------------------
                # g_on_i paths (vectorized)
                #   x_on_i(t)[j] = x[j] if j == i else t * x[j]
                # -------------------
                x_on_batch = base.unsqueeze(0).expand(D, -1).clone()  # (D, D)
                x_on_batch[torch.arange(D), torch.arange(D)] = x0     # clamp i to original x_i
                y_on_batch = teacher(x_on_batch)
                if y_on_batch.ndim > 1:
                    y_on_batch = y_on_batch.squeeze(-1)

                X_all[idx:idx + D] = x_on_batch
                Y_all[idx:idx + D] = y_on_batch
                idx += D

                # -------------------
                # g_off_i paths (vectorized)
                #   x_off_i(t)[j] = 0 if j == i else t * x[j]
                # -------------------
                x_off_batch = base.unsqueeze(0).expand(D, -1).clone()  # (D, D)
                x_off_batch[torch.arange(D), torch.arange(D)] = 0.0    # clamp i to baseline 0
                y_off_batch = teacher(x_off_batch)
                if y_off_batch.ndim > 1:
                    y_off_batch = y_off_batch.squeeze(-1)

                X_all[idx:idx + D] = x_off_batch
                Y_all[idx:idx + D] = y_off_batch
                idx += D

    assert idx == total_samples, "Index mismatch when building path-augmented dataset."

    print(
        f"Path-augmented dataset size: {X_all.shape[0]} points "
        f"= (1 + 2*{D}) * {N_base} * {N_T_NODES}"
    )

    # ---------------------------------------------------------
    # 4) Train Decision Tree on path-augmented data
    # ---------------------------------------------------------
    print("\n--- Training Decision Tree ---")
    
    # Convert to numpy for sklearn
    X_all_np = X_all.cpu().numpy()
    Y_all_np = Y_all.cpu().numpy()
    
    dt_model = DecisionTreeRegressor(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
    )
    
    train_start = time.time()
    dt_model.fit(X_all_np, Y_all_np)
    train_time = time.time() - train_start
    
    print(f"Training completed in {train_time:.2f} seconds")

    # ---------------------------------------------------------
    # 5) Evaluate Decision Tree
    # ---------------------------------------------------------
    print("\n--- Evaluating Decision Tree ---")
    
    # Training set performance (on path-augmented data)
    Y_pred_np = dt_model.predict(X_all_np)
    train_mse = mean_squared_error(Y_all_np, Y_pred_np)
    train_r2 = sklearn_r2_score(Y_all_np, Y_pred_np)
    
    print(f"Training set (path-augmented): MSE={train_mse:.6f}, R²={train_r2:.6f}")
    
    # Optional: Evaluate on original test set (if you want)
    x_test_np = x_test.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    y_test_pred = dt_model.predict(x_test_np)
    test_mse = mean_squared_error(y_test_np, y_test_pred)
    test_r2 = sklearn_r2_score(y_test_np, y_test_pred)
    
    print(f"Original test set:              MSE={test_mse:.6f}, R²={test_r2:.6f}")

    # ---------------------------------------------------------
    # 6) Compute TreeSHAP on base points
    # ---------------------------------------------------------
    print("\n--- Computing TreeSHAP Shapley values ---")
    
    if not SHAP_AVAILABLE:
        raise ImportError("TreeSHAP requires the 'shap' library. Install with: pip install shap")
    
    # Create TreeExplainer
    explainer = shap.TreeExplainer(dt_model)
    
    # Compute TreeSHAP on base points
    x_base_np = x_base.cpu().numpy()
    t0_tree = time.time()
    shap_values = explainer.shap_values(x_base_np)
    t1_tree = time.time()
    treeshap_time = t1_tree - t0_tree
    
    # Convert to torch
    shap_values_torch = torch.from_numpy(shap_values).to(DEVICE).float()
    
    print(f"TreeSHAP computed in {treeshap_time*1000:.2f} ms total "
          f"({treeshap_time*1000/N_base:.2f} ms per point)")
    
    # ---------------------------------------------------------
    # 7) Compute support accuracy (if teacher has support S)
    # ---------------------------------------------------------
    if has_support:
        print("\n--- Computing support accuracy ---")
        S_set = set(S)
        support_acc_list = []
        
        for idx in range(N_base):
            phi = shap_values_torch[idx]
            top_idx = torch.topk(phi.abs(), k_active).indices.tolist()
            correct = len(set(top_idx) & S_set)
            acc = correct / k_active
            support_acc_list.append(acc)
            print(f"[Point {idx:02d}] support acc={acc:.2f}")
        
        support_acc_tensor = torch.tensor(support_acc_list, device=DEVICE)
        mean_acc = support_acc_tensor.mean().item()
        std_acc = support_acc_tensor.std(unbiased=True).item() \
            if len(support_acc_tensor) > 1 else 0.0
        print(f"\nTreeSHAP support accuracy: mean={mean_acc:.4f}, std={std_acc:.4f}")
    else:
        support_acc_list = None
        print("\nNo support set available; skipping support accuracy.")

    # ---------------------------------------------------------
    # 8) Save results
    # ---------------------------------------------------------
    print("\n--- Saving results ---")
    
    # Create output directory
    os.makedirs("dt_results", exist_ok=True)
    
    # Save TreeSHAP results
    results_path = f"dt_results/{args.prefix}_dt_paths_treeshap.pt"
    torch.save(
        {
            "x_base": x_base.detach().cpu(),           # (N_base, D)
            "shapley": shap_values_torch.detach().cpu(), # (N_base, D)
            "t_nodes": t_nodes.detach().cpu(),         # (N_T_NODES,)
            "max_degree": args.max_degree,
            "prefix": args.prefix,
            "has_support": has_support,
            "support": S if has_support else None,
            "support_accuracy": support_acc_list,
            "dt_params": {
                "max_depth": args.max_depth,
                "min_samples_split": args.min_samples_split,
                "min_samples_leaf": args.min_samples_leaf,
                "seed": args.seed,
            },
            "performance": {
                "train_mse": train_mse,
                "train_r2": train_r2,
                "test_mse": test_mse,
                "test_r2": test_r2,
            },
            "treeshap_time": treeshap_time,
            "dataset_info": {
                "n_base": N_base,
                "n_t_nodes": N_T_NODES,
                "total_samples": total_samples,
                "dimension": D,
            },
        },
        results_path,
    )
    print(f"Saved TreeSHAP results to {results_path}")
    
    # Save Decision Tree model
    import pickle
    dt_model_path = f"dt_results/{args.prefix}_dt_paths_model.pkl"
    with open(dt_model_path, "wb") as f:
        pickle.dump(dt_model, f)
    print(f"Saved Decision Tree model to {dt_model_path}")
    
    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
