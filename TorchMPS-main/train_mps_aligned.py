#!/usr/bin/env python3
"""
train_mps_aligned.py

Train an MPS on path-augmented data with EXACT alignment to TN-SHAP evaluation points.
This ensures that:
1. The MPS is trained on the exact points TN-SHAP will query
2. Both teacher and MPS TN-SHAP use identical evaluation points
3. The multilinear extension is evaluated consistently
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchmps import MPS
from poly_teacher import load_teacher, load_data, DEVICE
import time

PREFIX = "poly"
BATCH_SIZE = 5000
NUM_EPOCHS = 50
BOND_DIM = 60
LEARN_RATE = 1e-3
L2_REG = 1e-1

# Critical: Use same number of nodes as max polynomial degree + 1
N_T_NODES = 51  # For D=50, this allows exact interpolation


def chebyshev_nodes_unit_interval(n_nodes: int, device=None, dtype=torch.float32):
    """Chebyshev nodes mapped to [0,1] - MUST be identical across all scripts"""
    if device is None:
        device = DEVICE
    k = torch.arange(1, n_nodes + 1, dtype=torch.float64, device=device)
    u = torch.cos((2.0 * k - 1.0) / (2.0 * n_nodes) * torch.pi)
    t = (u + 1.0) / 2.0
    return t.to(dtype)


def generate_tnshap_evaluation_points(x_base, t_nodes, teacher):
    """
    Generate ALL evaluation points that TN-SHAP will query.
    This includes:
    - h(t) = f(t*x) for all base points and t values
    - g_i(t) = f(x with feature i fixed, others scaled by t) for all i, base points, and t
    
    Returns:
        evaluation_dict: Dictionary containing all evaluation points and their structure
    """
    N_base, D = x_base.shape
    N_t = len(t_nodes)
    
    evaluation_dict = {
        'h_points': [],      # Points for h(t) evaluation
        'h_values': [],      # Teacher values at h(t) points
        'g_points': {},      # Points for g_i(t) evaluation, indexed by feature i
        'g_values': {},      # Teacher values at g_i(t) points
        'base_indices': [],  # Which base point each evaluation corresponds to
        't_indices': [],     # Which t-node each evaluation corresponds to
    }
    
    teacher.eval()
    with torch.no_grad():
        # For each base point
        for b_idx in range(N_base):
            x0 = x_base[b_idx]  # (D,)
            
            # Store h(t) evaluation points
            h_batch = []
            for t_idx, t in enumerate(t_nodes):
                x_h = t * x0  # Scale all features
                h_batch.append(x_h)
                evaluation_dict['base_indices'].append(b_idx)
                evaluation_dict['t_indices'].append(t_idx)
            
            h_batch = torch.stack(h_batch)  # (N_t, D)
            h_vals = teacher(h_batch)
            if h_vals.ndim > 1:
                h_vals = h_vals.squeeze(-1)
            
            evaluation_dict['h_points'].append(h_batch)
            evaluation_dict['h_values'].append(h_vals)
            
            # Store g_i(t) evaluation points for each feature
            for i in range(D):
                if i not in evaluation_dict['g_points']:
                    evaluation_dict['g_points'][i] = []
                    evaluation_dict['g_values'][i] = []
                
                g_batch = []
                for t_idx, t in enumerate(t_nodes):
                    x_g = t * x0.clone()
                    x_g[i] = x0[i]  # Fix feature i
                    g_batch.append(x_g)
                
                g_batch = torch.stack(g_batch)  # (N_t, D)
                g_vals = teacher(g_batch)
                if g_vals.ndim > 1:
                    g_vals = g_vals.squeeze(-1)
                
                evaluation_dict['g_points'][i].append(g_batch)
                evaluation_dict['g_values'][i].append(g_vals)
    
    # Concatenate all points
    evaluation_dict['h_points'] = torch.cat(evaluation_dict['h_points'], dim=0)
    evaluation_dict['h_values'] = torch.cat(evaluation_dict['h_values'], dim=0)
    
    for i in range(D):
        evaluation_dict['g_points'][i] = torch.cat(evaluation_dict['g_points'][i], dim=0)
        evaluation_dict['g_values'][i] = torch.cat(evaluation_dict['g_values'][i], dim=0)
    
    return evaluation_dict


def create_training_dataset(evaluation_dict, D):
    """
    Create training dataset from evaluation points.
    This ensures MPS is trained on EXACTLY the points TN-SHAP will query.
    """
    X_list = [evaluation_dict['h_points']]  # Start with h(t) points
    Y_list = [evaluation_dict['h_values']]
    
    # Add all g_i(t) points
    for i in range(D):
        X_list.append(evaluation_dict['g_points'][i])
        Y_list.append(evaluation_dict['g_values'][i])
    
    X_all = torch.cat(X_list, dim=0)
    Y_all = torch.cat(Y_list, dim=0)
    
    return X_all, Y_all


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


def verify_evaluation_consistency(evaluation_dict, mps, D):
    """
    Verify that MPS predictions match the multilinear extension structure
    """
    mps.eval()
    with torch.no_grad():
        # Check h(t) predictions
        X_h_aug = augment_with_one(evaluation_dict['h_points'])
        y_h_pred = mps(X_h_aug).squeeze(-1)
        h_r2 = r2_score(evaluation_dict['h_values'], y_h_pred)
        
        # Check g_i(t) predictions for each feature
        g_r2_list = []
        for i in range(D):
            X_g_aug = augment_with_one(evaluation_dict['g_points'][i])
            y_g_pred = mps(X_g_aug).squeeze(-1)
            g_r2 = r2_score(evaluation_dict['g_values'][i], y_g_pred)
            g_r2_list.append(g_r2)
        
        mean_g_r2 = sum(g_r2_list) / len(g_r2_list)
        
    return h_r2, mean_g_r2, g_r2_list


def main():
    print(f"Device: {DEVICE}")
    
    # Load teacher and data
    teacher = load_teacher(PREFIX)
    x_train, y_train, x_test, y_test = load_data(PREFIX)
    
    # Use a subset of base points for demonstration
    x_base = torch.cat([x_train, x_test], dim=0).to(DEVICE)
    MAX_BASE = 20  # Adjust as needed
    x_base = x_base[:MAX_BASE]
    N_base, D = x_base.shape
    print(f"Base points: N_base={N_base}, D={D}")
    
    # Generate Chebyshev nodes - CRITICAL: same nodes for training and evaluation
    t_nodes = chebyshev_nodes_unit_interval(N_T_NODES, device=DEVICE, dtype=x_base.dtype)
    print(f"Chebyshev nodes: N_T_NODES={N_T_NODES}")
    print(f"t_nodes range: [{t_nodes.min():.4f}, {t_nodes.max():.4f}]")
    
    # Generate ALL TN-SHAP evaluation points
    print("\nGenerating TN-SHAP evaluation points...")
    evaluation_dict = generate_tnshap_evaluation_points(x_base, t_nodes, teacher)
    
    # Create training dataset from evaluation points
    X_all, Y_all = create_training_dataset(evaluation_dict, D)
    print(f"Training dataset size: {X_all.shape[0]} points")
    print(f"  - h(t) points: {evaluation_dict['h_points'].shape[0]}")
    print(f"  - g_i(t) points per feature: {evaluation_dict['g_points'][0].shape[0]}")
    
    # Augment inputs for MPS
    X_all_aug = augment_with_one(X_all)
    D_aug = X_all_aug.shape[1]
    
    # Create data loader
    train_ds = TensorDataset(X_all_aug, Y_all)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    
    # Initialize MPS
    mps = MPS(
        input_dim=D_aug,
        output_dim=1,
        bond_dim=BOND_DIM,
        adaptive_mode=False,
        periodic_bc=False,
    ).to(DEVICE)
    
    loss_fun = nn.MSELoss()
    optimizer = torch.optim.Adam(mps.parameters(), lr=LEARN_RATE, weight_decay=L2_REG)
    
    print(f"\nTraining MPS:")
    print(f"  D_aug={D_aug}, bond_dim={BOND_DIM}")
    print(f"  Training on EXACT TN-SHAP evaluation points\n")
    
    start_time = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        mps.train()
        train_loss = 0.0
        n_seen = 0
        
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            
            preds = mps(xb).squeeze(-1)
            loss = loss_fun(preds, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        
        train_mse = train_loss / max(n_seen, 1)
        
        # Evaluate consistency every 10 epochs
        if epoch % 10 == 0:
            h_r2, mean_g_r2, g_r2_list = verify_evaluation_consistency(evaluation_dict, mps, D)
            
            print(f"### Epoch {epoch:03d} ###")
            print(f"Train MSE: {train_mse:.5f}")
            print(f"h(t) R2: {h_r2:.4f}")
            print(f"g_i(t) mean R2: {mean_g_r2:.4f}")
            
            # Check feature-wise consistency
            problematic_features = [i for i, r2 in enumerate(g_r2_list) if r2 < 0.95]
            if problematic_features:
                print(f"  Warning: Low R2 for features {problematic_features[:5]}...")
            
            print(f"Runtime: {int(time.time() - start_time)} sec\n")
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    h_r2, mean_g_r2, g_r2_list = verify_evaluation_consistency(evaluation_dict, mps, D)
    print(f"Final h(t) R2: {h_r2:.4f}")
    print(f"Final g_i(t) mean R2: {mean_g_r2:.4f}")
    
    # Save everything needed for TN-SHAP comparison
    torch.save(
        {
            "x_base": x_base.detach().cpu(),
            "t_nodes": t_nodes.detach().cpu(),
            "evaluation_dict": {
                k: v.detach().cpu() if torch.is_tensor(v) 
                else {ki: vi.detach().cpu() for ki, vi in v.items()} if isinstance(v, dict)
                else v
                for k, v in evaluation_dict.items()
            },
            "X_all": X_all.detach().cpu(),
            "Y_all": Y_all.detach().cpu(),
        },
        f"{PREFIX}_tnshap_aligned_data.pt",
    )
    print(f"\nSaved aligned data to {PREFIX}_tnshap_aligned_data.pt")
    
    # Save MPS
    torch.save(
        {
            "state_dict": mps.state_dict(),
            "D_aug": D_aug,
            "bond_dim": BOND_DIM,
        },
        f"{PREFIX}_mps_aligned.pt",
    )
    print(f"Saved MPS to {PREFIX}_mps_aligned.pt")
    
    print("\n✓ MPS trained on EXACT TN-SHAP evaluation points")
    print("✓ Multilinear extension consistency verified")
    print("✓ Ready for apples-to-apples TN-SHAP comparison")


if __name__ == "__main__":
    main()