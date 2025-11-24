#!/usr/bin/env python3
"""
eval_compare.py

Compare the original sparse polynomial teacher and the trained MPS
on the SAME Chebyshev test grid.

Run:
    python eval_compare.py
"""

import torch
from torchmps import MPS

from poly_teacher import load_teacher, load_data, DEVICE
from train_mps import r2_score


PREFIX = "poly"


def load_mps(prefix: str) -> MPS:
    state = torch.load(f"{prefix}_mps.pt", map_location=DEVICE)
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


def augment_with_one(x: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
    return torch.cat([ones, x], dim=1)


def main():
    teacher = load_teacher(PREFIX)
    x_train, y_train, x_test, y_test = load_data(PREFIX)
    mps = load_mps(PREFIX)

    x_test = x_test.to(DEVICE)
    y_test = y_test.to(DEVICE)

    # ----- Eval original polynomial teacher (sanity check) -----
    with torch.no_grad():
        y_teacher = teacher(x_test)
    r2_teacher = r2_score(y_test, y_teacher)
    print(f"Teacher vs ground truth (should be ~1.0, up to noise): R2 = {r2_teacher:.4f}")

    # ----- Eval MPS on same Chebyshev test set -----
    x_test_aug = augment_with_one(x_test)
    with torch.no_grad():
        y_mps = mps(x_test_aug).squeeze(-1)
    r2_mps = r2_score(y_test, y_mps)
    print(f"MPS vs ground truth on same Chebyshev grid: R2 = {r2_mps:.4f}")

    # ----- Define eval functions you can plug into TN-SHAP -----
    def eval_fn_teacher(x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D) on DEVICE, no leading 1.
        """
        x = x.to(DEVICE)
        with torch.no_grad():
            y = teacher(x)
        return y

    def eval_fn_mps(x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D) on DEVICE.
        We append a leading 1 to match MPS training.
        """
        x = x.to(DEVICE)
        x_aug = augment_with_one(x)
        with torch.no_grad():
            y = mps(x_aug).squeeze(-1)
        return y

    # Just to prove they run:
    with torch.no_grad():
        y_t_small = eval_fn_teacher(x_test[:5])
        y_m_small = eval_fn_mps(x_test[:5])
    print("Sample eval_fn_teacher outputs:", y_t_small[:3].cpu().numpy())
    print("Sample eval_fn_mps outputs:    ", y_m_small[:3].cpu().numpy())

    print("\nYou can now import eval_fn_teacher / eval_fn_mps logic into your TN-SHAP script.")


if __name__ == "__main__":
    main()
