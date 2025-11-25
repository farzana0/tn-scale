#!/usr/bin/env python3
"""
poly_teacher.py

Unified teacher generator for three regression tasks in R^D:

  - poly5 : degree-5 polynomial in the first |S| = D/3 features
  - poly10: degree-10 polynomial in the first |S| = D/3 features
  - sqexp : squared-exponential y = exp(sum_{i in S} x_i^2)

All tasks:
  - Active set S = {0, ..., k_active-1}, where k_active = D // 3
  - Remaining features are redundant.

This file keeps the same public API as before:

    DEVICE
    load_teacher(prefix)
    load_data(prefix)

and the same file naming pattern:

    <prefix>_teacher.pt
    <prefix>_train.pt
    <prefix>_test.pt
"""

import argparse
from typing import List, Tuple

import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def chebyshev_nodes_1d(n_nodes: int) -> torch.Tensor:
    """
    Chebyshev nodes of the first kind on [-1, 1]:
        x_k = cos((2k-1)/(2n) * pi), k = 1..n
    """
    k = torch.arange(1, n_nodes + 1, dtype=torch.float64)
    nodes = torch.cos((2.0 * k - 1.0) / (2.0 * n_nodes) * torch.pi)
    return nodes.to(torch.float32)


def sample_chebyshev_points(
    n_samples: int,
    d: int,
    n_nodes: int = 64,
    seed: int = 0,
) -> torch.Tensor:
    """
    Sample n_samples points in [-1,1]^d by choosing coordinates
    from a 1D Chebyshev grid of size n_nodes.

    This is NOT the full tensor product grid, just a convenient discrete support.
    """
    nodes_1d = chebyshev_nodes_1d(n_nodes)  # (n_nodes,)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(low=0, high=n_nodes, size=(n_samples, d), generator=g)
    x = nodes_1d[idx]  # (n_samples, d)
    return x


# ---------------------------------------------------------
# Teacher model
# ---------------------------------------------------------

class GeneralTeacher(nn.Module):
    """
    Teacher for 3 tasks:

      task='poly5' or 'poly10':
          y = sum_{i in S} sum_{p=1}^deg c_{i,p} x_i^p

      task='sqexp':
          y = exp( sum_{i in S} x_i^2 )

    where S = {0, ..., k_active-1}, k_active = D // 3.
    """
    def __init__(
        self,
        D: int,
        S: List[int],
        task: str,
        coeffs: torch.Tensor | None = None,
    ):
        super().__init__()
        self.D = D
        self.S = S
        self.task = task

        if task in ("poly5", "poly10"):
            assert coeffs is not None, "Polynomial tasks require coeffs."
            # coeffs shape: (k_active, deg)
            self.register_buffer("coeffs", coeffs.clone())
        else:
            self.register_buffer("coeffs", torch.empty(0))  # unused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D) in [-1, 1]^D.
        Returns: (B,) tensor.
        """
        assert x.shape[1] == self.D, f"Expected D={self.D}, got {x.shape[1]}"
        z = x[:, self.S]  # (B, k_active)
        B, k = z.shape

        if self.task in ("poly5", "poly10"):
            # coeffs: (k, deg)
            deg = self.coeffs.shape[1]

            # z_powers[b, i, p] = z[b, i]^(p+1)  for p=0..deg-1
            powers = [z ** (p + 1) for p in range(deg)]
            z_powers = torch.stack(powers, dim=2)  # (B, k, deg)

            # y = sum_{i,p} coeffs[i,p] * z_powers[b,i,p]
            y = torch.sum(z_powers * self.coeffs.unsqueeze(0), dim=(1, 2))  # (B,)
            return y

        elif self.task == "sqexp":
            # y = exp( sum_{i in S} x_i^2 )
            y = torch.exp(torch.sum(z ** 2, dim=1))  # (B,)
            return y

        else:
            raise ValueError(f"Unknown task '{self.task}'")


# ---------------------------------------------------------
# Data + teacher construction
# ---------------------------------------------------------

def make_teacher(
    D: int,
    task: str,
    seed: int = 0,
) -> GeneralTeacher:
    """
    Build a GeneralTeacher with active set S = {0, ..., k_active-1}.
    """
    assert task in ("poly5", "poly10", "sqexp")
    k_active = D // 3
    S = list(range(k_active))

    if task in ("poly5", "poly10"):
        deg = 5 if task == "poly5" else 10
        g = torch.Generator().manual_seed(seed)
        # coeffs[i,p] is coefficient for x_i^(p+1)
        coeffs = torch.randn(k_active, deg, generator=g) * 0.5
        teacher = GeneralTeacher(D=D, S=S, task=task, coeffs=coeffs)
    else:
        teacher = GeneralTeacher(D=D, S=S, task=task, coeffs=None)

    return teacher.to(DEVICE), S


def build_teacher_and_data(
    D: int,
    N_train: int,
    N_test: int,
    task: str,
    noise_std: float = 0.0,
    n_nodes: int = 64,
    seed_base: int = 0,
):
    """
    Create:
    - teacher (GeneralTeacher)
    - Chebyshev-distributed training and test sets
    """
    teacher, S = make_teacher(D, task=task, seed=seed_base)

    # Train set
    x_train = sample_chebyshev_points(
        n_samples=N_train,
        d=D,
        n_nodes=n_nodes,
        seed=seed_base + 1,
    ).to(DEVICE)
    with torch.no_grad():
        y_train = teacher(x_train)
        if noise_std > 0.0:
            y_train = y_train + noise_std * torch.randn_like(y_train)

    # Test set
    x_test = sample_chebyshev_points(
        n_samples=N_test,
        d=D,
        n_nodes=n_nodes,
        seed=seed_base + 2,
    ).to(DEVICE)
    with torch.no_grad():
        y_test = teacher(x_test)
        if noise_std > 0.0:
            y_test = y_test + noise_std * torch.randn_like(y_test)

    return teacher, (x_train, y_train), (x_test, y_test), S


def save_teacher_and_data(
    prefix: str,
    teacher: GeneralTeacher,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    S: List[int],
    task: str,
):
    """
    Save teacher parameters and datasets to disk.
    """
    # Save teacher
    teacher_state = {
        "D": teacher.D,
        "S": S,
        "task": task,
        "coeffs": teacher.coeffs.cpu() if teacher.task in ("poly5", "poly10") else None,
    }
    torch.save(teacher_state, f"{prefix}_teacher.pt")

    # Save datasets (on CPU for portability)
    torch.save({"x": x_train.cpu(), "y": y_train.cpu()}, f"{prefix}_train.pt")
    torch.save({"x": x_test.cpu(), "y": y_test.cpu()}, f"{prefix}_test.pt")
    print(f"Saved teacher and data with prefix '{prefix}_*.pt'")


def load_teacher(prefix: str) -> GeneralTeacher:
    """
    Reload the teacher from disk.
    """
    state = torch.load(f"{prefix}_teacher.pt", map_location=DEVICE)
    D = state["D"]
    S = state["S"]
    task = state.get("task", "poly5")  # default if not present in old files

    coeffs = state.get("coeffs", None)
    teacher = GeneralTeacher(
        D=D,
        S=S,
        task=task,
        coeffs=coeffs,
    ).to(DEVICE)
    return teacher


def load_data(prefix: str):
    train = torch.load(f"{prefix}_train.pt", map_location=DEVICE)
    test = torch.load(f"{prefix}_test.pt", map_location=DEVICE)
    return train["x"], train["y"], test["x"], test["y"]


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="poly",
                        help="Prefix for saved files: <prefix>_teacher.pt, _train.pt, _test.pt")
    parser.add_argument("--task", type=str, default="poly5",
                        choices=["poly5", "poly10", "sqexp"],
                        help="Teacher type")
    parser.add_argument("--D", type=int, default=50)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--n-nodes", type=int, default=64,
                        help="Number of Chebyshev nodes per dimension")
    parser.add_argument("--seed-base", type=int, default=0)
    args = parser.parse_args()

    print(f"Building teacher '{args.task}' and data on device {DEVICE} ...")

    teacher, (x_train, y_train), (x_test, y_test), S = build_teacher_and_data(
        D=args.D,
        N_train=args.n_train,
        N_test=args.n_test,
        task=args.task,
        noise_std=args.noise_std,
        n_nodes=args.n_nodes,
        seed_base=args.seed_base,
    )

    print(f"Task: {args.task}")
    print(f"True active set S (size {len(S)}): {S}")
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test  shape: {x_test.shape},  y_test  shape: {y_test.shape}")

    save_teacher_and_data(
        prefix=args.prefix,
        teacher=teacher,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        S=S,
        task=args.task,
    )


if __name__ == "__main__":
    main()
