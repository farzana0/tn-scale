#!/usr/bin/env python3
"""
poly_teacher.py

- Defines a sparse polynomial teacher f(x) on R^D.
- Uses Chebyshev-distributed samples in [-1, 1]^D to build train/test sets.
- Saves teacher parameters and datasets to disk so they can be reused later.

Run:
    python poly_teacher.py

This will create:
    poly_teacher.pt          # teacher coefficients
    poly_train.pt            # train set (x_train, y_train)
    poly_test.pt             # test set  (x_test,  y_test)
"""

import argparse
import torch
import torch.nn as nn
from typing import List, Tuple

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# Sparse polynomial definition
# -----------------------

def make_sparse_poly_coeffs(d: int, k_active: int = 5, seed: int = 0):
    """
    Simple sparse polynomial:
        S = {0, ..., k_active-1}
        f(x) = sum_i a_i x_i + sum_{i<j} b_ij x_i x_j
    """
    g = torch.Generator().manual_seed(seed)
    S = list(range(k_active))
    a = torch.randn(k_active, generator=g) * 1.0
    b = torch.randn(k_active, k_active, generator=g) * 0.5
    b = torch.triu(b, diagonal=1)  # only i<j part
    return S, a, b


def sparse_poly_f(x: torch.Tensor, S: List[int], a: torch.Tensor, b: torch.Tensor):
    """
    x: [B, D]  ->  f(x): [B]
    """
    xS = x[:, S]  # [B, k]
    B, k = xS.shape

    # Linear
    lin = (xS * a.unsqueeze(0)).sum(dim=1)

    # Quadratic
    quad = torch.zeros(B, device=x.device, dtype=x.dtype)
    for i in range(k):
        for j in range(i + 1, k):
            quad += b[i, j] * xS[:, i] * xS[:, j]

    return lin + quad


class SparsePolyTeacher(nn.Module):
    """
    Wraps the sparse polynomial into an nn.Module so we can use it
    like any other PyTorch model and plug it into eval_fn later.
    """
    def __init__(self, D: int, S: List[int], a: torch.Tensor, b: torch.Tensor):
        super().__init__()
        self.D = D
        self.S = S
        # Store coeffs as buffers so they move with .to(device)
        self.register_buffer("a", a.clone())
        self.register_buffer("b", b.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D) in [-1, 1]^D (no appended 1 here).
        Returns: (B,) tensor.
        """
        assert x.shape[1] == self.D, f"Expected D={self.D}, got {x.shape[1]}"
        return sparse_poly_f(x, self.S, self.a, self.b)


# -----------------------
# Chebyshev grid sampling
# -----------------------

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

    This is NOT the full tensor product grid (which would be gigantic),
    but uses Chebyshev nodes as a discrete support.
    """
    nodes_1d = chebyshev_nodes_1d(n_nodes)  # (n_nodes,)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(low=0, high=n_nodes, size=(n_samples, d), generator=g)
    x = nodes_1d[idx]  # (n_samples, d)
    return x


# -----------------------
# Data + teacher construction
# -----------------------

def build_teacher_and_data(
    D: int,
    N_train: int,
    N_test: int,
    k_active: int = 5,
    noise_std: float = 0.0,
    n_nodes: int = 64,
    seed_base: int = 0,
):
    """
    Create:
    - sparse polynomial teacher
    - Chebyshev-distributed training and test sets
    """
    # Teacher
    S, a, b = make_sparse_poly_coeffs(D, k_active=k_active, seed=seed_base)
    teacher = SparsePolyTeacher(D=D, S=S, a=a, b=b).to(DEVICE)

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
    teacher: SparsePolyTeacher,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    S: List[int],
):
    """
    Save teacher parameters and datasets to disk.
    """
    # Save teacher
    teacher_state = {
        "D": teacher.D,
        "S": S,
        "a": teacher.a.cpu(),
        "b": teacher.b.cpu(),
    }
    torch.save(teacher_state, f"{prefix}_teacher.pt")

    # Save datasets (on CPU for portability)
    torch.save({"x": x_train.cpu(), "y": y_train.cpu()}, f"{prefix}_train.pt")
    torch.save({"x": x_test.cpu(), "y": y_test.cpu()}, f"{prefix}_test.pt")
    print(f"Saved teacher and data with prefix '{prefix}_*.pt'")


def load_teacher(prefix: str) -> SparsePolyTeacher:
    """
    Reload the teacher from disk.
    """
    state = torch.load(f"{prefix}_teacher.pt", map_location=DEVICE)
    teacher = SparsePolyTeacher(
        D=state["D"],
        S=state["S"],
        a=state["a"],
        b=state["b"],
    ).to(DEVICE)
    return teacher


def load_data(prefix: str):
    train = torch.load(f"{prefix}_train.pt", map_location=DEVICE)
    test = torch.load(f"{prefix}_test.pt", map_location=DEVICE)
    return train["x"], train["y"], test["x"], test["y"]


# -----------------------
# CLI
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="poly",
                        help="Prefix for saved files: <prefix>_teacher.pt, _train.pt, _test.pt")
    parser.add_argument("--D", type=int, default=50)
    parser.add_argument("--n-train", type=int, default=10_000)
    parser.add_argument("--n-test", type=int, default=2_000)
    parser.add_argument("--k-active", type=int, default=5)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--n-nodes", type=int, default=64,
                        help="Number of Chebyshev nodes per dimension")
    parser.add_argument("--seed-base", type=int, default=0)
    args = parser.parse_args()

    print(f"Building teacher and data on device {DEVICE} ...")

    teacher, (x_train, y_train), (x_test, y_test), S = build_teacher_and_data(
        D=args.D,
        N_train=args.n_train,
        N_test=args.n_test,
        k_active=args.k_active,
        noise_std=args.noise_std,
        n_nodes=args.n_nodes,
        seed_base=args.seed_base,
    )

    print(f"True active set S: {S}")
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
    )


if __name__ == "__main__":
    main()
