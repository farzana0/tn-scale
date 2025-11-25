#!/usr/bin/env python3
"""
poly_teacher.py

- Defines a sparse degree-5 polynomial teacher f(x) on R^D.
- Uses Chebyshev-distributed samples in [-1, 1]^D to build train/test sets.
- Saves teacher parameters and datasets to disk so they can be reused later.

Run:
    python poly_teacher.py

This will create:
    <prefix>_teacher.pt          # teacher coefficients
    <prefix>_train.pt            # train set (x_train, y_train)
    <prefix>_test.pt             # test set  (x_test,  y_test)
"""

import argparse
import itertools
from typing import List, Tuple

import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# Sparse degree-5 polynomial definition
# -----------------------

def make_sparse_poly_coeffs(
    d: int,
    k_active: int = 5,
    max_degree: int = 5,
    seed: int = 0,
):
    """
    Build a sparse multilinear polynomial on R^d of maximum degree 5.

    Active feature set:
        S = {0, ..., k_active-1}

    Polynomial form (multilinear, degree <= 5):
        f(x) =
          sum_{i in S} a_i x_i
        + sum_{i<j in S} b_ij x_i x_j
        + sum_{i<j<k in S} c3_(i,j,k) x_i x_j x_k
        + sum_{i<j<k<l in S} c4_(i,j,k,l) x_i x_j x_k x_l
        + sum_{i<j<k<l<m in S} c5_(i,j,k,l,m) x_i x_j x_k x_l x_m

    For k_active = 5 this includes ALL monomials over S up to degree 5.
    For larger k_active, this still includes all monomials over S up to degree 5,
    which is combinatorially larger but still fine for typical k_active <= ~10.
    """
    assert max_degree == 5, "This helper is specialized for max_degree = 5."

    g = torch.Generator().manual_seed(seed)

    # Active subset of features
    S = list(range(k_active))

    # --- degree 1 (linear) ---
    a = torch.randn(k_active, generator=g) * 1.0  # (k_active,)

    # --- degree 2 (quadratic) ---
    b = torch.randn(k_active, k_active, generator=g) * 0.5
    b = torch.triu(b, diagonal=1)  # keep only i<j part

    # --- degrees 3,4,5: enumerate all combos on S and assign random coeffs ---
    def all_combos_and_coeffs(k: int, degree: int, scale: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return:
          idx_tensor: (n_terms, degree) long, each row a sorted tuple of indices in [0, k-1]
          coeffs:     (n_terms,) float tensor of coefficients
        """
        combos = list(itertools.combinations(range(k), degree))
        if len(combos) == 0:
            idx_tensor = torch.empty(0, degree, dtype=torch.long)
            coeffs = torch.empty(0, dtype=torch.float32)
            return idx_tensor, coeffs

        idx_tensor = torch.tensor(combos, dtype=torch.long)
        # Scale coefficients a bit smaller for higher degrees to keep outputs sane
        coeffs = (scale ** (degree - 1)) * torch.randn(len(combos), generator=g)
        return idx_tensor, coeffs

    idx3, c3 = all_combos_and_coeffs(k_active, 3, scale=0.5)
    idx4, c4 = all_combos_and_coeffs(k_active, 4, scale=0.5)
    idx5, c5 = all_combos_and_coeffs(k_active, 5, scale=0.5)

    return S, a, b, idx3, c3, idx4, c4, idx5, c5


class SparsePolyTeacher(nn.Module):
    """
    Sparse multilinear polynomial of maximum degree 5, wrapped as nn.Module.

    Attributes:
        D   : ambient dimension
        S   : list of active feature indices
        a   : (k_active,)      degree-1 coefficients
        b   : (k_active,k_active) upper-triangular degree-2 coefficients
        idx3: (n3,3)  indices for degree-3 monomials (over S positions)
        c3  : (n3,)   coefficients for degree-3 monomials
        idx4: (n4,4)  indices for degree-4 monomials
        c4  : (n4,)
        idx5: (n5,5)  indices for degree-5 monomials
        c5  : (n5,)
    """

    def __init__(
        self,
        D: int,
        S: List[int],
        a: torch.Tensor,
        b: torch.Tensor,
        idx3: torch.Tensor,
        c3: torch.Tensor,
        idx4: torch.Tensor,
        c4: torch.Tensor,
        idx5: torch.Tensor,
        c5: torch.Tensor,
    ):
        super().__init__()
        self.D = D
        self.S = S

        # Store all coefficients as buffers so they follow .to(device)
        self.register_buffer("a", a.clone())
        self.register_buffer("b", b.clone())

        # Higher-order indices & coeffs
        self.register_buffer("idx3", idx3.clone())  # (n3,3) or (0,3)
        self.register_buffer("c3", c3.clone())      # (n3,)  or (0,)
        self.register_buffer("idx4", idx4.clone())  # (n4,4)
        self.register_buffer("c4", c4.clone())
        self.register_buffer("idx5", idx5.clone())  # (n5,5)
        self.register_buffer("c5", c5.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D) in [-1, 1]^D (no appended 1 here).
        Returns: (B,) tensor.
        """
        assert x.shape[1] == self.D, f"Expected D={self.D}, got {x.shape[1]}"

        # Restrict to active coordinates S
        xS = x[:, self.S]  # (B, k_active)
        B, k = xS.shape

        # ----- degree 1 -----
        lin = (xS * self.a.unsqueeze(0)).sum(dim=1)  # (B,)

        # ----- degree 2 -----
        quad = torch.zeros(B, device=x.device, dtype=x.dtype)
        for i in range(k):
            for j in range(i + 1, k):
                quad = quad + self.b[i, j] * xS[:, i] * xS[:, j]

        out = lin + quad

        # ----- degree 3 -----
        if self.idx3.numel() > 0:
            # x3: (B, n3, 3)
            x3 = xS[:, self.idx3]          # advanced indexing
            prod3 = x3.prod(dim=2)         # (B, n3)
            out = out + (prod3 * self.c3.unsqueeze(0)).sum(dim=1)

        # ----- degree 4 -----
        if self.idx4.numel() > 0:
            x4 = xS[:, self.idx4]          # (B, n4, 4)
            prod4 = x4.prod(dim=2)         # (B, n4)
            out = out + (prod4 * self.c4.unsqueeze(0)).sum(dim=1)

        # ----- degree 5 -----
        if self.idx5.numel() > 0:
            x5 = xS[:, self.idx5]          # (B, n5, 5)
            prod5 = x5.prod(dim=2)         # (B, n5)
            out = out + (prod5 * self.c5.unsqueeze(0)).sum(dim=1)

        return out


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
    - sparse degree-5 polynomial teacher
    - Chebyshev-distributed training and test sets
    """
    # Teacher coefficients
    S, a, b, idx3, c3, idx4, c4, idx5, c5 = make_sparse_poly_coeffs(
        d=D,
        k_active=k_active,
        max_degree=5,
        seed=seed_base,
    )
    teacher = SparsePolyTeacher(
        D=D,
        S=S,
        a=a,
        b=b,
        idx3=idx3,
        c3=c3,
        idx4=idx4,
        c4=c4,
        idx5=idx5,
        c5=c5,
    ).to(DEVICE)

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
        "idx3": teacher.idx3.cpu(),
        "c3": teacher.c3.cpu(),
        "idx4": teacher.idx4.cpu(),
        "c4": teacher.c4.cpu(),
        "idx5": teacher.idx5.cpu(),
        "c5": teacher.c5.cpu(),
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

    # Backwards safety: if some keys are missing (old checkpoints), create empty ones.
    def get_or_empty(name: str, shape_suffix: Tuple[int, ...]) -> torch.Tensor:
        if name in state:
            return state[name]
        # Shape_suffix is e.g. (3,) for idx3 columns etc.; we only use it if missing.
        return torch.empty(0, *shape_suffix, dtype=torch.long if name.startswith("idx") else torch.float32)

    a = state["a"]
    b = state["b"]
    idx3 = get_or_empty("idx3", (3,))
    c3 = state.get("c3", torch.empty(0, dtype=torch.float32))
    idx4 = get_or_empty("idx4", (4,))
    c4 = state.get("c4", torch.empty(0, dtype=torch.float32))
    idx5 = get_or_empty("idx5", (5,))
    c5 = state.get("c5", torch.empty(0, dtype=torch.float32))

    teacher = SparsePolyTeacher(
        D=state["D"],
        S=state["S"],
        a=a,
        b=b,
        idx3=idx3,
        c3=c3,
        idx4=idx4,
        c4=c4,
        idx5=idx5,
        c5=c5,
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

    print(f"Building degree-5 teacher and data on device {DEVICE} ...")

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
