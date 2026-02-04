"""
Sampling distributions for flow-matching experiments.

- simple_sample(C, N, dims): callable that samples (x, c) from C classes of blobs in R^N.
- pair_sample(f, N): callable that samples z ~ N(0,I) and returns (x, c) = f(z).
"""

import torch
import math


def simple_sample(C, N, dims=None):
    """
    Create a distribution of C classes (c = 0, ..., C-1) of disparate amorphous blobs in R^N,
    Class c lives on a dims[c]-dimensional Gaussian subspace.

    Example: dims = [2,2,2,2,2,1] means 6 classes, 5 live on 2-dim, 1 live on 1-dim.

    For class c, we construct a mean μ_c in R^N and an orthonormal basis
    B_c in R^{N x d_c}, where d_c = dims[c]. Samples are drawn as
        z ~ N(0, I_{d_c}),  x = μ_c + B_c z,
    so the intrinsic dimension of class c is exactly d_c (embedded in R^N).

    Args:
        C: number of classes (labels 0, ..., C-1).
        N: ambient dimension (space is R^N).
        dims: list of length C; intrinsic dimension d_c for each class.
              Must satisfy 1 <= d_c <= N. Default [N, N, ..., N].

    Returns:
        A callable that, when called with K, returns (x, c) where
        x: (K, N) tensor, c: (K,) long tensor in {0, ..., C-1}.
        You can optionally pass a desired class at call time.
    """
    if dims is None:
        dims = [N] * C
    if len(dims) != C:
        raise ValueError(f"dims must have length C={C}, got {len(dims)}")
    if any(d <= 0 or d > N for d in dims):
        raise ValueError(f"each d_c in dims must satisfy 1 <= d_c <= N={N}, got {dims}")

    # Per-class mean in R^N (random, fixed at init), pushed far apart on a sphere
    # to reduce overlap between blobs.
    radius = 5.0 * math.sqrt(N)
    raw_means = torch.randn(C, N)
    norms = raw_means.norm(dim=1, keepdim=True).clamp_min(1e-6)
    means = radius * raw_means / norms

    # Per-class orthonormal bases B_c (N x d_c)
    bases = []
    for d in dims:
        # Random Gaussian matrix and QR to get orthonormal columns
        A = torch.randn(N, d)
        # torch.linalg.qr returns Q (N x N) and R; take first d columns
        Q, _ = torch.linalg.qr(A, mode="reduced")
        bases.append(Q[:, :d])  # (N, d)

    class _SimpleSampler:
        def __call__(self, K, device=None, class_idx=None):
            """
            Sample K points.

            Args:
                K: number of points.
                device: torch device.
                class_idx:
                    - None or int == C: each point gets a random class in {0, ..., C-1}.
                    - int in {0, ..., C-1}: all K points from this class.
                    - 1D tensor / list of length K: per-point class indices in {0, ..., C-1}.
            """
            dev = device or means.device

            # Case 1: random class per point
            if class_idx is None or (isinstance(class_idx, int) and class_idx == C):
                c = torch.randint(0, C, (K,), dtype=torch.long, device=dev)

            else:
                # User-specified class(es)
                if isinstance(class_idx, int):
                    # All points from the same specified class
                    if not (0 <= class_idx < C):
                        raise ValueError(
                            f"class_idx must be in [0, {C-1}] or equal to C for random, got {class_idx}"
                        )
                    c = torch.full((K,), class_idx, dtype=torch.long, device=dev)
                else:
                    # Per-point class indices
                    c = torch.as_tensor(class_idx, dtype=torch.long, device=dev)
                    if c.numel() == 1:
                        c = c.expand(K)
                    if c.shape[0] != K:
                        raise ValueError(f"class_idx must have length {K}, got {c.shape[0]}")
                    if not torch.all((0 <= c) & (c < C)):
                        raise ValueError(f"class indices must be in [0, {C-1}]")

            # Allocate output
            x = torch.empty(K, N, dtype=means.dtype, device=dev)

            # For each class, sample in its d_c-dimensional subspace and embed
            means_dev = means.to(dev)
            bases_dev = [B.to(dev) for B in bases]

            for cls in range(C):
                mask = (c == cls)
                n_k = int(mask.sum().item())
                if n_k == 0:
                    continue
                d_c = dims[cls]
                z_k = torch.randn(n_k, d_c, dtype=means.dtype, device=dev)  # (n_k, d_c)
                B_c = bases_dev[cls]  # (N, d_c)
                # x_k = μ_c + z_k @ B_c^T
                x_k = z_k @ B_c.T + means_dev[cls].unsqueeze(0)  # (n_k, N)
                x[mask] = x_k

            return x, c

    return _SimpleSampler()


def pair_sample(f, N):
    """
    Create a distribution that, when called with K, samples z ~ N(0, I_N)
    and returns (x, c) = f(z).

    Args:
        f: callable such that f(z) returns (x0, c) where z is (K, N),
           x0 is (K, N), and c is (K,) class indices.
        N: dimension of z.

    Returns:
        A callable that, when called with K, draws z ~ N(0, I_N) and
        returns (x0, c) = f(z).
    """
    class _PairSampler:
        def __init__(self, func, dim):
            self.f = func
            self.N = dim

        def __call__(self, K, device=None):
            dev = device or torch.device("cpu")
            z = torch.randn(K, self.N, device=dev)
            x0, c = self.f(z)
            return x0, c, z  # z is ε for flow-matching loss

    return _PairSampler(f, N)
