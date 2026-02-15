"""
Sampling distributions for flow-matching experiments.

- simple_sample(C, N, dims): callable that samples (x, c) from C classes of blobs in R^N.
- pair_sample(f, N): callable that samples z ~ N(0,I) and returns (x, c) = f(z).
"""

import torch
import math


def simple_Gaussian_sample(C, N, dims=None):
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







def simple_Sphere_sample(C, N, dims=None, noise_std=0.0):
    """
    Create a distribution of C classes of spheres in R^N.
    Class c is a dims[c]-dimensional sphere S^{d_c} (uniformly sampled),
    embedded in R^N via a random orthonormal frame, with random center and radius.

    A d-dimensional sphere S^d needs (d+1) ambient coordinates, so we require
    dims[c] <= N-1  (i.e. d+1 <= N).

    Concretely, for class c with d_c = dims[c]:
        1. Sample u ~ N(0, I_{d_c+1}), normalise to  û = u / ||u||  (uniform on S^{d_c}).
        2. Scale: û *= r_c  (random radius, fixed per class).
        3. Embed: x = μ_c + B_c @ û,  where B_c is (N, d_c+1) orthonormal.
        4. (Optional) Perturb: x += noise_std * N(0, I_N) to give the sphere thickness.

    Args:
        C: number of classes (labels 0, ..., C-1).
        N: ambient dimension (space is R^N).
        dims: list of length C; sphere dimension d_c for each class.
              Must satisfy 1 <= d_c <= N-1.  Default [N-1, N-1, ..., N-1].
        noise_std: standard deviation of ambient Gaussian noise added to each
                   sample after projection onto the sphere. Default 0.0 (no noise).
                   A small positive value (e.g. 0.05-0.1) gives the manifold
                   "thickness", which helps flow-matching learn singular distributions.

    Returns:
        A callable with the same interface as simple_Gaussian_sample:
        sampler(K, device=, class_idx=) -> (x, c).
    """
    if dims is None:
        dims = [N - 1] * C
    if len(dims) != C:
        raise ValueError(f"dims must have length C={C}, got {len(dims)}")
    if any(d <= 0 or d > N - 1 for d in dims):
        raise ValueError(
            f"each d_c in dims must satisfy 1 <= d_c <= N-1={N-1} "
            f"(S^d needs d+1 <= N), got {dims}"
        )

    # Per-class random center in R^N, pushed far apart
    center_radius = 5.0 * math.sqrt(N)
    raw_centers = torch.randn(C, N)
    norms = raw_centers.norm(dim=1, keepdim=True).clamp_min(1e-6)
    centers = center_radius * raw_centers / norms

    # Per-class random sphere radius (uniform in [0.5, 2.0])
    radii = torch.rand(C) * 1.5 + 0.5  # (C,)

    # Per-class orthonormal basis B_c of shape (N, d_c+1)
    # The (d_c+1) columns span the subspace that hosts S^{d_c}
    bases = []
    for d in dims:
        A = torch.randn(N, d + 1)
        Q, _ = torch.linalg.qr(A, mode="reduced")
        bases.append(Q[:, : d + 1])  # (N, d+1)

    class _SphereSampler:
        def __call__(self, K, device=None, class_idx=None):
            """
            Sample K points.

            Args:
                K: number of points.
                device: torch device.
                class_idx:
                    - None or int == C: each point gets a random class.
                    - int in {0, ..., C-1}: all K points from this class.
                    - 1D tensor / list of length K: per-point class indices.
            """
            dev = device or centers.device

            # --- resolve class labels (identical logic to Gaussian sampler) ---
            if class_idx is None or (isinstance(class_idx, int) and class_idx == C):
                c = torch.randint(0, C, (K,), dtype=torch.long, device=dev)
            else:
                if isinstance(class_idx, int):
                    if not (0 <= class_idx < C):
                        raise ValueError(
                            f"class_idx must be in [0, {C-1}] or equal to C for random, got {class_idx}"
                        )
                    c = torch.full((K,), class_idx, dtype=torch.long, device=dev)
                else:
                    c = torch.as_tensor(class_idx, dtype=torch.long, device=dev)
                    if c.numel() == 1:
                        c = c.expand(K)
                    if c.shape[0] != K:
                        raise ValueError(f"class_idx must have length {K}, got {c.shape[0]}")
                    if not torch.all((0 <= c) & (c < C)):
                        raise ValueError(f"class indices must be in [0, {C-1}]")

            # --- allocate output ---
            x = torch.empty(K, N, dtype=centers.dtype, device=dev)

            centers_dev = centers.to(dev)
            radii_dev = radii.to(dev)
            bases_dev = [B.to(dev) for B in bases]

            for cls in range(C):
                mask = (c == cls)
                n_k = int(mask.sum().item())
                if n_k == 0:
                    continue
                d_c = dims[cls]
                # Sample uniform on S^{d_c}: Gaussian in R^{d_c+1}, then normalise
                u = torch.randn(n_k, d_c + 1, dtype=centers.dtype, device=dev)
                u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-8)
                u = u * radii_dev[cls]         # scale by sphere radius
                B_c = bases_dev[cls]           # (N, d_c+1)
                # Embed: x_k = center_c + u @ B_c^T
                x_k = u @ B_c.T + centers_dev[cls].unsqueeze(0)  # (n_k, N)
                # Optionally perturb with ambient Gaussian noise for manifold thickness
                if noise_std > 0:
                    x_k = x_k + noise_std * torch.randn_like(x_k)
                x[mask] = x_k

            return x, c

    return _SphereSampler()


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
