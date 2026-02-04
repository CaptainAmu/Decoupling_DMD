

import torch

"""
Inference: integrate ODE backward from noise (t=1) to data (t=0) using
the trained velocity field v_θ, with T steps and step size 1/T.
"""


def ode_backward(v_model, T, K, c, device=None, guidance: float = 1.0):
    """
    Run ODE backward from t=1 (noise) to t≈0 (data) with T steps, step size 1/T.

    dx/dt = v_θ(x, t, c), integrate backward: x_{t - 1/T} = x_t - v_θ(x_t, t, c) * (1/T).

    Args:
        v_model: trained velocity model with forward(x_t, t, c) -> v.
        T: number of steps.
        K: number of samples (batch size).
        c: (K,) long class indices in {0, ..., num_classes} (num_classes = ∅).
        device: torch device; if None, use model's device.
        guidance: guidance scale g. When g != 1, we use
                  v = v(x,t,∅) + g * (v(x,t,c) - v(x,t,∅))
                  instead of v(x,t,c) as the velocity.

    Returns:
        x: (K, N) approximate samples at t≈0.
    """
    device = device or next(v_model.parameters()).device
    v_model.eval()
    N = v_model.dim
    dt = 1.0 / T

    with torch.no_grad():
        x = torch.randn(K, N, device=device)
        t = torch.full((K,), 1.0, device=device)

        for _ in range(T):
            if guidance == 1.0:
                v = v_model(x, t, c)
            else:
                # Class-conditional guidance:
                # v_guided = v(x,t,∅) + g * (v(x,t,c) - v(x,t,∅))
                empty_label = v_model.num_classes  # index for ∅
                c_empty = torch.full_like(c, empty_label)
                v_empty = v_model(x, t, c_empty)
                v_c = v_model(x, t, c)
                v = v_empty + guidance * (v_c - v_empty)

            x = x - v * dt
            t = t - dt

    return x


def compare_inference(v_model, T, K, c, sampler, device=None, guidance: float = 1.0):
    """
    Compare inference results with original sampler / trained v_model.

    Args:
        v_model: trained velocity model with forward(x_t, t, c) -> v.
        T: number of steps.
        K: number of samples (batch size).
        c: class control (see below).
        sampler: callable that samples (x, c) from C classes of blobs in R^N.
        device: torch device; if None, use model's device.
        guidance: guidance scale g passed through to ode_backward.

    Returns:
        x_infer: (K, N) approximate samples at t≈0 from the learned flow (conditioned on c).
        (x_data, c_data): samples from the true sampler with the same class pattern.

    Class control (c can be):
        - int in {0, ..., C-1}: all K samples are from that class.
        - int == C: all samples are uniformly drawn over {0, ..., C-1}.
        - 1D tensor / list / tuple of length K: per-sample class indices in {0, ..., C-1}.
    Here C = v_model.num_classes (data classes); empty label index is also C in the model.
    """
    device = device or next(v_model.parameters()).device
    v_model.eval()
    num_classes = v_model.num_classes  # data classes: 0..C-1, empty label index = C

    # Normalize c into:
    #   cond_labels: (K,) long on device -> for v_model
    #   sample_class_idx: class control passed to sampler(...)
    if isinstance(c, int):
        if c == num_classes:
            # Use all classes uniformly at random
            cond_labels = torch.randint(0, num_classes, (K,), device=device)
            sample_class_idx = num_classes  # triggers random-per-point in simple_sample
        else:
            if not (0 <= c < num_classes):
                raise ValueError(f"c must be in [0, {num_classes-1}] or equal to C for random, got {c}")
            cond_labels = torch.full((K,), c, dtype=torch.long, device=device)
            sample_class_idx = c
    elif torch.is_tensor(c):
        if c.shape[0] != K:
            raise ValueError(f"tensor c must have length K={K}, got {c.shape[0]}")
        c_long = c.to(device=device, dtype=torch.long)
        # Special case: all entries == C -> treat as random over {0,...,C-1}
        if torch.all(c_long == num_classes):
            cond_labels = torch.randint(0, num_classes, (K,), device=device)
            sample_class_idx = num_classes  # triggers random-per-point in simple_sample
        else:
            if not torch.all((0 <= c_long) & (c_long < num_classes)):
                raise ValueError(f"tensor c entries must be in [0, {num_classes-1}] or all equal to C for random")
            cond_labels = c_long
            sample_class_idx = cond_labels
    else:
        # list / tuple
        c_tensor = torch.as_tensor(c, dtype=torch.long)
        if c_tensor.shape[0] != K:
            raise ValueError(f"c must have length K={K}, got {c_tensor.shape[0]}")
        cond_labels = c_tensor.to(device=device)
        sample_class_idx = cond_labels

    x_infer = ode_backward(v_model, T, K, cond_labels, device=device, guidance=guidance)
    x_sampler = sampler(K, device=device, class_idx=sample_class_idx)
    return x_infer, x_sampler


