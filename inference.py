import torch

"""
Inference: integrate ODE backward from noise (t=1) to data (t=0) using
the trained velocity field v_θ, with T steps and step size 1/T.
"""


def ode_backward(v_model, T, K, c, device=None, guidance: float = 1.0, steps=None):
    """
    Run ODE backward from t=1 (noise) to t≈0 (data) in `steps` steps, step size 1/steps.

    dx/dt = v_θ(x, t, c), integrate backward: x_{t - dt} = x_t - v_θ(x_t, t, c) * dt with dt=1/steps.

    Args:
        v_model: trained velocity model with forward(x_t, t, c) -> v.
        T: default number of steps when steps is None.
        K: number of samples (batch size).
        c: int in {0, ..., C-1, C} or None. 0..C-1 = condition on that class; C or None = use
           label C (same as dropped label in training), i.e. unconditional / class-agnostic.
        steps: number of steps; dt = 1/steps. If None, use T.
        device: torch device; if None, use model's device.
        guidance: guidance scale g. When g != 1, use v_guided = v(∅) + g*(v(c)-v(∅)).

    Returns:
        x: (K, N) approximate samples at t≈0.
    """
    steps = steps if steps is not None else T
    dt = 1.0 / T
    device = device or next(v_model.parameters()).device
    v_model.eval()
    N = v_model.dim
    C = v_model.num_classes

    if c is None:
        cond_labels = torch.randint(0, C, (K,), device=device, dtype=torch.long)
    elif isinstance(c, int) and 0 <= c <= C:
        cond_labels = torch.full((K,), c, dtype=torch.long, device=device)
    else:
        raise ValueError(f"c class must be an int in {0,..., C} (C for unconditional), or None (uniform over all classes), got {c}")

    with torch.no_grad():
        x = torch.randn(K, N, device=device)
        t = torch.full((K,), 1.0, device=device)

        for _ in range(steps):
            if guidance == 1.0:
                v = v_model(x, t, cond_labels)
            else:
                c_empty = torch.full((K,), C, dtype=torch.long, device=device)
                v_empty = v_model(x, t, c_empty)
                v_c = v_model(x, t, cond_labels)
                v = v_empty + guidance * (v_c - v_empty)

            x = x - v * dt
            t = t - dt

    return x


def compare_inference(v_model, T, K, c, sampler, device=None, guidance: float = 1.0):
    """
    Compare inference results with the original sampler.

    Args:
        v_model: trained velocity model with forward(x_t, t, c) -> v.
        T: number of steps (passed to ode_backward; steps=T).
        K: number of samples (batch size).
        c: int in {0, ..., C-1, C} or None. 
            0...C-1 = condition on that class.
            C = unconditional (same as dropped label in training).
            None = uniform over all classes.
        sampler: callable(K, device=, class_idx=) returning (x, c) or just x from the data distribution.
        device: torch device; if None, use model's device.
        guidance: guidance scale passed to ode_backward.

    Returns:
        x_infer: (K, N) samples at t≈0 from the learned flow (conditioned on c).
        x_sampler: samples from the true sampler with the same class pattern (same as sampler output).
    """
    C = v_model.num_classes
    if c is not None and not (isinstance(c, int) and 0 <= c <= C):
        raise ValueError(f"c must be an int in [0, {C}], or None for uniform, got {c}")

    x_infer = ode_backward(v_model, T, K, c, device=device, guidance=guidance, steps=T)
    x_sampler = sampler(K, device=device, class_idx=c)
    return x_infer, x_sampler


