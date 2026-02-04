"""
Train flow-matching velocity model v_θ(x_t, t, c) with either:
- simple_sample: (x0, c) ~ p̃_data (labels dropped with prob p), ε ~ N(0,I), x_t = (1-t)x0 + tε.
- pair_sample: ε ~ N(0,I), (x0, c) = f(ε), labels dropped with prob p, x_t = (1-t)x0 + tε.

Loss: E ||v_θ(x_t, t, c) - (ε - x0)||^2.
"""

import torch
import torch.nn as nn


def train_flow_matching(
    model,
    data_source,
    num_steps,
    batch_size,
    drop_label_prob=0.0,
    lr=1e-4,
    device=None,
    verbose=True,
):
    """
    Train v_θ to match conditional flow-matching objective.

    Args:
        model: VelocityMLP (or any module with forward(x_t, t, c) -> v).
        data_source: callable(K, device=None) returning either
            (x0, c) for simple_sample, or (x0, c, epsilon) for pair_sample.
        num_steps: number of training steps.
        batch_size: K per step.
        drop_label_prob: p; with probability p set c = empty (num_classes).
        lr: learning rate.
        device: torch device; if None, use cuda if available.
        verbose: print loss every 100 steps.

    Returns:
        List of losses (per step) if verbose or for logging.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    num_classes = model.num_classes  # empty label index = num_classes
    losses = []

    for step in range(num_steps):
        out = data_source(batch_size, device=device)
        if len(out) == 3:
            x0, c, eps = out
        else:
            x0, c = out
            eps = torch.randn_like(x0, device=x0.device)

        # Drop label with prob p -> set c = empty (num_classes)
        if drop_label_prob > 0:
            mask = torch.rand(batch_size, device=device) < drop_label_prob
            c = c.clone()
            c[mask] = num_classes

        t = torch.rand(batch_size, device=device)
        x_t = (1 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * eps
        target = eps - x0

        v = model(x_t, t, c)
        loss = nn.functional.mse_loss(v, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if verbose and (step + 1) % 100 == 0:
            print(f"step {step + 1}/{num_steps} loss {loss.item():.6f}")

    return losses
