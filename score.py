"""
Given a fixed pretrained velocity model v(x_t, t, c), define:
- clean-prediction: x(x_t, t, c) := x_t - t * v(x_t, t, c)
- score model: s(x_t, t, c, g) with default g=1; g=0 recovers score for unconditional.
"""

import torch


def clean_pred(v_model, x_t, t, c):
    """
    Clean-prediction: x(x_t, t, c) := x_t - t * v(x_t, t, c).

    Args:
        v_model: pretrained velocity model with forward(x_t, t, c) -> v.
        x_t: (B, N).
        t: (B,) or (B, 1).
        c: (B,) long.

    Returns:
        x: (B, N) predicted clean x_0.
    """
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    v = v_model(x_t, t.squeeze(-1), c)
    return x_t - t * v


def score_s(v_model, x_t, t, c, g=1.0, empty_label=None):
    """
    Score model:
        s(x_t, t, c, g) = (t-1)/t * v(x_t, t, ∅) - 1/t * x_t
                          + g * ( (t-1)/t * v(x_t, t, c) - (t-1)/t * v(x_t, t, ∅) )
    With ∅ = empty label (index num_classes). Default g=1; g=0 recovers unconditional score.

    Args:
        v_model: pretrained velocity model with forward(x_t, t, c) -> v.
        x_t: (B, N).
        t: (B,) or (B, 1). Should be in (0, 1]; avoid t=0 in practice.
        c: (B,) long class indices.
        g: guidance scale; default 1; g=0 gives score for unconditional (∅).
        empty_label: index for ∅; default v_model.num_classes.

    Returns:
        s: (B, N) score.
    """
    if empty_label is None:
        empty_label = v_model.num_classes
    B = x_t.shape[0]
    device = x_t.device
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    t_safe = t.clamp(min=1e-5)  # avoid div by zero
    c_empty = torch.full((B,), empty_label, dtype=torch.long, device=device)
    v_empty = v_model(x_t, t.squeeze(-1), c_empty)
    v_c = v_model(x_t, t.squeeze(-1), c)
    term1 = (t_safe - 1) / t_safe * v_empty - (1.0 / t_safe) * x_t
    term2 = g * ((t_safe - 1) / t_safe) * (v_c - v_empty)
    return term1 + term2
