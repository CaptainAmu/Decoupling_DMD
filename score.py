"""
Clean wrappers around a pretrained velocity model v(x_t, t, c):

- CleanPredModel(v_model): forward(x_t, t, c) -> x*(x_t,t,c) = x_t - t v(x_t,t,c)
- ScoreModel(v_model):     forward(x_t, t, c, g=1.0) -> score s(x_t,t,c,g)

We intentionally do not export functional helpers here; use the two nn.Module
wrappers directly so that you always call them as score_model(xt, t, c, g).
"""

import torch
import torch.nn as nn

from inference import ode_backward


class CleanPredModel(nn.Module):
    """
    Wraps a pretrained velocity model v(x_t, t, c) into a clean-prediction model
    x*(x_t, t, c) = x_t - t * v(x_t, t, c).
    """

    def __init__(self, velocity_model, guidance = 4.5):
        super().__init__()
        self.velocity_model = velocity_model
        self.guidance = guidance

    def forward(self, x_t, t, c):
        B = x_t.shape[0]
        device = x_t.device
        if not isinstance(t, torch.Tensor):
            t = torch.full((B,), t, device=device, dtype=x_t.dtype)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if c is None:
            c = torch.randint(0, self.velocity_model.num_classes, (B,), device=device, dtype=torch.long)
        elif isinstance(c, int):
            c = torch.full((B,), c, dtype=torch.long, device=device)

        c_empty = torch.full((B,), self.velocity_model.num_classes, dtype=torch.long, device=device)
        v_empty = self.velocity_model(x_t, t.squeeze(-1), c_empty)
        v = self.velocity_model(x_t, t.squeeze(-1), c)
        return x_t - t * (v_empty + self.guidance * (v - v_empty))


class ScoreModel(nn.Module):
    """
    Wraps a pretrained velocity model v(x_t, t, c) into a score model.

    Default (g=1.0, no class guidance):
        s*(x_t, t, c) = (t-1)/t * v(x_t, t, c) - (1/t) * x_t.

    With guidance g and empty label ∅:
        s(x_t, t, c, g) = (t-1)/t * v(x_t, t, ∅) - 1/t * x_t
                          + g * ( (t-1)/t * (v(x_t,t,c) - v(x_t,t,∅)) )
    """

    def __init__(self, velocity_model, guidance = 1.0, empty_label=None): # empty_label is the unconditional label, in training used C by default.
        super().__init__()
        self.velocity_model = velocity_model
        self.empty_label = empty_label
        self.guidance = guidance

    def forward(self, x_t, t, c):
        v_model = self.velocity_model
        empty_label = self.empty_label
        if empty_label is None:
            empty_label = v_model.num_classes # Unconditional label
        B = x_t.shape[0]
        device = x_t.device
        if not isinstance(t, torch.Tensor):
            t = torch.full((B,), t, device=device, dtype=x_t.dtype)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_safe = t.clamp(min=1e-5)
        if c is None:
            c = torch.randint(0, v_model.num_classes, (B,), device=device, dtype=torch.long)  # Set random target labels for all samples
        elif isinstance(c, int):
            c = torch.full((B,), c, dtype=torch.long, device=device) # Set target label as c for all samples

        # unconditional velocity (empty label)
        c_empty = torch.full((B,), empty_label, dtype=torch.long, device=device)
        v_empty = v_model(x_t, t.squeeze(-1), c_empty)
        v_c = v_model(x_t, t.squeeze(-1), c)

        # base unconditional score term
        term1 = (t_safe - 1) / t_safe * v_empty - (1.0 / t_safe) * x_t
        # guided correction
        term2 = self.guidance * ((t_safe - 1) / t_safe) * (v_c - v_empty)
        return term1 + term2
