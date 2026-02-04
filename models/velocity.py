"""
Flow-matching velocity model v_θ(x_t, t, c) that predicts instantaneous velocity.

Features:
- Class embedding for C+1 classes (0,...,C-1 data classes, C = empty label ∅).
- Simple time embedding.
- Concat [x_t, class_emb(c), time_emb(t)] and forward through an MLP.
"""

import torch
import torch.nn as nn
import math


class VelocityMLP(nn.Module):
    """
    Predicts instantaneous velocity v_θ(x_t, t, c) for flow matching.

    Args:
        num_classes: number of data classes (model has num_classes + 1 for empty label ∅).
        dim: ambient dimension N (x_t in R^N).
        hidden_dim: hidden size of MLP.
        num_layers: number of hidden layers.
        embed_dim: dimension of class and time embeddings.
    """

    def __init__(
        self,
        num_classes,
        dim,
        hidden_dim=256,
        num_layers=4,
        embed_dim=64,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.embed_dim = embed_dim
        # C+1 classes (including empty)
        self.class_embed = nn.Embedding(num_classes + 1, embed_dim)
        # Time t in [0,1] -> embed_dim
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        inp_dim = dim + embed_dim + embed_dim  # x_t + class_emb + time_emb
        layers = []
        layers.append(nn.Linear(inp_dim, hidden_dim))
        layers.append(nn.SiLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_t, t, c):
        """
        Args:
            x_t: (B, N) current state.
            t: (B,) or (B, 1) time in [0, 1].
            c: (B,) long class indices in {0, ..., num_classes} (num_classes = ∅).

        Returns:
            v: (B, N) predicted velocity.
        """
        B = x_t.shape[0]
        if t.dim() == 1:
            t = t.view(-1, 1)
        c_emb = self.class_embed(c)           # (B, embed_dim)
        t_emb = self.time_embed(t)           # (B, embed_dim)
        h = torch.cat([x_t, c_emb, t_emb], dim=-1)  # (B, N + 2*embed_dim)
        return self.mlp(h)
