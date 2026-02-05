"""
One-step generator G_θ(z, c) for DMD distillation.

Maps noise z ~ N(0, I_N) and class c to a sample x = G_θ(z, c) in R^N.
Small MLP: concat [z, class_emb(c)] -> hidden -> x.
"""

import torch
import torch.nn as nn


class OneStepGenerator(nn.Module):
    """
    One-step generator G_θ(z, c): maps (z, c) to x in R^N.

    Args:
        num_classes: number of data classes (labels 0, ..., num_classes-1).
        dim: ambient dimension N (z and x in R^N).
        hidden_dim: hidden size of MLP.
        num_layers: number of hidden layers.
        embed_dim: dimension of class embedding.
    """

    def __init__(
        self,
        num_classes,
        dim,
        hidden_dim=128,
        num_layers=3,
        embed_dim=32,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.embed_dim = embed_dim
        self.class_embed = nn.Embedding(num_classes + 1, embed_dim)  # +1 for empty label
        inp_dim = dim + embed_dim
        layers = []
        layers.append(nn.Linear(inp_dim, hidden_dim))
        layers.append(nn.SiLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z, c):
        """
        Args:
            z: (B, N) noise.
            c: (B,) long class indices in {0, ..., num_classes}, or int (broadcast).

        Returns:
            x: (B, N) generated sample.
        """
        B = z.shape[0]
        device = z.device
        if isinstance(c, int):
            c = torch.full((B,), c, dtype=torch.long, device=device)
        c_emb = self.class_embed(c)  # (B, embed_dim)
        h = torch.cat([z, c_emb], dim=-1)  # (B, N + embed_dim)
        return self.mlp(h)
