"""TASK 3 (reference solution) -- the two embedding matrices as an nn.Module."""

from __future__ import annotations

import torch
from torch import nn

from config import DEFAULT_DIM


class SkipGram(nn.Module):
    """[P1] Fig. 1 as two embedding tables.

    `W_in`  (V, dim) is the table you *look up*:    h = W_in(center_id)
    `W_out` (V, dim) is the table you *score with*: u = W_out.weight @ h

    Both are nn.Embedding modules, so the parameters are called `.weight`, gradients
    accumulate in `.grad`, and `model.parameters()` hands them to the optimizer.
    """

    def __init__(self, vocab_size, dim=DEFAULT_DIM):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.W_in = nn.Embedding(self.vocab_size, self.dim)
        self.W_out = nn.Embedding(self.vocab_size, self.dim)
        self.reset_parameters()

    def reset_parameters(self):
        """[P1] Sec. 4.2: inputs ~ U(-0.5/dim, 0.5/dim), outputs start at zero."""
        nn.init.uniform_(self.W_in.weight, -0.5 / self.dim, 0.5 / self.dim)
        nn.init.zeros_(self.W_out.weight)

    def forward(self, center_ids):
        """Return the logits u of shape (N, V) for a batch of centre-word ids."""
        h = self.W_in(center_ids)                    # (N, dim)
        return h @ self.W_out.weight.t()             # (N, V)


def build_model(vocab_size, dim=DEFAULT_DIM, seed=None):
    """Create a SkipGram model, seeded so that two runs start from the same weights."""
    if seed is not None:
        torch.manual_seed(seed)
    return SkipGram(vocab_size, dim)


def hidden_vector(model, center_ids):
    """h, the hidden layer: a lookup of the input table, shape (N, dim)."""
    return model.W_in(center_ids)
