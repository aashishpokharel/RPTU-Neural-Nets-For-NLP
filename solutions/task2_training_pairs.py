"""TASK 2 (reference solution) -- training pairs and the PyTorch Dataset that wraps them."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from config import DEFAULT_WINDOW


def generate_skipgram_pairs(token_ids, window=DEFAULT_WINDOW, rng=None):
    """Every (centre, context) pair of one epoch of Skip-gram, flattened into a list."""
    if rng is None:
        rng = np.random.default_rng()
    token_ids = list(token_ids)
    pairs = []
    n = len(token_ids)
    for position, center in enumerate(token_ids):
        effective = int(rng.integers(1, window + 1))          # [P1] Sec. 4.2 dynamic window
        low = max(0, position - effective)
        high = min(n - 1, position + effective)
        for other in range(low, high + 1):
            if other != position:
                pairs.append((int(center), int(token_ids[other])))
    return pairs


def generate_cbow_pairs(token_ids, window=DEFAULT_WINDOW, rng=None):
    """One example per centre token: (averaged context ids, centre id)."""
    if rng is None:
        rng = np.random.default_rng()
    token_ids = list(token_ids)
    pairs = []
    n = len(token_ids)
    for position, center in enumerate(token_ids):
        effective = int(rng.integers(1, window + 1))
        low = max(0, position - effective)
        high = min(n - 1, position + effective)
        context = [int(token_ids[other]) for other in range(low, high + 1) if other != position]
        pairs.append((context, int(center)))
    return pairs


class SkipGramDataset(Dataset):
    """Map-style dataset: index -> (centre id, context id) as two 0-dim long tensors.

    DataLoader turns a batch of these into two (batch,) LongTensors, which is exactly the
    shape nn.Embedding wants.  Keeping the pairs in one (N, 2) tensor avoids per-item
    Python overhead.
    """

    def __init__(self, pairs):
        self.pairs = torch.as_tensor(list(pairs), dtype=torch.long).reshape(-1, 2)

    def __len__(self):
        return int(self.pairs.shape[0])

    def __getitem__(self, index):
        return self.pairs[index, 0], self.pairs[index, 1]

    def __repr__(self):
        return f"SkipGramDataset(len={len(self)})"
