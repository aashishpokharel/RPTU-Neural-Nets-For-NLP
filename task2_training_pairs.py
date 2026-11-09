"""
TASK 2 -- training pairs, the dynamic window, and a Dataset for the DataLoader.

    [P1] Sec. 3.2 (Skip-gram and CBOW pairs), Sec. 4.2 (the dynamic window)

Skip-gram: every token of the corpus becomes a *centre word* and every neighbour inside
the window becomes a *context word*:

    the quick brown fox jumps
            ^ centre = "brown" (position 2), window = 2
    pairs: (brown, the), (brown, quick), (brown, fox), (brown, jumps)

CBOW is the other way round: the neighbours predict the centre word, one example per
centre token, e.g. ([the, quick, fox, jumps], brown).

The dynamic window of [P1] Sec. 4.2 samples the effective window size

    r ~ U{1, ..., window}

once per centre word, so nearby words are seen more often than distant ones -- it is a
cheap form of weighting the context by distance.

Why the pairs are a list and not an array: they are *generated fresh for every epoch*,
and PyTorch would like them as a `Dataset` anyway (see `SkipGramDataset`), so that
`DataLoader` can shuffle and batch them for you.

Run this file on its own to check yourself:

    python task2_training_pairs.py
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from config import DEFAULT_WINDOW


def generate_skipgram_pairs(token_ids, window=DEFAULT_WINDOW, rng=None):
    """Return every (center, context) pair of one epoch, as a list of 2-tuples of ints.

    Parameters
    ----------
    token_ids : list[int]
        The encoded corpus (task 1): the ids of the words, in order.
    window : int
        Maximum distance between a centre and its context words.
    rng : np.random.Generator or None
        Source of the random window sizes.  When it is None, draw from a fresh
        `np.random.default_rng()`; when it is given, the same seed must reproduce exactly
        the same pairs.

    Returns
    -------
    list[tuple[int, int]]
        (center_id, context_id) for every pair in the corpus, in corpus order.

    Hints
    -----
    * `effective = rng.integers(1, window + 1)` gives r in {1, ..., window}.
    * clamp the position range to the corpus: `max(0, i - r)` and
      `min(len(token_ids) - 1, i + r)`.
    * every neighbour *except* the centre word itself is a pair, on both sides, so the
      pairs (i, i+1) and (i+1, i) both appear.
    * a corpus with a single token produces no pair at all.
    """
    if rng is None:
        rng = np.random.default_rng()

    
    skipgrap_pairs = []
    for idx, token in enumerate(token_ids):

        effective_window = rng.integers(1, window+1)
        for i in range(1, effective_window+1):
            if idx - i >= 0:
                skipgrap_pairs.append((token, token_ids[idx-i]))
            if idx+i < len(token_ids):
                skipgrap_pairs.append((token, token_ids[idx+i]))
    
    return skipgrap_pairs
    # raise NotImplementedError("Task 2: generate_skipgram_pairs()")


def generate_cbow_pairs(token_ids, window=DEFAULT_WINDOW, rng=None):
    """Return the CBOW examples of one epoch: one (context, center) per centre token.

    Parameters
    ----------
    token_ids : list[int]
    window : int
    rng : np.random.Generator or None
        Same dynamic-window rule as Skip-gram: one window size per centre word.

    Returns
    -------
    list[tuple[list[int], int]]
        (context_ids, center_id), where context_ids is the *list* of the ids inside the
        window (possibly shorter near the edges of the corpus) and never contains the
        centre word.
    """
    if rng is None:
        rng = np.random.default_rng()
    
        
    cbow_pairs = []
    for idx, token in enumerate(token_ids):

        effective_window = rng.integers(1, window+1)
        context_ids = []
        for i in range(1, effective_window+1):
            if idx - i >= 0:
                context_ids.append(token_ids[idx-i])
            if idx+i < len(token_ids):
                context_ids.append(token_ids[idx+i])
        cbow_pairs.append((context_ids, token))
    
    return cbow_pairs
    # raise NotImplementedError("Task 2: generate_skipgram_pairs()")
    


class SkipGramDataset(Dataset):
    """A `torch.utils.data.Dataset` that serves the pairs of one epoch.

    `DataLoader` needs three things from you: `__len__`, `__getitem__`, and items that
    can be collated.  Returning two 0-dim `torch.long` tensors means a batch comes out as
    two (N,) long tensors -- exactly what `nn.Embedding` and `F.cross_entropy` want.

    Hint: store the pairs in one `torch.as_tensor(pairs, dtype=torch.long).reshape(-1, 2)`
    tensor (shape (N, 2)) and return `table[index, 0], table[index, 1]`.  Keep in mind that
    an empty pair list must give an empty dataset (len 0), not a crash.
    """

    def __init__(self, pairs):
        self.pairs = pairs
        # raise NotImplementedError("Task 2: SkipGramDataset.__init__()")

    def __len__(self):
        return len(self.pairs)
        # raise NotImplementedError("Task 2: SkipGramDataset.__len__()")

    def __getitem__(self, index):
        # print('The index and pair',index, self.pairs)
        context = torch.tensor(self.pairs[index][0])
        center = torch.tensor(self.pairs[index][-1])

        return context, center
if __name__ == "__main__":
    import sys

    from grader.runner import main as _grade
    from grader.runner import set_module_override

    set_module_override(2, sys.modules[__name__])
    raise SystemExit(_grade(tasks=[2]))
