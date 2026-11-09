"""
grader/task2.py -- checks for TASK 2 (skip-gram / CBOW training pairs).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DEFAULT_WINDOW
from grader.runner import current_exercise_module
from grader.support import require

CHECKS = []


def check(fn):
    CHECKS.append((fn.__doc__, fn))
    return fn


def exercise():
    return current_exercise_module(2)


def as_pairs(pairs, name="pairs"):
    require(isinstance(pairs, list),
            f"{name} must be a list, got {type(pairs).__name__}")
    for item in pairs[:50]:
        require(isinstance(item, (tuple, list)) and len(item) == 2,
                f"every entry of {name} must be a 2-tuple, got {item!r}")
    return [(int(a), int(b)) for a, b in pairs]


@check
def skipgram_pairs_are_valid():
    """skip-gram: list of (center, context) ids inside the window"""
    ex = exercise()
    stream = list(range(40))                      # every token has a unique id
    pairs = as_pairs(ex.generate_skipgram_pairs(stream, window=3,
                                                rng=np.random.default_rng(0)))
    require(len(pairs) > 150, f"expected hundreds of pairs, got {len(pairs)}")
    for center, context in pairs:
        require(0 <= center < 40 and 0 <= context < 40,
                f"ids must stay inside the corpus, got ({center}, {context})")
        require(center != context, "a word must never be paired with itself")
        require(abs(center - context) <= 3,
                f"|center - context| must be <= window (3), got "
                f"({center}, {context})")
    distances = {abs(c - x) for c, x in pairs}
    require(distances == {1, 2, 3},
            f"with window=3 the pairs must cover distances 1..3, got {sorted(distances)}")


@check
def skipgram_window_is_dynamic():
    """skip-gram: the window is sampled at random, so near words pair up more often"""
    ex = exercise()
    stream = list(range(400))
    pairs = as_pairs(ex.generate_skipgram_pairs(stream, window=5,
                                                rng=np.random.default_rng(0)))
    counts = Counter(abs(c - x) for c, x in pairs)
    require(max(counts.keys()) == 5, f"the largest distance must be the window (5), "
                                     f"got {max(counts.keys())}")
    ordered = [counts[d] for d in range(1, 6)]
    require(all(a > b for a, b in zip(ordered, ordered[1:])),
            f"distance counts must shrink with distance (dynamic window), got {ordered}")
    require(ordered[0] > 3 * ordered[4],
            f"distance 1 should be much more frequent than distance 5, got {ordered}")
    forward = {(c, x) for c, x in pairs if x > c}
    backward = {(c, x) for c, x in pairs if x < c}
    require(forward and backward,
            "both directions must be generated: (i, i+k) and (k, i)")



@check
def skipgram_edge_cases_and_reproducibility():
    """skip-gram: no pairs for one token, both directions for two, seeds repeat"""
    ex = exercise()
    require(as_pairs(ex.generate_skipgram_pairs([7], window=2,
                                                rng=np.random.default_rng(0))) == [],
            "a stream with a single token cannot produce any pair")
    pair_set = set(as_pairs(ex.generate_skipgram_pairs([0, 1], window=2,
                                                       rng=np.random.default_rng(0))))
    require(pair_set == {(0, 1), (1, 0)},
            f"two tokens must give {{(0, 1), (1, 0)}}, got {sorted(pair_set)}")

    stream = list(range(60))
    first = as_pairs(ex.generate_skipgram_pairs(stream, window=4,
                                                rng=np.random.default_rng(7)))
    second = as_pairs(ex.generate_skipgram_pairs(stream, window=4,
                                                 rng=np.random.default_rng(7)))
    third = as_pairs(ex.generate_skipgram_pairs(stream, window=4,
                                                rng=np.random.default_rng(8)))
    require(first == second, "the same `rng` seed must give the same pairs")
    require(first != third, "a different `rng` seed must give a different window sample")

    default = as_pairs(ex.generate_skipgram_pairs(stream), "default pairs")
    require(len(default) > 0, "calling generate_skipgram_pairs without `rng` must work")
    require(max(abs(c - x) for c, x in default) <= DEFAULT_WINDOW,
            f"the default window must be {DEFAULT_WINDOW}")


@check
def cbow_pairs_are_valid():
    """CBOW: (context, center) with a non-empty list of neighbours"""
    ex = exercise()
    stream = list(range(30))
    examples = ex.generate_cbow_pairs(stream, window=2, rng=np.random.default_rng(0))
    require(isinstance(examples, list), "generate_cbow_pairs must return a list")
    require(len(examples) == len(stream),
            f"every token must appear once as a center word, got {len(examples)} "
            f"examples for {len(stream)} tokens")
    for item in examples:
        require(isinstance(item, (tuple, list)) and len(item) == 2,
                f"every example must be (context, center), got {item!r}")
        context, center = item
        center = int(center)
        context = [int(v) for v in context]
        require(0 <= center < 30, f"center id out of range: {center}")
        require(len(context) > 0, "the context of a center word must not be empty")
        require(len(context) <= 4,
                f"with window=2 a context has at most 4 words, got {len(context)}")
        require(center not in context, "the center word must not be part of the context")
        require(all(0 <= v < 30 for v in context), f"context id out of range: {context}")
        require(max(abs(v - center) for v in context) <= 2,
                f"context word beyond the window: center {center}, context {context}")
    same = ex.generate_cbow_pairs(stream, window=2, rng=np.random.default_rng(5))
    again = ex.generate_cbow_pairs(stream, window=2, rng=np.random.default_rng(5))
    require([(list(c), int(y)) for c, y in same] ==
            [(list(c), int(y)) for c, y in again],
            "the same `rng` seed must give the same CBOW examples")


@check
def dataset_wraps_the_pairs_for_the_dataloader():
    """SkipGramDataset: len + (center, context) long tensors, and it batches"""
    ex = exercise()
    pairs = [(0, 1), (1, 0), (3, 2), (2, 3)]
    dataset = ex.SkipGramDataset(pairs)
    require(len(dataset) == len(pairs),
            f"len(dataset) must be the number of pairs ({len(pairs)}), got {len(dataset)}")
    center, context = dataset[2]
    require(torch.is_tensor(center) and torch.is_tensor(context),
            "dataset[i] must return two tensors, otherwise DataLoader cannot collate them")
    require(center.dtype == torch.long and context.dtype == torch.long,
            f"ids must be torch.long (int64), got {center.dtype}")
    require((int(center), int(context)) == pairs[2],
            f"dataset[2] returned {(int(center), int(context))}, expected {pairs[2]}")
    for index in range(len(pairs)):
        item = dataset[index]
        require((int(item[0]), int(item[1])) == pairs[index],
                f"dataset[{index}] does not match pairs[{index}]")

    loader = DataLoader(dataset, batch_size=3)
    batches = list(loader)
    require(len(batches) == 2, f"batch_size=3 over 4 pairs must give 2 batches, "
                               f"got {len(batches)}")
    centers, contexts = batches[0]
    require(tuple(centers.shape) == (3,) and tuple(contexts.shape) == (3,),
            f"a batch must be two (3,) tensors, got {tuple(centers.shape)} and "
            f"{tuple(contexts.shape)}")
    require(torch.equal(centers, torch.tensor([p[0] for p in pairs[:3]])),
            "the batches must follow the order of the pairs")

    empty = ex.SkipGramDataset([])
    require(len(empty) == 0, "an empty pair list must give an empty dataset")

