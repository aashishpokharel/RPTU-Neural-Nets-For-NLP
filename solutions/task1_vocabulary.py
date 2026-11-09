"""TASK 1 (reference solution) -- vocabulary."""

from __future__ import annotations

from collections import Counter

import numpy as np

from config import CORPUS_PATH, DEFAULT_MIN_COUNT


def read_tokens(path=CORPUS_PATH, limit=None):
    tokens = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            tokens.extend(line.lower().split())
            if limit is not None and len(tokens) >= limit:
                return tokens[:limit]
    return tokens


def build_vocab(tokens, min_count=DEFAULT_MIN_COUNT):
    counter = Counter(tokens)
    items = [(w, c) for w, c in counter.items() if c >= min_count]
    items.sort(key=lambda wc: (-wc[1], wc[0]))

    idx2word = [w for w, _ in items]
    word2idx = {w: i for i, w in enumerate(idx2word)}
    counts = np.array([c for _, c in items], dtype=np.int64)
    return counts, word2idx, idx2word


def encode(tokens, word2idx):
    return [word2idx[t] for t in tokens if t in word2idx]
