"""
grader/task1.py -- checks for TASK 1 (reading the corpus, vocabulary, encoding).

Every check raises AssertionError on failure; a NotImplementedError (the `raise` that is
already in your stub) is reported as "not implemented yet" instead of "wrong".
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from config import DEFAULT_MIN_COUNT
from grader.runner import current_exercise_module
from grader.support import close, corpus, require
from grader.support import build_vocab as reference_vocab

CHECKS = []


def check(fn):
    CHECKS.append((fn.__doc__, fn))
    return fn


def exercise():
    return current_exercise_module(1)


@check
def read_tokens_lowercases_and_truncates():
    """read_tokens: lower-cased words on whitespace, `limit` keeps the first N"""
    ex = exercise()
    full = ex.read_tokens()
    require(isinstance(full, list), "read_tokens must return a list, "
                                    f"got {type(full).__name__}")
    expected = corpus(None)
    require(len(full) == len(expected),
            f"read_tokens() returned {len(full)} tokens, expected {len(expected)}")
    require(all(isinstance(t, str) for t in full), "every token must be a str")
    require(all(t == t.lower() for t in full), "tokens must be lower-cased")
    require(all(t and t.split() == [t] for t in full),
            "tokens must not contain whitespace or be empty")
    require(full == expected, "the token stream does not match a plain "
                             "line.lower().split() of the corpus")

    head = ex.read_tokens(limit=50)
    require(len(head) == 50, f"read_tokens(limit=50) returned {len(head)} tokens")
    require(head == expected[:50], "read_tokens(limit=50) must be the first 50 tokens")
    require(ex.read_tokens(limit=0) == [], "read_tokens(limit=0) must be empty")


@check
def build_vocab_drops_rare_words():
    """build_vocab: min_count filtering, frequency order, matching ids/counts"""
    ex = exercise()
    tokens = "a a a b b c d d d d".split()          # a:3 b:2 c:1 d:4
    result = ex.build_vocab(tokens, min_count=2)
    require(isinstance(result, tuple) and len(result) == 3,
            "build_vocab must return the triple (counts, word2idx, idx2word)")
    counts, word2idx, idx2word = result

    require(list(idx2word) == ["d", "a", "b"],
            f"idx2word must be frequency sorted and skip rare words, "
            f"got {list(idx2word)}")
    require(counts.shape == (3,), f"counts has shape {counts.shape}, expected (3,)")
    require(np.issubdtype(np.asarray(counts).dtype, np.integer),
            f"counts must hold integers, got dtype {np.asarray(counts).dtype}")
    require(list(map(int, counts)) == [4, 3, 2],
            f"counts must be [4, 3, 2] in id order, got {list(map(int, counts))}")
    require(dict(word2idx) == {"d": 0, "a": 1, "b": 2},
            f"word2idx does not match idx2word: {dict(word2idx)}")


@check
def build_vocab_ties_and_real_corpus():
    """build_vocab: ties broken alphabetically, counts sum to the kept tokens"""
    ex = exercise()
    counts, word2idx, idx2word = ex.build_vocab("pear apple fig apple fig pear".split(),
                                                min_count=1)
    require(list(idx2word) == ["apple", "fig", "pear"],
            f"equal counts must be ordered alphabetically, got {list(idx2word)}")

    tokens = corpus(6000)
    counts, word2idx, idx2word = ex.build_vocab(tokens)
    require(word2idx[idx2word[0]] == 0, "word2idx and idx2word are out of sync")
    require(idx2word[0] == "the", f"the most frequent word must get id 0, "
                                  f"got {idx2word[0]!r}")
    kept = Counter(tokens)
    expected_counts, expected_word2idx, expected_idx2word = reference_vocab(
        tokens, min_count=DEFAULT_MIN_COUNT)
    require(list(idx2word) == list(expected_idx2word),
            "the vocabulary of the real corpus does not match min_count="
            f"{DEFAULT_MIN_COUNT} + frequency/alphabetical ordering")
    require(dict(word2idx) == dict(expected_word2idx), "word2idx indices are off")
    require(list(map(int, counts)) == list(map(int, expected_counts)),
            "counts do not match the corpus frequencies")
    require(all(c >= DEFAULT_MIN_COUNT for c in map(int, counts)),
            f"every kept word must occur at least {DEFAULT_MIN_COUNT} times")
    require(int(np.sum(counts)) == sum(c for c in kept.values()
                                       if c >= DEFAULT_MIN_COUNT),
            "counts.sum() must equal the number of tokens inside the vocabulary")


@check
def encode_maps_words_to_ids():
    """encode: keeps the token order and silently drops out-of-vocabulary words"""
    ex = exercise()
    _counts, word2idx, _idx2word = ex.build_vocab("a a b c c c".split(), min_count=2)
    ids = ex.encode(["c", "zzz", "a", "c"], word2idx)
    require(np.issubdtype(np.asarray(ids).dtype, np.integer),
            f"encode must return integers, got dtype {np.asarray(ids).dtype}")
    require(list(map(int, ids)) == [word2idx["c"], word2idx["a"], word2idx["c"]],
            f"encode([...]) returned {list(map(int, ids))}, expected "
            "[id(c), id(a), id(c)] with 'zzz' dropped")

    tokens = corpus(6000)
    counts, word2idx, _idx2word = ex.build_vocab(tokens)
    ids = ex.encode(tokens, word2idx)
    require(len(ids) == int(np.sum(counts)),
            f"encode(all tokens) has length {len(ids)}, expected {int(np.sum(counts))} "
            "(only out-of-vocabulary words may be dropped)")
    require(min(ids) >= 0 and max(ids) < len(word2idx), "encode produced an invalid id")
    close(ids[:25], [word2idx[t] for t in tokens if t in word2idx][:25], 0,
          "encode must keep the order of the input tokens")
