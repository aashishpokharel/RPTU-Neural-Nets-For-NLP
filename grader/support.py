"""
grader/support.py -- shared helpers for the automatic grader.

Three kinds of helpers live here:

* assertion utilities used by the individual checks (`require`, `close`,
  `finite_difference_gradients`);
* conversion utilities that accept whatever a student returns (`to_numpy`, `to_tensor`,
  `input_table`), so a check never crashes just because a result is a tensor, an array or
  a plain float;
* *independent* reference implementations of the boring parts (reading the corpus,
  building a vocabulary, cosine similarity, the analogy metric).  The checks for task N
  must never call the code under test to build their own test data, otherwise a buggy
  task 1 would silently "validate" a buggy task 6.
"""

from __future__ import annotations

import numpy as np
import torch

from config import CORPUS_PATH, QUESTIONS_PATH

# --------------------------------------------------------------------------------------
# assertions
# --------------------------------------------------------------------------------------


def require(condition, message):
    """Raise AssertionError(message) unless `condition` is true."""
    if not condition:
        raise AssertionError(message)


def close(actual, expected, tol=1e-9, message=""):
    """Assert that two arrays/scalars agree up to an absolute+relative tolerance."""
    a = to_numpy(actual).astype(np.float64)
    b = to_numpy(expected).astype(np.float64)
    require(a.shape == b.shape,
            f"shape mismatch: got {a.shape}, expected {b.shape}. {message}")
    if a.size == 0:
        return
    diff = np.max(np.abs(a - b))
    scale = max(1.0, float(np.max(np.abs(b))))
    require(np.allclose(a, b, rtol=tol, atol=tol * scale),
            f"values differ: max |got - expected| = {diff:.3e} "
            f"(tolerance {tol:.0e}). {message}")


def finite_difference_gradients(value_fn, parameters, analytic, eps=1e-4, tol=2e-3,
                                name=""):
    """Check autograd gradients against central finite differences of `value_fn()`.

    `value_fn()` must return a scalar tensor built from `parameters`; `analytic` holds the
    gradients you expect.  Nothing about *how* the gradients were obtained is enforced --
    only that the derivative of the loss you wrote is the one you use to update.
    """
    for index, (parameter, grad) in enumerate(zip(parameters, analytic)):
        got = to_numpy(grad).astype(np.float64).reshape(-1)
        flat = parameter.reshape(-1)
        numeric = np.zeros(flat.numel(), dtype=np.float64)
        for position in range(flat.numel()):
            with torch.no_grad():
                original = float(flat[position])
                flat[position] = original + eps
                plus = float(value_fn().detach())
                flat[position] = original - eps
                minus = float(value_fn().detach())
                flat[position] = original
            numeric[position] = (plus - minus) / (2.0 * eps)
        require(got.size == numeric.size,
                f"gradient {name} #{index}: got {got.size} entries, "
                f"expected {numeric.size}")
        scale = max(1.0, float(np.max(np.abs(numeric))))
        error = float(np.max(np.abs(numeric - got))) / scale
        require(error < tol,
                f"gradient {name} #{index} does not match finite differences "
                f"(relative error {error:.2e}, tolerance {tol:.0e}). "
                "Is the loss the one whose gradient you computed?")


# --------------------------------------------------------------------------------------
# conversions -- accept tensors, arrays or plain numbers
# --------------------------------------------------------------------------------------


def to_numpy(value):
    """numpy view of a tensor / array / list."""
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def to_tensor(value, dtype=torch.float64):
    """float64 tensor view of a tensor / array / list (no grad, on the CPU)."""
    if torch.is_tensor(value):
        return value.detach().to("cpu", dtype)
    return torch.as_tensor(np.asarray(value), dtype=dtype)


def _table(model, attribute, what):
    holder = getattr(model, attribute, None)
    names = ", ".join(name for name, _p in model.named_parameters()) \
        if hasattr(model, "named_parameters") else "?"
    require(holder is not None,
            f"the model must keep its {what} embeddings in `{attribute}` -- an "
            f"nn.Embedding module or an (V, dim) tensor. Your model has: {names}")
    return getattr(holder, "weight", holder)


def input_table(model):
    """The (V, dim) matrix you look the centre word up in (`model.W_in`)."""
    return _table(model, "W_in", "input")


def output_table(model):
    """The (V, dim) matrix you compute the scores with (`model.W_out`)."""
    return _table(model, "W_out", "output")


# --------------------------------------------------------------------------------------
# reference helpers -- deliberately independent of the code under test
# --------------------------------------------------------------------------------------
def read_tokens(limit=None, path=CORPUS_PATH):
    """Whitespace tokeniser used to build test data."""
    tokens = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            tokens.extend(line.lower().split())
            if limit is not None and len(tokens) >= limit:
                return tokens[:limit]
    return tokens


def build_vocab(tokens, min_count=3):
    """Frequency-sorted vocabulary (the reference behaviour of task 1)."""
    from collections import Counter

    counter = Counter(tokens)
    items = [(word, count) for word, count in counter.items() if count >= min_count]
    items.sort(key=lambda item: (-item[1], item[0]))
    idx2word = [word for word, _count in items]
    word2idx = {word: index for index, word in enumerate(idx2word)}
    counts = np.array([count for _word, count in items], dtype=np.int64)
    return counts, word2idx, idx2word


def log_softmax(logits):
    z = to_numpy(logits).astype(np.float64)
    m = np.max(z)
    return z - m - np.log(np.sum(np.exp(z - m)))


def softmax(logits):
    return np.exp(log_softmax(logits))


def cosine_scores(W, target):
    """Cosine similarity of every row of W with `target`; 0.0 for zero vectors."""
    W = to_numpy(W).astype(np.float64)
    target = to_numpy(target).astype(np.float64)
    den = np.linalg.norm(W, axis=1) * np.linalg.norm(target)
    out = np.zeros(W.shape[0], dtype=np.float64)
    np.divide(W @ target, den, out=out, where=den != 0.0)
    return out


def load_questions(path=QUESTIONS_PATH):
    questions, category = [], "uncategorised"
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                category = line[1:].strip()
                continue
            parts = line.split()
            if len(parts) == 4:
                questions.append((parts[0], parts[1], parts[2], parts[3], category))
    return questions


def analogy_accuracy(W, word2idx, idx2word, questions):
    """3CosAdd accuracy ([P1] Sec. 4) computed without the student's code."""
    W = to_numpy(W).astype(np.float64)
    correct = total = 0
    for a, b, c, d, _category in questions:
        if not all(word in word2idx for word in (a, b, c, d)):
            continue
        target = W[word2idx[b]] - W[word2idx[a]] + W[word2idx[c]]
        scores = cosine_scores(W, target)
        skip = {word2idx[a], word2idx[b], word2idx[c]}
        order = [i for i in np.argsort(-scores) if i not in skip]
        total += 1
        correct += int(idx2word[order[0]] == d)
    return (correct / total) if total else 0.0


def neighbours(W, word2idx, idx2word, word, topn=5):
    """Nearest neighbours by cosine similarity, computed without the student's code."""
    scores = cosine_scores(W, to_numpy(W)[word2idx[word]])
    order = [i for i in np.argsort(-scores) if i != word2idx[word]]
    return [idx2word[i] for i in order[:topn]]


# --------------------------------------------------------------------------------------
# cached fixtures (training is the slow part of the grader, so we do it only once)
# --------------------------------------------------------------------------------------
_CACHE = {}


def corpus(limit=None):
    key = ("tokens", limit)
    if key not in _CACHE:
        _CACHE[key] = read_tokens(limit=limit)
    return _CACHE[key]


def quality_model(limit=12000, dim=40, seed=0):
    """Train the student's model once and reuse it for the "does it learn?" checks."""
    from grader.runner import current_exercise_module

    key = ("quality", limit, dim, seed)
    if key not in _CACHE:
        ex = current_exercise_module(6)
        tokens = corpus(limit)
        _counts, word2idx, idx2word = build_vocab(tokens, min_count=3)
        ids = [word2idx[token] for token in tokens if token in word2idx]
        trained = ex.train_skipgram(ids, len(idx2word), dim=dim, seed=seed)
        require(isinstance(trained, tuple) and len(trained) == 2,
                "train_skipgram must return the pair (model, history) -- the trained "
                "nn.Module and the list of per-epoch mean log-likelihoods")
        model, history = trained
        _CACHE[key] = (to_numpy(input_table(model)).astype(np.float64),
                       [float(value) for value in history], word2idx, idx2word)
    return _CACHE[key]
