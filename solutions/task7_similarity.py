"""TASK 7 (reference solution) -- cosine similarity, neighbours and analogies in torch."""

from __future__ import annotations

import torch

from config import QUESTIONS_PATH


def cosine_similarity(a, b):
    """Cosine similarity of two 1-D vectors as a Python float (0.0 for a zero vector)."""
    a = torch.as_tensor(a).flatten()
    b = torch.as_tensor(b, dtype=a.dtype).flatten()
    denominator = a.norm() * b.norm()
    if float(denominator) == 0.0:
        return 0.0
    return float(torch.dot(a, b) / denominator)


def cosine_scores(vectors, target):
    """One cosine similarity per row of `vectors`, shape (V,) -- no Python loop needed."""
    table = torch.as_tensor(vectors).detach()
    target = torch.as_tensor(target, dtype=table.dtype).flatten()
    numerator = table @ target
    denominator = table.norm(dim=1) * target.norm()
    return numerator / denominator.clamp_min(torch.finfo(table.dtype).tiny)


def most_similar(vectors, word2idx, idx2word, word, topn=8):
    """The `topn` nearest neighbours of `word` (itself excluded), best first."""
    if word not in word2idx:
        raise KeyError(f"{word!r} is not in the vocabulary")
    index = int(word2idx[word])
    scores = cosine_scores(vectors, torch.as_tensor(vectors)[index])
    scores[index] = float("-inf")
    order = torch.argsort(scores, descending=True)
    return [(idx2word[int(i)], float(scores[int(i)])) for i in order[:topn]]


def analogy(vectors, word2idx, idx2word, a, b, c, topn=4):
    """3CosAdd of [P1] Sec. 4: answer b - a + c, with a, b and c excluded."""
    table = torch.as_tensor(vectors)
    target = table[word2idx[b]] - table[word2idx[a]] + table[word2idx[c]]
    scores = cosine_scores(table, target)
    for word in (a, b, c):
        scores[word2idx[word]] = float("-inf")
    order = torch.argsort(scores, descending=True)
    return [(idx2word[int(i)], float(scores[int(i)])) for i in order[:topn]]


def load_questions(path=QUESTIONS_PATH):
    """Parse questions.txt into (a, b, c, d, category); ': ...' lines set the category."""
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


def evaluate_analogies(vectors, word2idx, idx2word, questions):
    """Accuracy over the questions, split into semantic/syntactic and per category."""
    report = {"accuracy": 0.0, "semantic": 0.0, "syntactic": 0.0,
              "per_category": {}, "n": 0}
    hits = {"semantic": [0, 0], "syntactic": [0, 0]}
    for a, b, c, d, category in questions:
        if not all(word in word2idx for word in (a, b, c, d)):
            continue
        predicted = analogy(vectors, word2idx, idx2word, a, b, c, topn=1)[0][0]
        correct = int(predicted == d)
        bucket = report["per_category"].setdefault(category, {"correct": 0, "total": 0,
                                                             "accuracy": 0.0})
        bucket["correct"] += correct
        bucket["total"] += 1
        bucket["accuracy"] = bucket["correct"] / bucket["total"]
        report["n"] += 1
        for kind in ("semantic", "syntactic"):
            if category.lower().startswith(kind):
                hits[kind][0] += correct
                hits[kind][1] += 1
    if report["n"]:
        report["accuracy"] = sum(b["correct"] for b in report["per_category"].values()) / report["n"]
    for kind, (correct, total) in hits.items():
        report[kind] = (correct / total) if total else 0.0
    return report
