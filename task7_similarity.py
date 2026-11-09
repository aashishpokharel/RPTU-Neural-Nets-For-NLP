"""
TASK 7 -- cosine similarity, nearest neighbours, analogies and the evaluation.

    [P1] Sec. 4: "3CosAdd"

    b* = argmax_{w != a,b,c} cos(w, b - a + c)

Nothing in this task trains anything: it is the part that turns a (V, dim) table into
answers a human can read.

    cos(u, v) = (u . v) / (|u| |v|)      -- the length of the vectors drops out, which is
                                            why the analogy arithmetic works at all

A word can be the answer of a question only if it is *not* one of the three words of the
question, and a zero vector has no direction at all (its similarity must be 0.0, not NaN).

Run this file on its own to check yourself:

    python task7_similarity.py
"""

from __future__ import annotations

import torch

from config import QUESTIONS_PATH


def cosine_similarity(a, b):
    """Cosine similarity of two 1-D vectors, as a Python float.

    Must return 0.0 when either vector is the zero vector (no division by zero, no NaN).
    """
    a = torch.as_tensor(a).flatten()
    b = torch.as_tensor(b).flatten()

    # The two arguments may have different dtypes (a float64 numpy row and a float32
    # tensor, for instance); compute in a common floating-point dtype.
    dtype = torch.promote_types(a.dtype, b.dtype)
    if not dtype.is_floating_point:
        dtype = torch.float32
    a, b = a.to(dtype), b.to(dtype)

    denominator = a.norm() * b.norm()
    if float(denominator) == 0.0:
        return 0.0                       # a zero vector has no direction
    return float(torch.dot(a, b) / denominator)


def cosine_scores(vectors, target):
    """Cosine similarity of `target` with every row of `vectors`.

    Parameters
    ----------
    vectors : torch.Tensor, shape (V, dim)  (usually `model.W_in.weight`)
    target : torch.Tensor, shape (dim,)

    Returns
    -------
    torch.Tensor, shape (V,)

    Hint: one matrix-vector product and two norms are enough -- no Python loop:
    `scores = vectors @ target / (vectors.norm(dim=1) * target.norm())`, with the
    denominator clamped (or `torch.where`) so that zero rows give 0.0 instead of NaN.
    """
    table = torch.as_tensor(vectors).detach()
    if not table.is_floating_point():
        table = table.float()
    target = torch.as_tensor(target, dtype=table.dtype).flatten()

    numerator = table @ target
    denominator = table.norm(dim=1) * target.norm()
    # clamp_min keeps the division well defined for zero rows / a zero target: the
    # numerator is then zero as well, so the result is 0.0 instead of NaN.
    return numerator / denominator.clamp_min(torch.finfo(table.dtype).tiny)


def most_similar(vectors, word2idx, idx2word, word, topn=8):
    """The `topn` nearest neighbours of `word`, best first, without the word itself.

    Parameters
    ----------
    vectors : torch.Tensor, shape (V, dim)
    word2idx : dict[str, int]
    idx2word : list[str]                (index -> word, the order of the rows)
    word : str
        Must raise KeyError when the word is not in the vocabulary.

    Returns
    -------
    list[(str, float)]
        Sorted by decreasing similarity.  Hint: `torch.argsort(scores, descending=True)`
        and set the score of the query word to `-inf` (or filter it out).
    """
    if word not in word2idx:
        raise KeyError(f"{word!r} is not in the vocabulary")

    table = torch.as_tensor(vectors).detach()
    index = int(word2idx[word])
    scores = cosine_scores(table, table[index])
    scores[index] = float("-inf")        # never its own neighbour
    order = torch.argsort(scores, descending=True)
    return [(idx2word[int(i)], float(scores[int(i)])) for i in order[:topn]]


def analogy(vectors, word2idx, idx2word, a, b, c, topn=4):
    """3CosAdd of [P1]: "a is to b as c is to ?" -- return the best candidates.

    The target vector is `vectors[word2idx[b]] - vectors[word2idx[a]] +
    vectors[word2idx[c]]`; the three words of the question are never answers.

    Returns
    -------
    list[(str, float)]
        The `topn` candidates, best first.  `analogy(..., topn=1)[0][0]` is the answer.
    """
    for word in (a, b, c):
        if word not in word2idx:
            raise KeyError(f"{word!r} is not in the vocabulary")

    table = torch.as_tensor(vectors).detach()
    target = table[word2idx[b]] - table[word2idx[a]] + table[word2idx[c]]
    scores = cosine_scores(table, target)
    for word in (a, b, c):
        scores[int(word2idx[word])] = float("-inf")   # exclude the question itself
    order = torch.argsort(scores, descending=True)
    return [(idx2word[int(i)], float(scores[int(i)])) for i in order[:topn]]


def load_questions(path=QUESTIONS_PATH):
    """Parse `data/questions.txt` into a list of (a, b, c, d, category) tuples.

    Format: lines starting with ':' introduce a category ("semantic capital city",
    "syntactic verb form", ...), blank lines are skipped, and every other line holds four
    words.  The category of a question is the header it follows.
    """
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
    """Score the analogy questions and return a report dictionary.

    The report must contain

        "accuracy"      overall accuracy over the evaluable questions, 0.0 if none
        "semantic"      accuracy of the categories whose name starts with "semantic"
        "syntactic"     accuracy of the categories whose name starts with "syntactic"
        "per_category"  {category: {"correct": int, "total": int, "accuracy": float}}
        "n"             how many questions were evaluated

    A question is evaluable only when all four words are in the vocabulary; questions
    with an unknown word are skipped, never counted as wrong.
    """
    report = {"accuracy": 0.0, "semantic": 0.0, "syntactic": 0.0,
              "per_category": {}, "n": 0}
    correct_total = 0
    split_correct = {"semantic": 0, "syntactic": 0}
    split_total = {"semantic": 0, "syntactic": 0}

    for a, b, c, d, category in questions:
        if any(word not in word2idx for word in (a, b, c, d)):
            continue                      # an impossible question is not a wrong one

        predicted = analogy(vectors, word2idx, idx2word, a, b, c, topn=1)[0][0]
        correct = int(predicted == d)
        report["n"] += 1
        correct_total += correct

        bucket = report["per_category"].setdefault(
            category, {"correct": 0, "total": 0, "accuracy": 0.0})
        bucket["correct"] += correct
        bucket["total"] += 1
        bucket["accuracy"] = bucket["correct"] / bucket["total"]

        name = category.lower()
        for kind in ("semantic", "syntactic"):
            if name.startswith(kind):
                split_correct[kind] += correct
                split_total[kind] += 1

    if report["n"]:
        report["accuracy"] = correct_total / report["n"]
    for kind in ("semantic", "syntactic"):
        if split_total[kind]:
            report[kind] = split_correct[kind] / split_total[kind]
    return report


if __name__ == "__main__":
    import sys

    from grader.runner import main as _grade
    from grader.runner import set_module_override

    set_module_override(7, sys.modules[__name__])
    raise SystemExit(_grade(tasks=[7]))
