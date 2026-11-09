"""
grader/task7.py -- checks for TASK 7 (cosine similarity, neighbours, analogies).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from grader.runner import current_exercise_module
from grader.support import cosine_scores as ref_cosine_scores
from grader.support import load_questions as ref_load_questions
from grader.support import require

CHECKS = []


def check(fn):
    CHECKS.append((fn.__doc__, fn))
    return fn


def exercise():
    return current_exercise_module(7)


def parallelogram_world():
    """a1 -> b1 :: a2 -> b2 is a perfect parallelogram in 2-D."""
    words = ["alpha", "alpha_x", "beta", "beta_x", "noise1", "noise2"]
    W = np.array([[1.0, 0.0], [1.5, 1.0], [2.0, 0.0], [2.5, 1.0], [0.0, 3.0], [-3.0, -0.2]])
    word2idx = {w: i for i, w in enumerate(words)}
    return W, word2idx, words


@check
def cosine_similarity_basics():
    """cosine_similarity: 1 for identical, -1 for opposite, 0 for orthogonal/zero"""
    ex = exercise()
    a = np.array([1.0, 2.0, 3.0])
    require(abs(float(ex.cosine_similarity(a, a.copy())) - 1.0) < 1e-12,
            "a vector must have similarity 1 with itself")
    require(abs(float(ex.cosine_similarity(a, -a)) + 1.0) < 1e-12,
            "opposite vectors must have similarity -1")
    require(abs(float(ex.cosine_similarity(np.array([1.0, 0.0]),
                                           np.array([0.0, 2.0])))) < 1e-12,
            "orthogonal vectors must have similarity 0")
    require(abs(float(ex.cosine_similarity(a, 7.5 * a)) - 1.0) < 1e-12,
            "cosine similarity ignores the length of the vectors")
    require(abs(float(ex.cosine_similarity(np.zeros(3), a))) < 1e-12,
            "the zero vector must give 0.0 instead of a division by zero")
    value = float(ex.cosine_similarity(np.array([1.0, 1.0]), np.array([1.0, 0.0])))
    require(abs(value - (1.0 / np.sqrt(2.0))) < 1e-12,
            f"cos([1, 1], [1, 0]) should be 1/sqrt(2), got {value}")


@check
def cosine_scores_matches_pairwise():
    """cosine_scores: one similarity per row, zero for zero rows"""
    ex = exercise()
    W, word2idx, words = parallelogram_world()
    target = np.array([1.0, 0.7])
    got = np.asarray(ex.cosine_scores(W, target), dtype=np.float64)
    require(got.shape == (len(words),),
            f"cosine_scores has shape {got.shape}, expected ({len(words)},)")
    expected = [float(ex.cosine_similarity(W[i], target)) for i in range(len(words))]
    require(np.allclose(got, expected, atol=1e-12),
            "cosine_scores must equal cosine_similarity row by row")
    require(np.allclose(got, ref_cosine_scores(W, target), atol=1e-12),
            "cosine_scores does not compute the cosine similarity")
    W_zero = W.copy()
    W_zero[4] = 0.0
    zero_scores = np.asarray(ex.cosine_scores(W_zero, target), dtype=np.float64)
    require(np.all(np.isfinite(zero_scores)),
            "a zero row must give 0.0, not NaN or a division-by-zero warning")
    require(abs(float(zero_scores[4])) < 1e-12, "a zero row must give similarity 0.0")
    require(np.all(np.abs(np.asarray(ex.cosine_scores(W, np.zeros(2)))) < 1e-12),
            "a zero target vector must give similarity 0.0 everywhere")


@check
def most_similar_ranks_neighbours():
    """most_similar: sorted by similarity, without the query itself"""
    ex = exercise()
    W, word2idx, words = parallelogram_world()
    result = ex.most_similar(W, word2idx, words, "alpha", topn=3)
    require(isinstance(result, list) and len(result) == 3,
            f"most_similar(topn=3) must return 3 entries, got {result!r}")
    for word, score in result:
        require(word in word2idx, f"{word!r} is not a word of the vocabulary")
        require(word != "alpha", "the queried word must not be its own neighbour")
        require(isinstance(score, float) or np.asarray(score).ndim == 0,
                "every entry must be a (word, similarity) pair")
    scores = [float(s) for _w, s in result]
    require(all(a >= b for a, b in zip(scores, scores[1:])),
            f"the neighbours must be sorted by decreasing similarity, got {scores}")
    expected = ref_cosine_scores(W, W[word2idx["alpha"]])
    best = [words[i] for i in np.argsort(-expected) if words[i] != "alpha"][:3]
    require([w for w, _s in result] == best,
            f"the neighbours of 'alpha' should be {best}, got {[w for w, _s in result]}")
    try:
        ex.most_similar(W, word2idx, words, "does-not-exist", topn=1)
    except KeyError:
        pass
    else:
        raise AssertionError("most_similar must raise KeyError for a word "
                             "outside the vocabulary")


@check
def analogy_solves_the_parallelogram():
    """analogy: a1 -> b1 :: a2 -> ?  returns b2 (and never the input words)"""
    ex = exercise()
    W, word2idx, words = parallelogram_world()
    result = ex.analogy(W, word2idx, words, "alpha", "alpha_x", "beta", topn=2)
    require(isinstance(result, list) and len(result) == 2,
            f"analogy(topn=2) must return 2 candidates, got {result!r}")
    top_word, top_score = result[0]
    require(top_word == "beta_x",
            f"alpha -> alpha_x :: beta -> ? should be 'beta_x', got {top_word!r}")
    require(float(top_score) > 0.99, f"the answer should be nearly parallel to "
                                     f"b1 - a1 + a2, got similarity {float(top_score):.3f}")
    require(not {w for w, _s in result} & {"alpha", "alpha_x", "beta"},
            "the three words of the question must be excluded from the answers")
    scores = [float(s) for _w, s in result]
    require(all(a >= b for a, b in zip(scores, scores[1:])),
            f"the candidates must be sorted by decreasing similarity, got {scores}")
    try:
        ex.analogy(W, word2idx, words, "alpha", "alpha_x", "nope", topn=1)
    except KeyError:
        pass
    else:
        raise AssertionError("analogy must raise KeyError for an unknown word")



@check
def load_questions_parses_the_question_file():
    """load_questions: 5-tuples (a, b, c, d, category) and skips comments/blanks"""
    ex = exercise()
    questions = ex.load_questions()
    expected = ref_load_questions()
    require(isinstance(questions, list), "load_questions must return a list")
    require(len(questions) == len(expected),
            f"questions-words.txt holds {len(expected)} questions, you read "
            f"{len(questions)}")
    require(all(len(q) == 5 for q in questions),
            "every question must be (word_a, word_b, word_c, word_d, category)")
    categories = {q[4] for q in questions}
    require(categories == {q[4] for q in expected},
            f"the categories must be read from the ': ...' header lines, got {categories}")
    require(tuple(questions[0][:4]) == tuple(expected[0][:4]),
            f"the first question should be {expected[0][:4]}, got {questions[0][:4]}")

    handle, path = tempfile.mkstemp(suffix=".txt")
    os.close(handle)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(": semantic capital city\n\nathens greece rome italy\n"
                     "   \nmadrid spain paris france\n")
        parsed = ex.load_questions(path)
        require([tuple(q[:4]) for q in parsed] ==
                [("athens", "greece", "rome", "italy"),
                 ("madrid", "spain", "paris", "france")],
                f"blank lines must be skipped, got {parsed}")
        require({q[4] for q in parsed} == {"semantic capital city"},
                "the category of the ': ...' header must be attached to its questions")
    finally:
        os.remove(path)


@check
def evaluate_analogies_scores_the_questions():
    """evaluate_analogies: accuracy, semantic/syntactic split and per-category counts"""
    ex = exercise()
    W = np.array([
        [1.0, 0.0, 0.0, 0.0],      # athens
        [1.0, 1.0, 0.0, 0.0],      # greece
        [2.0, 0.0, 0.0, 0.0],      # rome
        [2.0, 1.0, 0.0, 0.0],      # italy
        [0.0, 0.0, 1.0, 0.0],      # man
        [0.0, 0.0, 1.0, 1.0],      # woman
        [0.0, 0.0, 2.0, 0.0],      # king
        [0.0, 0.0, 2.0, 1.0],      # queen
    ])
    words = ["athens", "greece", "rome", "italy", "man", "woman", "king", "queen"]
    word2idx = {w: i for i, w in enumerate(words)}
    questions = [("athens", "greece", "rome", "italy", "semantic capital city"),
                 ("man", "woman", "king", "queen", "syntactic verb form")]

    result = ex.evaluate_analogies(W, word2idx, words, questions)
    require(isinstance(result, dict), "evaluate_analogies must return a dict")
    for key in ("accuracy", "semantic", "syntactic", "per_category", "n"):
        require(key in result, f"the result dict needs the key {key!r}")
    require(int(result["n"]) == 2, f"2 questions should be evaluated, got {result['n']}")
    for key in ("accuracy", "semantic", "syntactic"):
        require(abs(float(result[key]) - 1.0) < 1e-12,
                f"these questions are answered perfectly, so {key} must be 1.0, "
                f"got {result[key]}")
    require(result["per_category"]["syntactic verb form"]["correct"] == 1,
            "per_category must report the correct/total counts of each category")

    with_oov = questions + [("nowhere", "greece", "rome", "italy",
                             "semantic capital city")]
    oov_result = ex.evaluate_analogies(W, word2idx, words, with_oov)
    require(int(oov_result["n"]) == 2,
            "questions with out-of-vocabulary words must be skipped, not counted")
    require(abs(float(oov_result["accuracy"]) - 1.0) < 1e-12,
            "skipping an impossible question must not change the accuracy")

    empty = ex.evaluate_analogies(W, word2idx, words, [])
    require(abs(float(empty["accuracy"])) < 1e-12,
            "with no evaluable question the accuracy must be 0.0, not an exception")
