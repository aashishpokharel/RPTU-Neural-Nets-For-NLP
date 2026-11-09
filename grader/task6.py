"""
grader/task6.py -- checks for TASK 6 (the Skip-gram training loop).

The last check measures what the vectors are good for: with the defaults of `config.py` a
correct loop reaches far more than 15% accuracy on the analogy questions of
`data/questions.txt`, while untrained vectors score ~1%.
"""

from __future__ import annotations

import numpy as np

from grader.runner import current_exercise_module
from grader.support import (analogy_accuracy, build_vocab as ref_vocab, corpus,
                            input_table, load_questions, output_table, quality_model,
                            require, to_numpy)

CHECKS = []


def check(fn):
    CHECKS.append((fn.__doc__, fn))
    return fn


def exercise():
    return current_exercise_module(6)


def tiny_problem(n_tokens=1500, dim=10, epochs=2):
    tokens = corpus(n_tokens)
    _counts, word2idx, idx2word = ref_vocab(tokens, min_count=3)
    ids = [word2idx[token] for token in tokens if token in word2idx]
    return ids, len(idx2word), dim, epochs


def train(ids, vocab_size, dim, epochs, seed=0, **kwargs):
    out = exercise().train_skipgram(ids, vocab_size, dim=dim, epochs=epochs, seed=seed,
                                    **kwargs)
    require(isinstance(out, tuple) and len(out) == 2,
            "train_skipgram must return the pair (model, history) -- the trained nn.Module "
            "and the list of per-epoch mean log-likelihoods")
    model, history = out
    require(isinstance(history, (list, tuple)),
            f"history must be a list of numbers, got {type(history).__name__}")
    return (model, to_numpy(input_table(model)).astype(np.float64),
            to_numpy(output_table(model)).astype(np.float64), [float(v) for v in history])


@check
def training_produces_a_model_and_a_history():
    """train_skipgram: (V, dim) matrices, finite, and one log-likelihood per epoch"""
    ids, vocab_size, dim, epochs = tiny_problem()
    model, W_in, W_out, history = train(ids, vocab_size, dim, epochs)
    require(W_in.shape == (vocab_size, dim),
            f"W_in has shape {W_in.shape}, expected ({vocab_size}, {dim})")
    require(W_out.shape == (vocab_size, dim),
            f"W_out has shape {W_out.shape}, expected ({vocab_size}, {dim})")
    require(np.all(np.isfinite(W_in)), "W_in contains NaN/inf -- is the loss exploding?")
    require(np.all(np.isfinite(W_out)), "W_out contains NaN/inf -- is the loss exploding?")
    require(float(np.std(W_in)) > 1e-4, "W_in did not move: nothing was trained")
    require(float(np.std(W_out)) > 0.0,
            "W_out is still all zeros: the output table never got a gradient")
    require(len(history) == epochs,
            f"the history must have one entry per epoch, got {len(history)} "
            f"for {epochs} epochs")
    require(all(np.isfinite(history)), "the history contains NaN/inf")
    require(history[-1] < 0.0,
            "the reported value must be the mean log-probability, i.e. negative")
    require(bool(model.W_in.weight.requires_grad),
            "the returned model must still be the trainable one")


@check
def loss_improves_while_training():
    """train_skipgram: the average log-probability grows from epoch to epoch"""
    tokens = corpus(4000)
    _counts, word2idx, idx2word = ref_vocab(tokens, min_count=3)
    ids = [word2idx[token] for token in tokens if token in word2idx]
    _model, _W_in, _W_out, history = train(ids, len(idx2word), 20, 3)
    require(history[-1] > history[0],
            f"the loss must improve, but it went {history[0]:+.4f} -> {history[-1]:+.4f}")
    require(all(value < 0.0 for value in history),
            "every entry of the history must be a (negative) log-probability")
    uniform = -float(np.log(len(idx2word)))
    require(history[-1] > uniform + 0.05,
            f"the mean log-probability ({history[-1]:+.4f}) is no better than the uniform "
            f"baseline ({uniform:+.4f}): the loop is not learning")


@check
def training_is_repeatable():
    """train_skipgram: the same seed reproduces the run, a new seed does not"""
    ids, vocab_size, dim, epochs = tiny_problem(n_tokens=1000, dim=6, epochs=1)
    _m, first, _o, _h = train(ids, vocab_size, dim, epochs, seed=1)
    _m, again, _o, _h = train(ids, vocab_size, dim, epochs, seed=1)
    _m, other, _o, _h = train(ids, vocab_size, dim, epochs, seed=2)
    require(np.allclose(first, again),
            "the same seed must reproduce the same embeddings (seed both the RNG that "
            "draws the windows and torch.manual_seed for the initialisation)")
    require(not np.allclose(first, other),
            "a different seed must produce different embeddings")


@check
def hyper_parameters_are_respected():
    """train_skipgram: dim, epochs and batch_size really change the run"""
    ids, vocab_size, _dim, _epochs = tiny_problem(n_tokens=1200)
    _model, W_in, _W_out, history = train(ids, vocab_size, 8, 2)
    require(W_in.shape == (vocab_size, 8),
            f"train_skipgram(dim=8) must give (V, 8) matrices, got {W_in.shape}")
    require(len(history) == 2,
            f"train_skipgram(epochs=2) must report 2 values, got {len(history)}")
    _m, _W, _O, other_history = train(ids, vocab_size, 8, 2, batch_size=32)
    require(len(other_history) == 2,
            "the batch size must not change the number of reported epochs")
    require(np.all(np.isfinite(other_history)),
            "a different batch size must still give a finite history")


@check
def trained_vectors_capture_meaning():
    """train_skipgram: the vectors solve analogy questions (>= 15% accuracy)"""
    W_in, history, word2idx, idx2word = quality_model()
    questions = load_questions()
    accuracy = analogy_accuracy(W_in, word2idx, idx2word, questions)
    require(accuracy >= 0.15,
            f"only {accuracy:.1%} of the analogy questions are answered correctly "
            f"(chance is ~{1.0 / len(idx2word):.1%}). Train on more of the corpus / more "
            "epochs, and check that the mean log-probability in `history` really improves")
    require(history[-1] > history[0],
            "the embeddings are trained but the reported loss did not improve")
