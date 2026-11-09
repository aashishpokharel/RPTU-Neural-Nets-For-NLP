"""
grader/task4.py -- checks for TASK 4 (the full softmax and the Skip-gram loss).

Two things are verified: the *objective* (the log-probability of the context word, as a
mean over the batch) and the *calculus* (the gradients that autograd produces must be the
true derivative of that objective, checked against central finite differences).  How you
write the loss is free -- `torch.logsumexp`, `F.log_softmax` and `F.cross_entropy` all
pass, as long as the number and its derivative are right.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from grader.runner import current_exercise_module
from grader.support import (close, finite_difference_gradients, log_softmax, require,
                            softmax as ref_softmax, to_numpy, to_tensor)

CHECKS = []


def check(fn):
    CHECKS.append((fn.__doc__, fn))
    return fn


def exercise():
    return current_exercise_module(4)


class ReferenceSkipGram(nn.Module):
    """A minimal (V, dim) Skip-gram with frozen test weights -- independent of task 3."""

    def __init__(self, W_in, W_out):
        super().__init__()
        self.W_in = nn.Embedding.from_pretrained(W_in.clone(), freeze=False)
        self.W_out = nn.Embedding.from_pretrained(W_out.clone(), freeze=False)

    def forward(self, center_ids):
        return self.W_in(center_ids) @ self.W_out.weight.t()


def problem(seed=0, vocab=7, dim=5, n_pairs=6):
    """A small, non-degenerate scoring problem: (model, centre ids, context ids)."""
    generator = torch.Generator().manual_seed(seed)
    W_in = torch.randn(vocab, dim, generator=generator) * 0.4
    W_out = torch.randn(vocab, dim, generator=generator) * 0.7
    center = torch.randint(0, vocab, (n_pairs,), generator=generator)
    context = torch.randint(0, vocab, (n_pairs,), generator=generator)
    return ReferenceSkipGram(W_in, W_out), center, context


def reference_log_probabilities(W_in, W_out, center):
    logits = W_in[center] @ W_out.t()
    return logits - torch.logsumexp(logits, dim=-1, keepdim=True)


@check
def softmax_is_a_distribution():
    """softmax: positive, sums to 1 along the last dimension, survives huge logits"""
    ex = exercise()
    logits = torch.tensor([1.0, 2.0, 3.0])
    out = ex.softmax(logits)
    require(torch.is_tensor(out),
            f"softmax must return a tensor, got {type(out).__name__}")
    require(tuple(out.shape) == (3,), f"softmax has shape {tuple(out.shape)}, expected (3,)")
    close(out, ref_softmax([1.0, 2.0, 3.0]), 1e-6,
          "softmax(.) does not match exp(x_i)/sum(exp(x_j))")
    require(bool(torch.all(out > 0)), "every probability must be strictly positive")
    close(out.sum(), 1.0, 1e-6, "the probabilities must sum to 1")

    big = ex.softmax(torch.tensor([1000.0, 1001.0, 999.0]))
    require(bool(torch.all(torch.isfinite(big))),
            "softmax overflowed for large logits -- subtract the maximum (or use "
            "torch.softmax / torch.logsumexp)")
    close(big, ref_softmax([1000.0, 1001.0, 999.0]), 1e-6,
          "softmax must be shift invariant: softmax(x) == softmax(x - max(x))")
    close(ex.softmax(torch.tensor([6.0, 7.0, 8.0])), out, 1e-6,
          "softmax(x + c) must equal softmax(x)")

    batched = ex.softmax(torch.tensor([[0.0, 1.0], [2.0, -2.0]]))
    require(tuple(batched.shape) == (2, 2),
            f"a batch of logits must stay a batch, got {tuple(batched.shape)}")
    close(batched.sum(dim=-1), torch.ones(2), 1e-6,
          "every row of a batch must be a distribution (softmax over the last dim)")


@check
def log_probabilities_is_the_stable_log_softmax():
    """log_probabilities: log softmax, finite for large logits, one row per pair"""
    ex = exercise()
    logits = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]])
    got = ex.log_probabilities(logits)
    require(tuple(got.shape) == (2, 3),
            f"log_probabilities has shape {tuple(got.shape)}, expected (2, 3)")
    close(got, torch.log_softmax(logits, dim=-1), 1e-6,
          "log_probabilities must be log softmax(u) row by row")
    close(torch.exp(got).sum(dim=-1), torch.ones(2), 1e-6,
            "exp of the log-probabilities must sum to 1")
    huge_logits = torch.tensor([[1000.0, 1001.0, 999.0]], dtype=torch.float64)
    huge = ex.log_probabilities(huge_logits)
    require(bool(torch.all(torch.isfinite(huge))),
            "log_probabilities must stay finite for large logits (use logsumexp)")
    close(huge, torch.log_softmax(huge_logits, dim=-1), 1e-6,
          "log_probabilities must be shift invariant")


@check
def loss_is_the_negative_log_probability_of_the_label():
    """skipgram_loss: -mean log p(context | center), and the sum/none reductions"""
    ex = exercise()
    model, center, context = problem(seed=1)
    W_in, W_out = model.W_in.weight.detach(), model.W_out.weight.detach()
    per_pair = -reference_log_probabilities(W_in, W_out, center).gather(
        1, context.view(-1, 1)).squeeze(1)

    mean_loss = ex.skipgram_loss(model, center, context)
    require(torch.is_tensor(mean_loss) and mean_loss.ndim == 0,
            "skipgram_loss(reduction='mean') must return a scalar tensor, got "
            f"{type(mean_loss).__name__} of shape {tuple(getattr(mean_loss, 'shape', ()))}")
    close(mean_loss, per_pair.mean(), 1e-5,
          "the loss must be the average of -log p(context | center) over the batch")
    close(ex.skipgram_loss(model, center, context, reduction="sum"), per_pair.sum(), 1e-5,
          "reduction='sum' must return the *sum* of the per-pair losses")
    none = ex.skipgram_loss(model, center, context, reduction="none")
    close(none, per_pair, 1e-5, "reduction='none' must return one loss per pair")

    require(float(mean_loss.detach()) > 0.0,
            "the loss must be positive: it is minus a log-probability")
    require(abs(float(mean_loss.detach()) - float(torch.log(torch.tensor(7.0)))) > 1e-6,
            "the loss must depend on the weights, not be a constant")


@check
def loss_gradients_match_finite_differences():
    """skipgram_loss: the gradients autograd produces match central differences"""
    ex = exercise()
    model, center, context = problem(seed=2)
    model = model.double()
    W_in, W_out = model.W_in.weight, model.W_out.weight

    def total_loss():
        return ex.skipgram_loss(model, center, context, reduction="sum")

    loss = total_loss()
    gradients = torch.autograd.grad(loss, [W_in, W_out])
    finite_difference_gradients(total_loss, [W_in, W_out], gradients, name="skipgram_loss")

    # the label must be pushed up and every other word pushed down
    logits = W_in[center] @ W_out.t()
    probabilities = torch.softmax(logits.detach(), dim=-1)
    require(float(probabilities.gather(1, context.view(-1, 1)).mean()) < 1.0,
            "the softmax of the scores must not be a one-hot distribution")


@check
def mean_log_likelihood_is_the_number_to_report():
    """mean_log_likelihood: J = -loss as a plain float, and it builds no graph"""
    ex = exercise()
    model, center, context = problem(seed=3)
    value = ex.mean_log_likelihood(model, center, context)
    require(isinstance(value, float) or (torch.is_tensor(value) and value.ndim == 0)
            or np.asarray(to_numpy(value)).ndim == 0,
            f"mean_log_likelihood must return a plain number, got {type(value).__name__}")
    reference = -float(ex.skipgram_loss(model, center, context).detach())
    close(float(value), reference, 1e-5,
          "J must be the average log-probability of the batch, i.e. minus the loss")
    require(float(value) < 0.0,
            "a log-probability of a whole vocabulary is negative -- check the sign")

    model.zero_grad(set_to_none=True)
    ex.mean_log_likelihood(model, center, context)
    require(model.W_in.weight.grad is None and model.W_out.weight.grad is None,
            "mean_log_likelihood is only a *report*: wrap it in torch.no_grad() (and "
            "return float(...)) so that it accumulates no gradients and no graph")
