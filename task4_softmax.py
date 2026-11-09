"""
TASK 4 -- the full softmax and the Skip-gram loss.

    [P1] Sec. 3.1 and Sec. 4, first objective (the "full softmax")

For a pair (centre word I, context word O) the network predicts a distribution over the
whole vocabulary

    p(w | I) = softmax(u)[w] = exp(u_w) / sum_j exp(u_j)

and the objective this exercise maximises is the log-probability of the *correct* word

    J = log p(O | I)

so the loss to minimise is `-J`.  With a mini-batch of N pairs, both the loss and J are
averages over the batch (this is what `reduction="mean"` does).

Two numerical details, which the grader checks:

* subtract the maximum before exponentiating (or use `torch.logsumexp`): with scores
  around 1000, `exp` overflows to inf and the whole run becomes NaN;
* pick the label with `gather`, not with a Python loop over the batch.

Why this cost is expensive: the denominator sums over all V words, so one pair costs
O(V * dim) -- that is the reason [P2] introduces negative sampling.  Here V is small.

Run this file on its own to check yourself:

    python task4_softmax.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from config import DEFAULT_DIM


def softmax(logits):
    """Softmax over the last dimension, shape (..., V) -> (..., V).

    Parameters
    ----------
    logits : torch.Tensor
        Scores u, any shape whose last dimension is the vocabulary.

    Returns
    -------
    torch.Tensor
        Probabilities of the same shape: strictly positive and summing to 1 along the
        last dimension.

    Hints: `shifted = logits - logits.amax(dim=-1, keepdim=True)` makes the computation
    stable; `torch.softmax(logits, dim=-1)` is the one-line version (accepted too, but you
    are here to see what it does).
    """
    return torch.softmax(logits, dim= -1)
    # raise NotImplementedError("Task 4: softmax()")


def log_probabilities(logits):
    """log softmax(logits) in one numerically stable step, same shape as `logits`.

    Hint: `logits - torch.logsumexp(logits, dim=-1, keepdim=True)`.  Do not write
    `softmax(logits).log()` -- it returns -inf as soon as a probability underflows to 0.
    """
    return logits - torch.logsumexp(logits, dim = -1, keepdim= True)
    # raise NotImplementedError("Task 4: log_probabilities()")


def skipgram_loss(model, center_ids, context_ids, reduction="mean"):
    """The Skip-gram loss of one batch: -mean log p(context | center).

    Parameters
    ----------
    model : SkipGram
        A model whose forward pass returns the scores (N, V) -- your task 3 model.
    center_ids, context_ids : torch.Tensor, shape (N,), dtype torch.long
    reduction : {"mean", "sum", "none"}
        "mean" is the default (one number per batch); "none" returns one loss per pair,
        shape (N,).

    Returns
    -------
    torch.Tensor
        A scalar (or (N,) for reduction="none") that autograd can differentiate.

    Hints
    -----
    * `logits = model(center_ids)` has shape (N, V); the log-probability of the label of
      row n is `log_probabilities(logits)[n, context_ids[n]]`.
    * `log_probs.gather(1, context_ids.view(-1, 1)).squeeze(1)` selects those N numbers
      without a Python loop.
    * `F.cross_entropy(logits, context_ids)` computes exactly this loss -- compare your
      result with it once, then keep whichever you prefer.
    """
    logits = model(center_ids)
    log_probs = log_probabilities(logits=logits)
    selected_log_probs = log_probs.gather(
        1, context_ids.view(-1,1)
    ).squeeze(1)

    loss = -selected_log_probs

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

    # raise NotImplementedError("Task 4: skipgram_loss()")


def mean_log_likelihood(model, center_ids, context_ids):
    """Return J = -loss as a plain Python float, for reporting in the training loop.

    This function must not build a graph: wrap the computation in `torch.no_grad()` and
    return `float(...)`, otherwise every logging call keeps a batch of activations alive
    (and quietly doubles the memory use of a big training run).
    """
    # raise NotImplementedError("Task 4: mean_log_likelihood()")
    with torch.no_grad():
        return float (
            -skipgram_loss(model, center_ids=center_ids, context_ids= context_ids)
        )


if __name__ == "__main__":
    import sys

    from grader.runner import main as _grade
    from grader.runner import set_module_override

    set_module_override(4, sys.modules[__name__])
    raise SystemExit(_grade(tasks=[4]))
