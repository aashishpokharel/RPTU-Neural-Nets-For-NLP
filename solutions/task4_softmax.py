"""TASK 4 (reference solution) -- the full softmax and the Skip-gram loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def softmax(logits):
    """softmax over the last dimension, shifted by the maximum for stability."""
    shifted = logits - logits.amax(dim=-1, keepdim=True)
    exp = shifted.exp()
    return exp / exp.sum(dim=-1, keepdim=True)


def log_probabilities(logits):
    """log softmax(logits), i.e. logits - logsumexp(logits), computed in one stable step."""
    return logits - torch.logsumexp(logits, dim=-1, keepdim=True)


def skipgram_loss(model, center_ids, context_ids, reduction="mean"):
    """-log p(context | center), averaged over the batch (the loss to minimise)."""
    logits = model(center_ids)                                  # (N, V)
    log_probs = log_probabilities(logits)                        # (N, V)
    label_log_prob = log_probs.gather(1, context_ids.view(-1, 1)).squeeze(1)   # (N,)
    if reduction == "mean":
        return -label_log_prob.mean()
    if reduction == "sum":
        return -label_log_prob.sum()
    if reduction == "none":
        return -label_log_prob
    raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")


def mean_log_likelihood(model, center_ids, context_ids):
    """J, the number the training loop reports: the average log-probability of the batch.

    `mean_log_likelihood == -skipgram_loss(..., reduction="mean")`; it is the quantity
    [P1] maximises, so it must go *up* during training.
    """
    with torch.no_grad():
        return float(-skipgram_loss(model, center_ids, context_ids, reduction="mean"))


def cross_entropy_check(model, center_ids, context_ids):
    """The same loss through PyTorch's own op -- they must agree to the last bit."""
    return F.cross_entropy(model(center_ids), context_ids)
