"""TASK 5 (reference solution) -- the learning-rate schedule and one optimizer step."""

from __future__ import annotations

import torch

from config import DEFAULT_END_LR, DEFAULT_START_LR
from solutions.task4_softmax import skipgram_loss


def learning_rate(progress, start_lr=DEFAULT_START_LR, end_lr=DEFAULT_END_LR):
    progress = float(min(max(progress, 0.0), 1.0))
    return start_lr + (end_lr - start_lr) * progress


def make_optimizer(model, lr=DEFAULT_START_LR):
    """Plain SGD over every parameter of the model (both embedding tables)."""
    return torch.optim.SGD(model.parameters(), lr=lr)


def set_learning_rate(optimizer, lr):
    """Write a new learning rate into every parameter group."""
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def train_step(model, optimizer, center_ids, context_ids):
    """One forward pass, one backward pass, one optimizer step.

    Returns the *mean log-likelihood* of the batch, measured before the update, so that
    the training loop can average it into the epoch history.
    """
    optimizer.zero_grad(set_to_none=True)
    loss = skipgram_loss(model, center_ids, context_ids)
    loss.backward()
    optimizer.step()
    return -float(loss.detach())
