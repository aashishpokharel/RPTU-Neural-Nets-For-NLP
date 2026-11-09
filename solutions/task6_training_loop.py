"""TASK 6 (reference solution) -- the training loop that ties everything together."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (DEFAULT_BATCH_SIZE, DEFAULT_DEVICE, DEFAULT_DIM, DEFAULT_END_LR,
                    DEFAULT_EPOCHS, DEFAULT_START_LR, DEFAULT_WINDOW)
from solutions.task2_training_pairs import SkipGramDataset, generate_skipgram_pairs
from solutions.task3_embeddings import build_model
from solutions.task5_learning_rate import (learning_rate, make_optimizer, set_learning_rate,
                                           train_step)


def train_skipgram(token_ids, vocab_size, dim=DEFAULT_DIM, window=DEFAULT_WINDOW,
                   epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
                   start_lr=DEFAULT_START_LR, end_lr=DEFAULT_END_LR,
                   device=DEFAULT_DEVICE, seed=None, verbose=False):
    """Train a Skip-gram model and return `(model, history)`.

    `history[e]` is the average log-probability of epoch e -- it must grow.  Every epoch
    draws fresh training pairs (the dynamic window makes them slightly different), and
    the learning rate decays linearly over *all* the mini-batches of the run.
    """
    if seed is not None:
        torch.manual_seed(int(seed))
    rng = np.random.default_rng(seed)

    model = build_model(vocab_size, dim, seed=seed).to(device)
    optimizer = make_optimizer(model, lr=start_lr)

    epoch_pairs = [generate_skipgram_pairs(token_ids, window=window, rng=rng)
                   for _ in range(epochs)]
    total_steps = sum(max(1, math.ceil(len(pairs) / batch_size)) for pairs in epoch_pairs) or 1

    history = []
    step = 0
    for epoch, pairs in enumerate(epoch_pairs, start=1):
        loader = DataLoader(SkipGramDataset(pairs), batch_size=batch_size, shuffle=True)
        log_likelihood_sum = 0.0
        seen = 0
        for center, context in loader:
            center, context = center.to(device), context.to(device)
            set_learning_rate(optimizer, learning_rate(step / total_steps, start_lr, end_lr))
            log_likelihood_sum += train_step(model, optimizer, center, context) * int(center.shape[0])
            seen += int(center.shape[0])
            step += 1
        average = log_likelihood_sum / seen if seen else 0.0
        history.append(average)
        if verbose:
            print(f"epoch {epoch:3d}/{epochs}  mean log p = {average:+.4f}  "
                  f"({seen} pairs, lr = {optimizer.param_groups[0]['lr']:.5f})")
    return model, history
