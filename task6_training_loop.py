"""
TASK 6 -- the training loop: DataLoader, epochs, decaying learning rate.

Put the pieces together:

    for epoch in 1..epochs:
        pairs   = generate_skipgram_pairs(token_ids, window, rng)   # task 2
        dataset = SkipGramDataset(pairs)                           # task 2
        loader  = DataLoader(dataset, batch_size, shuffle=True)    # torch
        for center, context in loader:
            set_learning_rate(optimizer, learning_rate(step / total_steps))  # task 5
            train_step(model, optimizer, center, context)          # task 5
        history.append(mean log-probability of the epoch)

Three details that decide whether the vectors are any good:

* draw the pairs again every epoch -- the dynamic window of task 2 makes each pass
  slightly different, which is a free kind of data augmentation;
* decay the learning rate over *all* the batches of the run, not per epoch, so the last
  batch of the last epoch really has the end learning rate;
* report the mean log-probability (a negative number that must go up), not the loss.

Run this file on its own to check yourself:

    python task6_training_loop.py
"""

from __future__ import annotations

import numpy as np
import torch

from config import (DEFAULT_BATCH_SIZE, DEFAULT_DEVICE, DEFAULT_DIM, DEFAULT_END_LR,
                    DEFAULT_EPOCHS, DEFAULT_START_LR, DEFAULT_WINDOW)
from task2_training_pairs import SkipGramDataset, generate_skipgram_pairs
from task3_embeddings import build_model
from task5_learning_rate import (learning_rate, make_optimizer, set_learning_rate,
                                 train_step)


def train_skipgram(token_ids, vocab_size, dim=DEFAULT_DIM, window=DEFAULT_WINDOW,
                   epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
                   start_lr=DEFAULT_START_LR, end_lr=DEFAULT_END_LR,
                   device=DEFAULT_DEVICE, seed=None, verbose=False):
    """Train a Skip-gram model and return `(model, history)`.

    Parameters
    ----------
    token_ids : list[int]
        The encoded corpus: the ids of the in-vocabulary tokens, in order (task 1).
    vocab_size : int
        V, the size of the vocabulary.
    dim, window, epochs, batch_size, start_lr, end_lr, device : see config.py
    seed : int or None
        Must make the whole run reproducible: seed `torch.manual_seed(seed)` for the
        weight initialisation *and* `np.random.default_rng(seed)` for the window draws.
    verbose : bool
        Print one line per epoch.

    Returns
    -------
    model : SkipGram
        The trained model; its word vectors are `model.W_in.weight`.
    history : list[float]
        `history[e]` is the average log-probability of epoch e (a negative number that
        must increase), i.e. the objective J of [P1] averaged over the epoch.

    Hints
    -----
    * `torch.utils.data.DataLoader(dataset, batch_size=..., shuffle=True)` collates the
      (center, context) tensors into two (N,) long tensors.
    * move the batch with `center.to(device)` before the forward pass; the model must be
      on the same device (`build_model(...).to(device)`).
    * the total number of optimizer steps is `sum(ceil(len(pairs) / batch_size))` over all
      epochs -- compute it once, before the loop, so `progress` runs from 0 to 1.
    * weighting the per-batch J by the batch size keeps the epoch average correct when the
      last batch is smaller than the others.
    """
    # raise NotImplementedError("Task 6: train_skipgram()")
    
    if seed is not None:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    model = build_model(vocab_size=vocab_size, dim=dim).to(device=device)
    optimizer = make_optimizer(model, start_lr)


    epoch_pairs = [generate_skipgram_pairs(token_ids=token_ids, window=window, rng=rng) for _ in range (epochs)]
    total_steps = sum(
                    (len(pairs)+ batch_size-1) // batch_size
                    for pairs in epoch_pairs
                )
    if total_steps <=0:
        raise ValueError(" No Skip Gram Pairs Found")
    step = 0
    
    history = []
    for epoch, pairs in enumerate(epoch_pairs, start=1):
        pairs   = generate_skipgram_pairs(token_ids, window, rng)   # task 2
        dataset = SkipGramDataset(pairs)                           # task 2
        loader  = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)    # torch
        epoch_log_likelihood = 0.0
        epoch_examples = 0
        for center, context in loader:
            center = center.to(device)
            context = context.to(device)
            progress = (step+1)/total_steps
            lr = learning_rate(progress=progress)
            set_learning_rate (optimizer=optimizer, lr=lr)

            # Single Batch
            batch_j = train_step(model=model, optimizer=optimizer, center_ids= center, context_ids= context)

            actual_batch_size = center.size(0)

            epoch_log_likelihood += (batch_j * actual_batch_size)

            epoch_examples+= actual_batch_size

            step += 1
        epoch_j = epoch_log_likelihood / epoch_examples

        history.append(epoch_j)

    if verbose:
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Mean log-likelihood: {epoch_j:.4f} | "
            f"Learning rate: {lr:.6f}"
        )
    return model, history

if __name__ == "__main__":
    import sys

    from grader.runner import main as _grade
    from grader.runner import set_module_override

    set_module_override(6, sys.modules[__name__])
    raise SystemExit(_grade(tasks=[6]))
