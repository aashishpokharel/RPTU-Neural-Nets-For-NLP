"""
TASK 5 -- the learning-rate schedule, the optimizer and one training step.

    [P1] Sec. 4.2: "we chose starting learning rate 0.025 and decreased it linearly, so
         that it approaches zero at the end of the last training epoch"

The three pieces of a PyTorch training step, in the order they happen:

    optimizer.zero_grad()      # 1. forget yesterday's gradients
    loss = skipgram_loss(...)  # 2. forward pass + loss
    loss.backward()            # 3. autograd fills .grad of every parameter
    optimizer.step()           # 4. move the parameters downhill

The decay matters more than it looks: early in training the vectors have to travel a long
way, while late in training a big step would overshoot and the softmax would become almost
one-hot for whatever pair was seen last.

Note on the scale of the learning rate: our loss is the *mean* over a mini-batch, so its
gradient is `batch_size` times smaller than the per-pair gradient of [P1].  The default
`DEFAULT_START_LR = 0.5` with `DEFAULT_BATCH_SIZE = 64` is therefore the equivalent of
[P1]'s 0.025 for one pair at a time (0.5 / 64 ~ 0.008 -- same ballpark).

Run this file on its own to check yourself:

    python task5_learning_rate.py
"""

from __future__ import annotations

import torch

from config import DEFAULT_END_LR, DEFAULT_START_LR
from task4_softmax import skipgram_loss


def learning_rate(progress, start_lr=DEFAULT_START_LR, end_lr=DEFAULT_END_LR):
    """Linear interpolation between `start_lr` and `end_lr`.

    Parameters
    ----------
    progress : float
        Fraction of the training run that has been done: 0.0 = nothing, 1.0 = the last
        step.  Values outside [0, 1] must be clamped, so that floating point noise in
        `step / total_steps` can never produce a negative learning rate.
    """
    progress = max(0,min(1, progress))

    return start_lr * (1-progress) + end_lr * (progress)


    # raise NotImplementedError("Task 5: learning_rate()")


def make_optimizer(model, lr=DEFAULT_START_LR):
    """Create the optimizer that will update the two embedding tables.

    Parameters
    ----------
    model : SkipGram
        Your task 3 model.  Pass `model.parameters()`: an optimizer built from anything
        else will silently leave one of the tables frozen.
    lr : float
        Starting learning rate.

    Returns
    -------
    torch.optim.Optimizer
        `torch.optim.SGD(model.parameters(), lr=lr)` is the faithful choice: [P1] uses
        plain SGD (mini-batch, no momentum, no adaptive step).
    """
    return torch.optim.SGD(model.parameters(), lr=lr)
    # raise NotImplementedError("Task 5: make_optimizer()")


def set_learning_rate(optimizer, lr):
    """Write `lr` into every parameter group of the optimizer (the decay schedule).

    Hint: `for group in optimizer.param_groups: group["lr"] = lr`.  Building a new
    optimizer would throw away the state (momentum, step counters), so change the number
    in place.
    """
    for group in optimizer.param_groups:
        group['lr'] = lr

    # raise NotImplementedError("Task 5: set_learning_rate()")


def train_step(model, optimizer, center_ids, context_ids):
    """One zero_grad / forward / backward / step cycle.

    Parameters
    ----------
    model : SkipGram
    optimizer : torch.optim.Optimizer
    center_ids, context_ids : torch.Tensor, shape (N,), dtype torch.long

    Returns
    -------
    float
        The mean log-probability J of this batch, measured *before* the update, so the
        caller can average it into the epoch history.

    Hints
    -----
    * `optimizer.zero_grad(set_to_none=True)` is the modern spelling; without it the
      gradients of the previous batch add up and the run diverges.
    * the update itself must come from `optimizer.step()` -- do not write
      `weight -= lr * weight.grad` by hand, or the learning-rate decay of task 6 will be
      ignored (the grader checks this: with lr = 0 nothing may move).
    """

    optimizer.zero_grad(set_to_none = True)
    logits = model(center_ids)
    loss = skipgram_loss(model=model, center_ids=center_ids, context_ids=context_ids)

    loss.backward()
    optimizer.step()

    return -loss.item()
    # raise NotImplementedError("Task 5: train_step()")


if __name__ == "__main__":
    import sys

    from grader.runner import main as _grade
    from grader.runner import set_module_override

    set_module_override(5, sys.modules[__name__])
    raise SystemExit(_grade(tasks=[5]))
