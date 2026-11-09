"""
grader/task5.py -- checks for TASK 5 (learning-rate schedule, optimizer and one step).

The reference maths is done here, in torch, independently of the task files: the checks
know what one SGD step on a mean-reduced log-likelihood has to do to the two tables.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from config import DEFAULT_END_LR, DEFAULT_START_LR
from grader.runner import current_exercise_module
from grader.support import close, input_table, output_table, require, to_numpy

CHECKS = []


def check(fn):
    CHECKS.append((fn.__doc__, fn))
    return fn


def exercise():
    return current_exercise_module(5)


class ToyModel(nn.Module):
    """A (V, dim) Skip-gram with fixed test weights -- independent of tasks 3 and 4."""

    def __init__(self, W_in, W_out):
        super().__init__()
        self.W_in = nn.Embedding.from_pretrained(W_in.clone(), freeze=False)
        self.W_out = nn.Embedding.from_pretrained(W_out.clone(), freeze=False)

    def forward(self, center_ids):
        return self.W_in(center_ids) @ self.W_out.weight.t()


def toy_model(seed=0, vocab=8, dim=5, zero_output=False):
    generator = torch.Generator().manual_seed(seed)
    W_in = (torch.rand(vocab, dim, generator=generator) - 0.5) / dim
    if zero_output:
        W_out = torch.zeros(vocab, dim)
    else:
        W_out = (torch.rand(vocab, dim, generator=generator) - 0.5) / dim
    return ToyModel(W_in, W_out)


def reference_log_likelihood(model, center, context):
    """Mean log p(context | center) for the current weights."""
    with torch.no_grad():
        logits = model.W_in(center) @ model.W_out.weight.t()
        log_probabilities = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        return float(log_probabilities.gather(1, context.view(-1, 1)).mean())


@check
def learning_rate_decays_linearly():
    """learning_rate: start lr -> end lr, linear in progress, clamped to [0, 1]"""
    ex = exercise()
    close(ex.learning_rate(0.0, start_lr=0.5, end_lr=1e-4), 0.5, 1e-12,
          "progress=0 must return start_lr")
    close(ex.learning_rate(1.0, start_lr=0.5, end_lr=1e-4), 1e-4, 1e-12,
          "progress=1 must return end_lr")
    close(ex.learning_rate(0.5, start_lr=0.5, end_lr=5e-3), 0.5 * (0.5 + 5e-3), 1e-12,
          "half way through training the rate must be the average of the two ends")
    values = [float(ex.learning_rate(p, 0.5, 1e-5)) for p in np.linspace(0, 1, 11)]
    require(all(a > b for a, b in zip(values, values[1:])),
            f"the learning rate must decrease monotonically, got {values}")
    close(ex.learning_rate(-0.5, 0.5, 1e-5), 0.5, 1e-12,
          "progress below 0 must be clamped to start_lr")
    close(ex.learning_rate(3.0, 0.5, 1e-5), 1e-5, 1e-12,
          "progress above 1 must be clamped to end_lr")
    close(ex.learning_rate(0.25),
          float(ex.learning_rate(0.25, DEFAULT_START_LR, DEFAULT_END_LR)), 1e-12,
          f"the defaults must be start_lr={DEFAULT_START_LR}, end_lr={DEFAULT_END_LR}")


@check
def make_optimizer_covers_both_tables():
    """make_optimizer: a torch optimizer over every parameter, with the given lr"""
    ex = exercise()
    model = toy_model(seed=1)
    optimizer = ex.make_optimizer(model, lr=0.25)
    require(isinstance(optimizer, torch.optim.Optimizer),
            f"make_optimizer must return a torch.optim.Optimizer, got "
            f"{type(optimizer).__name__}")
    tracked = {id(p) for group in optimizer.param_groups for p in group["params"]}
    expected = {id(p) for p in model.parameters()}
    require(expected <= tracked,
            "the optimizer must receive model.parameters(), otherwise one of the two "
            "tables never gets updated")
    require(any(abs(float(group["lr"]) - 0.25) < 1e-12
                for group in optimizer.param_groups),
            "the learning rate passed to make_optimizer must reach optimizer.param_groups")

    default = ex.make_optimizer(model)
    require(any(abs(float(group["lr"]) - DEFAULT_START_LR) < 1e-12
                for group in default.param_groups),
            f"make_optimizer without an explicit lr must use DEFAULT_START_LR "
            f"({DEFAULT_START_LR})")


@check
def set_learning_rate_writes_into_the_optimizer():
    """set_learning_rate: every parameter group gets the new rate"""
    ex = exercise()
    model = toy_model(seed=2)
    optimizer = ex.make_optimizer(model, lr=0.5)
    ex.set_learning_rate(optimizer, 0.0125)
    rates = [float(group["lr"]) for group in optimizer.param_groups]
    require(all(abs(rate - 0.0125) < 1e-12 for rate in rates),
            f"after set_learning_rate(optimizer, 0.0125) the groups must hold 0.0125, "
            f"got {rates}")
    ex.set_learning_rate(optimizer, 1e-4)
    require(all(abs(float(group["lr"]) - 1e-4) < 1e-12
                for group in optimizer.param_groups),
            "set_learning_rate must overwrite the previous rate")


@check
def train_step_reports_the_log_likelihood_before_the_update():
    """train_step: returns the mean log-probability of the pair it was given"""
    ex = exercise()
    model = toy_model(seed=3)
    optimizer = ex.make_optimizer(model, lr=0.0)          # no movement: only a report
    center = torch.tensor([2, 5])
    context = torch.tensor([4, 1])
    before = reference_log_likelihood(model, center, context)
    returned = ex.train_step(model, optimizer, center, context)
    require(isinstance(returned, float) or np.asarray(to_numpy(returned)).ndim == 0,
            f"train_step must return the mean log-likelihood as a number, got "
            f"{type(returned).__name__}")
    close(float(returned), before, 1e-5,
          "the value reported by train_step must be measured *before* the update")
    require(float(returned) < 0.0,
            "the mean log-probability of a batch is negative -- check the sign")


@check
def train_step_moves_both_tables():
    """train_step: one optimizer step raises the log-likelihood and hides behind lr"""
    ex = exercise()
    model = toy_model(seed=4)
    center = torch.tensor([1, 3, 6])
    context = torch.tensor([5, 2, 0])
    before_in = input_table(model).detach().clone()
    before_out = output_table(model).detach().clone()
    before = reference_log_likelihood(model, center, context)

    frozen = ex.make_optimizer(model, lr=0.0)
    ex.train_step(model, frozen, center, context)
    close(input_table(model), before_in, 1e-9,
          "with lr = 0 the tables must not move: the update has to go through "
          "optimizer.step(), not through hand-written in-place arithmetic")
    close(output_table(model), before_out, 1e-9,
          "with lr = 0 the output table must not move either")

    optimizer = ex.make_optimizer(model, lr=0.5)
    ex.train_step(model, optimizer, center, context)
    after = reference_log_likelihood(model, center, context)
    require(after > before,
            f"one step on the batch must raise the mean log-probability, but it went "
            f"{before:+.5f} -> {after:+.5f}")

    changed_in = torch.any(torch.abs(input_table(model) - before_in) > 0, dim=1)
    rows = sorted(int(index) for index in torch.nonzero(changed_in).flatten())
    require(rows == sorted(set(int(i) for i in center)),
            f"only the rows of the centre words {sorted(set(int(i) for i in center))} may "
            f"change in W_in, these did: {rows}")
    require(bool(torch.any(torch.abs(output_table(model) - before_out) > 0)),
            "W_out must change as well: the softmax pushes the label up and every other "
            "word down, so every row gets a gradient")
    require(bool(torch.all(torch.isfinite(input_table(model)))),
            "the tables must stay finite after one step -- is the sign right?")


@check
def repeated_steps_increase_the_log_likelihood():
    """train_step: looping on one batch improves J at every step"""
    ex = exercise()
    model = toy_model(seed=5)
    optimizer = ex.make_optimizer(model, lr=0.2)
    center = torch.tensor([3, 3, 7])
    context = torch.tensor([1, 6, 2])
    values = []
    for _ in range(40):
        values.append(float(ex.train_step(model, optimizer, center, context)))
    require(all(np.isfinite(values)),
            "J must stay finite during training (NaN means the step size or the sign "
            "of the gradient is wrong)")
    worsened = [i for i, (a, b) in enumerate(zip(values, values[1:])) if b < a - 1e-4]
    require(not worsened,
            "J must increase after every gradient step, but it dropped at step(s) "
            f"{worsened[:5]} of 40")
    require(values[-1] > values[0] + 0.5,
            f"40 steps of size 0.2 should clearly improve J, "
            f"got {values[0]:+.4f} -> {values[-1]:+.4f}")
