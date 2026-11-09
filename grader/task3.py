"""
grader/task3.py -- checks for TASK 3 (the SkipGram module with its two embedding tables).

The checks look at the *module*: the two (V, dim) nn.Embedding tables with the
initialisation [P1] Sec. 4.2 prescribes, a lookup that reaches autograd, and a forward()
that computes the scores of the output layer.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from grader.runner import current_exercise_module
from grader.support import close, input_table, output_table, require, to_numpy

CHECKS = []


def check(fn):
    CHECKS.append((fn.__doc__, fn))
    return fn


def exercise():
    return current_exercise_module(3)


def require_table(holder, name, vocab_size, dim):
    require(holder is not None, f"the model needs a `{name}` attribute holding its table")
    weight = getattr(holder, "weight", holder)
    require(torch.is_tensor(weight), f"`{name}` must be an nn.Embedding or a tensor")
    require(tuple(weight.shape) == (vocab_size, dim),
            f"`{name}` has shape {tuple(weight.shape)}, expected ({vocab_size}, {dim})")
    require(weight.is_floating_point(),
            f"`{name}` must hold floats, got dtype {weight.dtype}")
    require(weight.requires_grad,
            f"`{name}` must be a trainable parameter (requires_grad=True), otherwise "
            "autograd cannot fill .grad and the optimizer has nothing to update")


@check
def model_is_two_trainable_embedding_tables():
    """build_model: an nn.Module with W_in and W_out of shape (V, dim)"""
    ex = exercise()
    model = ex.build_model(20, 8, seed=0)
    require(isinstance(model, nn.Module),
            f"build_model must return a torch.nn.Module, got {type(model).__name__}")
    require_table(getattr(model, "W_in", None), "W_in", 20, 8)
    require_table(getattr(model, "W_out", None), "W_out", 20, 8)

    parameters = list(model.parameters())
    tables = {id(input_table(model)), id(output_table(model))}
    require(tables <= {id(p) for p in parameters},
            "both tables must be registered parameters, so that model.parameters() "
            "hands them to the optimizer and .to(device) moves them")
    require(len(parameters) >= 2,
            f"the model should have (at least) the two tables as parameters, "
            f"it has {len(parameters)}")

    plain = ex.build_model(7, 4)
    require(tuple(plain.W_in.weight.shape) == (7, 4),
            "build_model must work without an explicit dim (default from config.py)")


@check
def initialisation_follows_the_paper():
    """build_model: W_in ~ U(-0.5/dim, 0.5/dim) and W_out starts at zero"""
    ex = exercise()
    model = ex.build_model(20, 8, seed=0)
    W_in = to_numpy(input_table(model))
    W_out = to_numpy(output_table(model))
    require(np.allclose(W_out, 0.0), "W_out must start at zero (u_w = 0 for every word)")
    limit = 0.5 / 8
    require(np.max(np.abs(W_in)) <= limit + 1e-6,
            f"W_in entries must stay inside +/-0.5/dim = +/-{limit:.4f}, "
            f"got {np.max(np.abs(W_in)):.4f}")
    require(np.max(np.abs(W_in)) > 0.3 * limit,
            f"W_in looks degenerate: it does not use the +/-{limit:.4f} range")
    require(float(np.std(W_in)) > 1e-3, "W_in must be random, not constant")


@check
def seeds_reproduce_the_initialisation():
    """build_model: the same seed gives the same start, a new seed does not"""
    ex = exercise()
    a = ex.build_model(12, 5, seed=3)
    b = ex.build_model(12, 5, seed=3)
    c = ex.build_model(12, 5, seed=4)
    require(torch.equal(input_table(a), input_table(b)),
            "the same seed must produce exactly the same W_in")
    require(torch.equal(output_table(a), output_table(b)),
            "the same seed must produce exactly the same W_out")
    require(not torch.equal(input_table(a), input_table(c)),
            "a different seed must produce a different W_in")


@check
def hidden_vector_is_the_input_row():
    """hidden_vector: the lookup of the input table, shape (N, dim), reachable by autograd"""
    ex = exercise()
    model = ex.build_model(10, 6, seed=1)
    ids = torch.tensor([0, 4, 9, 4])
    hidden = ex.hidden_vector(model, ids)
    require(torch.is_tensor(hidden),
            f"hidden_vector must return a tensor, got {type(hidden).__name__}")
    require(tuple(hidden.shape) == (4, 6),
            f"hidden_vector has shape {tuple(hidden.shape)}, expected (4, 6)")
    close(hidden, input_table(model)[ids], 1e-6,
          "hidden_vector must return rows of the *input* table W_in")

    model.zero_grad(set_to_none=True)
    hidden.sum().backward()
    grad = input_table(model).grad
    require(grad is not None and torch.count_nonzero(grad) > 0,
            "no gradient reached W_in: build the hidden layer with the embedding table "
            "(model.W_in(ids)), not with a detached copy")


@check
def forward_computes_the_scores():
    """forward: (N, V) logits equal to h @ W_out.T, differentiable w.r.t. both tables"""
    ex = exercise()
    model = ex.build_model(9, 4, seed=2)
    with torch.no_grad():
        input_table(model).copy_(torch.linspace(-0.05, 0.05, 36).reshape(9, 4))
        output_table(model).copy_(torch.linspace(-0.3, 0.4, 36).reshape(9, 4))
    ids = torch.tensor([0, 3, 8, 3, 1])
    logits = model(ids)
    require(torch.is_tensor(logits),
            f"forward must return a tensor, got {type(logits).__name__}")
    require(tuple(logits.shape) == (5, 9),
            f"forward has shape {tuple(logits.shape)}, expected (5, 9) = (N, V)")
    with torch.no_grad():
        expected = input_table(model)[ids] @ output_table(model).t()
    close(logits, expected, 1e-6,
          "forward must return one score per vocabulary word: u = W_out @ h")

    model.zero_grad(set_to_none=True)
    logits.sum().backward()
    for name, holder in (("W_in", input_table(model)), ("W_out", output_table(model))):
        require(holder.grad is not None and torch.count_nonzero(holder.grad) > 0,
                f"no gradient reached {name} through forward -- are the scores computed "
                "with the output table?")
