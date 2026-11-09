"""
TASK 3 -- the model: two embedding matrices inside an nn.Module.

    [P1] Sec. 3.1, Figure 1

The network has a linear hidden layer and a softmax output layer, but with PyTorch it is
one line of code: both layers are embedding tables.

    W_in   shape (V, dim)   row i = v_{w_i}:   the "input vector"
    W_out  shape (V, dim)   row i = v'_{w_i}:  the "output vector"

For a batch of centre words the hidden layer is just a lookup

    h = W_in(center_ids)                     # (N, dim)

and the output layer scores every vocabulary word with a dot product

    u = h @ W_out.weight.T                   # (N, V),  u[n, w] = v'_{w} . h_n

softmax(u) (task 4) turns the scores of row n into the predicted distribution over the
context words of the n-th pair.

Because the hidden layer is linear, `W_in.weight` *is* the matrix of word vectors you will
plot at the end: that is the whole trick of word2vec.

[P1] Sec. 4.2 initialises the input vectors uniformly in [-0.5/dim, +0.5/dim] and the
output vectors at zero.  Scaling by 1/dim keeps the initial dot products small, which
matters: with large random vectors the softmax would start almost one-hot and the first
gradient steps would be enormous.

Run this file on its own to check yourself:

    python task3_embeddings.py

Hints
-----
* `nn.Embedding(V, dim)` gives you a table of shape (V, dim) whose `forward(ids)`
  returns `weight[ids]` -- exactly the lookup, and autograd connects it to `weight`.
* `nn.init.uniform_(tensor, -limit, +limit)` and `nn.init.zeros_(tensor)` write in place.
* `torch.manual_seed(seed)` makes the random initialisation reproducible: the grader
  builds the model twice with the same seed and compares.
* Everything you write in this task must live in `nn.Module`/`nn.Embedding` objects, so
  that `model.parameters()` finds the weights and `.to(device)` can move them.
"""

from __future__ import annotations

import torch
from torch import nn

from config import DEFAULT_DIM


class SkipGram(nn.Module):
    """The two-layer Skip-gram network of [P1] Figure 1.

    Attributes
    ----------
    W_in : nn.Embedding, shape (V, dim)
        The table you look the centre word up in.
    W_out : nn.Embedding, shape (V, dim)
        The table you compute the scores with.
    """

    def __init__(self, vocab_size, dim=DEFAULT_DIM):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        self.W_in = torch.nn.Embedding(vocab_size, dim)

        
        self.W_out = torch.nn.Embedding(vocab_size, dim)
        
        # raise NotImplementedError("Task 3: SkipGram.__init__()")

    def reset_parameters(self):
        """[P1] Sec. 4.2: v_w ~ U(-0.5/dim, +0.5/dim), v'_w = 0."""
        torch.nn.init.uniform_(
                            self.W_in.weight, 
                            a=  -0.5 / self.dim, 
                            b=  0.5 / self.dim
                        )
                
        torch.nn.init.zeros_(self.W_out.weight)
                
    def forward(self, center_ids):
        """Return the scores u of shape (N, V) for a batch of centre-word ids.
   
        Parameters
        ----------
        center_ids : torch.Tensor, shape (N,), dtype torch.long

        Returns
        -------
        torch.Tensor, shape (N, V), float
            Row n holds one score per vocabulary word: u[n, w] = v'_w . v_{center_ids[n]}.
        """

        v = self.W_in(center_ids)
        v_out = self.W_out.weight
        scores = v @ v_out.T

        return scores
        # raise NotImplementedError("Task 3: SkipGram.forward()")


def build_model(vocab_size, dim=DEFAULT_DIM, seed=None):
    """Create a SkipGram model, optionally with a reproducible seed.

    Parameters
    ----------
    vocab_size : int
        V, the number of words in the vocabulary (task 1).
    dim : int
        D, the size of a word vector (config.DEFAULT_DIM).
    seed : int or None
        Seed for `torch.manual_seed`; None means "different every run".

    Returns
    -------
    SkipGram
        A freshly initialised model, ready for `model.parameters()`.
    """
    if seed is not None:
        torch.manual_seed(seed)
    model = SkipGram(vocab_size= vocab_size, dim=dim)
    model.reset_parameters()

    return model
    # raise NotImplementedError("Task 3: build_model()")


def hidden_vector(model, center_ids):
    """Return h, the hidden layer of the network, shape (N, dim).

    This is a one-liner (`model.W_in(center_ids)`), but it is worth writing down: the
    hidden layer of word2vec is *not* computed, it is a row of the input table.  Because
    the lookup is a differentiable operation, the gradient of the loss with respect to h
    lands in `model.W_in.weight.grad` without any extra work from you.
    """

    return model.W_in(center_ids)

    # raise NotImplementedError("Task 3: hidden_vector()")


if __name__ == "__main__":
    import sys

    from grader.runner import main as _grade
    from grader.runner import set_module_override

    set_module_override(3, sys.modules[__name__])
    raise SystemExit(_grade(tasks=[3]))
