"""
grader/task8.py -- checks for TASK 8 (bonus: exporting vectors, checkpoints, PCA figure).
"""

from __future__ import annotations

import os
import tempfile

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
    return current_exercise_module(8)


class ToyModel(nn.Module):
    """Same attribute names as the exercise model, used only to test save/load."""

    def __init__(self, W_in, W_out):
        super().__init__()
        self.W_in = nn.Embedding.from_pretrained(W_in.clone(), freeze=False)
        self.W_out = nn.Embedding.from_pretrained(W_out.clone(), freeze=False)

    def forward(self, center_ids):
        return self.W_in(center_ids) @ self.W_out.weight.t()


def toy_embeddings():
    idx2word = ["the", "cat", "dog"]
    W = torch.tensor([[0.25, -1.5, 0.0], [1.0, 2.0, -3.0], [-0.125, 0.5, 0.75]])
    word2idx = {word: index for index, word in enumerate(idx2word)}
    return W, word2idx, idx2word


@check
def save_and_load_round_trip():
    """save_embeddings + load_embeddings: the vectors survive a round trip"""
    ex = exercise()
    W, word2idx, idx2word = toy_embeddings()
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "vectors.txt")
    try:
        ex.save_embeddings(path, W, idx2word)
        require(os.path.exists(path), f"save_embeddings did not create {path}")
        with open(path, encoding="utf-8") as fh:
            first = fh.readline().split()
        require(len(first) == 2 and [int(x) for x in first] == [3, 3],
                f'the first line must be "<n_words> <dim>", got {" ".join(first)!r}')

        loaded = ex.load_embeddings(path)
        require(isinstance(loaded, tuple) and len(loaded) == 3,
                "load_embeddings must return (vectors, word2idx, idx2word)")
        W_back, word2idx_back, idx2word_back = loaded
        require(list(idx2word_back) == idx2word,
                f"the word order changed: {list(idx2word_back)}")
        require(dict(word2idx_back) == word2idx, "the loaded word2idx is wrong")
        require(torch.is_tensor(W_back),
                f"load_embeddings should return a torch tensor, got {type(W_back).__name__}")
        close(W_back, W, 1e-5, "the loaded vectors must be (almost) identical to the "
                               "saved ones")
    finally:
        for name in os.listdir(directory):
            os.remove(os.path.join(directory, name))
        os.rmdir(directory)


@check
def load_embeddings_reads_a_real_file():
    """load_embeddings: parses a hand written vector file"""
    ex = exercise()
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "hand.txt")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("2 3\n")
            fh.write("alpha 1.0 2.0 3.0\n")
            fh.write("beta -0.5 0.25 4\n")
        W, word2idx, idx2word = ex.load_embeddings(path)
        W = to_numpy(W).astype(np.float64)
        require(W.shape == (2, 3), f"loaded matrix has shape {W.shape}, expected (2, 3)")
        close(W, np.array([[1.0, 2.0, 3.0], [-0.5, 0.25, 4.0]]), 1e-6,
              "load_embeddings mis-read the numbers")
        require(list(idx2word) == ["alpha", "beta"], f"word order is {list(idx2word)}")
        require(dict(word2idx) == {"alpha": 0, "beta": 1}, "word2idx is wrong")
    finally:
        for name in os.listdir(directory):
            os.remove(os.path.join(directory, name))
        os.rmdir(directory)


@check
def checkpoint_round_trip():
    """save_checkpoint + load_checkpoint: a state_dict survives torch.save/torch.load"""
    ex = exercise()
    generator = torch.Generator().manual_seed(0)
    model = ToyModel(torch.randn(6, 3, generator=generator) * 0.1,
                     torch.randn(6, 3, generator=generator) * 0.1)
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "model.pt")
    try:
        ex.save_checkpoint(model, path)
        require(os.path.exists(path), f"save_checkpoint did not create {path}")
        require(os.path.getsize(path) > 0, "the checkpoint file is empty")
        restored = ex.load_checkpoint(path, 6, 3)
        require(isinstance(restored, nn.Module),
                f"load_checkpoint must return a model, got {type(restored).__name__}")
        close(input_table(restored), input_table(model), 1e-6,
              "the restored input table must hold the saved weights")
        close(output_table(restored), output_table(model), 1e-6,
              "the restored output table must hold the saved weights")
        ids = torch.tensor([0, 5])
        close(restored(ids), model(ids), 1e-6,
              "the restored model must compute the same scores")
    finally:
        for name in os.listdir(directory):
            os.remove(os.path.join(directory, name))
        os.rmdir(directory)


@check
def pca_projects_on_the_main_axes():
    """pca_2d: (N, 2) coordinates, centred, PC1 explains more variance than PC2"""
    ex = exercise()
    generator = torch.Generator().manual_seed(0)
    scale = torch.tensor([5.0, 2.0, 0.5, 0.1])
    X = torch.randn(200, 4, generator=generator, dtype=torch.float64) * scale \
        + torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float64)
    xy = to_numpy(ex.pca_2d(X)).astype(np.float64)
    require(xy.shape == (200, 2), f"pca_2d has shape {xy.shape}, expected (200, 2)")
    require(np.all(np.isfinite(xy)), "pca_2d produced NaN/inf")
    require(np.max(np.abs(xy.mean(axis=0))) < 1e-8,
            "PCA coordinates must be centred on zero -- subtract the mean first")

    eigenvalues = np.sort(np.linalg.eigvalsh(np.cov(to_numpy(X).T)))[::-1]
    ratio = float(np.var(xy[:, 0]) / np.var(xy[:, 1]))
    expected = float(eigenvalues[0] / eigenvalues[1])
    require(abs(ratio / expected - 1.0) < 0.05,
            f"the first component must capture the largest variance (variance ratio "
            f"{ratio:.2f}, expected {expected:.2f})")

    collinear = torch.outer(torch.linspace(-3.0, 5.0, 40, dtype=torch.float64),
                            torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64))
    flat = to_numpy(ex.pca_2d(collinear)).astype(np.float64)
    require(flat.shape == (40, 2), f"pca_2d has shape {flat.shape}, expected (40, 2)")
    require(np.max(np.abs(flat[:, 1])) < 1e-6,
            "data that lives on a line must collapse onto the first component")


@check
def plot_pca_writes_a_figure():
    """plot_pca: writes a PNG file (skipped when matplotlib is missing)"""
    try:
        import matplotlib                                     # noqa: F401
    except ImportError:                                       # pragma: no cover
        print("        (matplotlib is not installed -- check skipped)")
        return
    ex = exercise()
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "pca.png")
    try:
        W, _word2idx, idx2word = toy_embeddings()
        ex.plot_pca(W, idx2word, path)
        require(os.path.exists(path), f"plot_pca did not write {path}")
        require(os.path.getsize(path) > 1000,
                "the PNG file is suspiciously small -- is the figure really drawn?")
    finally:
        for name in os.listdir(directory):
            os.remove(os.path.join(directory, name))
        os.rmdir(directory)
