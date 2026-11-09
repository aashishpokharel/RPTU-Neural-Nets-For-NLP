"""
TASK 8 (bonus) -- exporting the vectors, saving a checkpoint and the PCA figure.

Two different ways to keep a trained model, and one way to look at it.

1. The *text* format of the original word2vec (and of GloVe / fastText):

        3 3
        the 0.250000 -1.500000 0.000000
        cat 1.000000 2.000000 -3.000000
        dog -0.125000 0.500000 0.750000

   The first line is "<n_words> <dim>", then one line per word.  It is portable (no
   Python needed to read it) and it is what `gensim`'s KeyedVectors loads.

2. A PyTorch *checkpoint*: `torch.save(model.state_dict(), path)` stores the weights
   without the class definition, and `model.load_state_dict(torch.load(path))` puts them
   back into a freshly built model.  Use this to continue training later; use the text
   format to share the vectors.

3. PCA: keep the two directions of largest variance (the first two right singular
   vectors of the mean-centred matrix) and scatter the words on them.  Word vectors live
   in 40 dimensions, but the semantic groups are usually visible in 2.

Run this file on its own to check yourself (matplotlib is needed for the figure):

    python task8_io_and_pca.py
"""

from __future__ import annotations

import os

import torch

from task3_embeddings import build_model


def word_vectors(model):
    """Return the (V, dim) matrix of word vectors, detached from autograd, on the CPU.

    Hint: `model.W_in.weight.detach().cpu()`.  Forgetting `detach()` keeps a reference to
    the autograd graph alive; forgetting `.cpu()` breaks on a GPU run.
    """
    return model.W_in.weight.detach().cpu()


def save_embeddings(path, vectors, idx2word, precision=6):
    """Write `vectors` (V, dim) with the words of `idx2word` in the word2vec text format.

    The first line is "<n_words> <dim>"; every following line is the word and its
    coordinates separated by spaces.  Create the parent directory if it does not exist --
    the driver script writes into `outputs/`, which is not in the repository.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    table = torch.as_tensor(vectors).detach().cpu()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{table.shape[0]} {table.shape[1]}\n")
        for word, row in zip(idx2word, table.tolist()):
            coordinates = " ".join(f"{value:.{precision}f}" for value in row)
            fh.write(f"{word} {coordinates}\n")
    return path


def load_embeddings(path):
    """Read a vector file back.

    Returns
    -------
    vectors : torch.Tensor, shape (V, dim), float32
    word2idx : dict[str, int]
    idx2word : list[str]

    Hints: read the header to learn `dim`, then split every following line with
    `line.split()`; a word line holds `dim + 1` fields.  Build the tensor with
    `torch.tensor(rows, dtype=torch.float32)`.
    """
    words, rows = [], []
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().split()
        if len(header) < 2:
            raise ValueError(f"{path}: the first line must be '<n_words> <dim>'")
        dim = int(header[1])
        for line in fh:
            parts = line.split()
            if len(parts) != dim + 1:
                continue                      # skip blank or malformed lines
            words.append(parts[0])
            rows.append([float(value) for value in parts[1:]])

    vectors = torch.tensor(rows, dtype=torch.float32).reshape(len(words), dim)
    word2idx = {word: index for index, word in enumerate(words)}
    return vectors, word2idx, words


def save_checkpoint(model, path):
    """`torch.save(model.state_dict(), path)` -- the weights, without the class."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_checkpoint(path, vocab_size, dim):
    """Rebuild a model with `build_model(vocab_size, dim)` and load the saved weights.

    Hint: `torch.load(path, map_location="cpu")` then `model.load_state_dict(state)`.
    `map_location` makes a checkpoint saved on a GPU readable on a laptop.
    """
    model = build_model(vocab_size, dim)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model


def pca_2d(vectors):
    """Project the rows of `vectors` (N, dim) on their two principal axes, shape (N, 2).

    Hints
    -----
    * centre the data: `centred = X - X.mean(dim=0, keepdim=True)`;
    * `u, s, vt = torch.linalg.svd(centred, full_matrices=False)` -- the rows of `vt` are
      the principal directions, in decreasing order of variance;
    * the coordinates are `centred @ vt[:2].t()`.

    The result must be centred on zero and data that lives on a line must collapse onto
    the first component (its second coordinate is then ~0).
    """
    table = torch.as_tensor(vectors).detach()
    if not table.is_floating_point():
        table = table.float()
    centred = table - table.mean(dim=0, keepdim=True)
    _u, _s, vt = torch.linalg.svd(centred, full_matrices=False)
    return centred @ vt[:2].t()


def plot_pca(vectors, labels, out_path=None, annotate=True):
    """Scatter the two PCA coordinates, label the words, write a PNG when `out_path` is set.

    Use the non-interactive backend (`matplotlib.use("Agg")`) so that the grader can call
    this on a machine without a display, and `figure.savefig(out_path, dpi=150)`.
    Return the (N, 2) coordinates.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coordinates = pca_2d(vectors).cpu().numpy()
    figure, axes = plt.subplots(figsize=(9, 7))
    axes.scatter(coordinates[:, 0], coordinates[:, 1], s=18, alpha=0.75)
    if annotate:
        for (x, y), label in zip(coordinates, labels):
            axes.annotate(str(label), (x, y), fontsize=7, alpha=0.8)
    axes.set_title("word vectors on the first two principal components")
    axes.set_xlabel("PC 1")
    axes.set_ylabel("PC 2")
    figure.tight_layout()

    if out_path:
        directory = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(directory, exist_ok=True)
        figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return coordinates


if __name__ == "__main__":
    import sys

    from grader.runner import main as _grade
    from grader.runner import set_module_override

    set_module_override(8, sys.modules[__name__])
    raise SystemExit(_grade(tasks=[8]))
