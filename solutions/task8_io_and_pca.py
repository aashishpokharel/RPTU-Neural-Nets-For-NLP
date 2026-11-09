"""TASK 8 (reference solution) -- exporting the vectors, checkpoints and the PCA figure."""

from __future__ import annotations

import os

import torch

from solutions.task3_embeddings import build_model


def word_vectors(model):
    """The vectors you actually use: W_in, detached from autograd and moved to the CPU."""
    return model.W_in.weight.detach().cpu()


def save_embeddings(path, vectors, idx2word, precision=6):
    """Write the word2vec text format: '<n_words> <dim>' then 'word x1 x2 ... x_dim'."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    table = torch.as_tensor(vectors).detach().cpu()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{table.shape[0]} {table.shape[1]}\n")
        for word, row in zip(idx2word, table.tolist()):
            fh.write(word + " " + " ".join(f"{value:.{precision}f}" for value in row) + "\n")
    return path


def load_embeddings(path):
    """Read a vector file back: (vectors float32 (V, dim), word2idx, idx2word)."""
    rows, words = [], []
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().split()
        if len(header) < 2:
            raise ValueError(f"{path}: the first line must be '<n_words> <dim>'")
        dim = int(header[1])
        for line in fh:
            parts = line.split()
            if len(parts) != dim + 1:
                continue
            words.append(parts[0])
            rows.append([float(value) for value in parts[1:]])
    table = torch.tensor(rows, dtype=torch.float32).reshape(len(rows), dim)
    word2idx = {word: index for index, word in enumerate(words)}
    return table, word2idx, words


def save_checkpoint(model, path):
    """torch.save of the state dict -- the weights without the class definition."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_checkpoint(path, vocab_size, dim):
    """Rebuild a model and fill it with the saved weights."""
    model = build_model(vocab_size, dim)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model


def pca_2d(vectors):
    """Project the rows of `vectors` on their two principal axes, shape (N, 2).

    Centring + SVD of the mean-centred matrix: the right singular vectors are the
    principal directions, so X @ Vt[:2].T are the coordinates on the first two.
    """
    table = torch.as_tensor(vectors).detach()
    centred = table - table.mean(dim=0, keepdim=True)
    _u, _s, vt = torch.linalg.svd(centred, full_matrices=False)
    return centred @ vt[:2].t()


def plot_pca(vectors, labels, out_path=None, annotate=True):
    """Scatter the two PCA coordinates and write them to `out_path` (a PNG)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = pca_2d(vectors).cpu().numpy()
    figure, axes = plt.subplots(figsize=(9, 7))
    axes.scatter(coords[:, 0], coords[:, 1], s=18, alpha=0.75)
    if annotate:
        for (x, y), word in zip(coords, labels):
            axes.annotate(str(word), (x, y), fontsize=7, alpha=0.8)
    axes.set_title("word vectors projected on the first two principal components")
    axes.set_xlabel("PC 1")
    axes.set_ylabel("PC 2")
    figure.tight_layout()
    if out_path:
        directory = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(directory, exist_ok=True)
        figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return coords
