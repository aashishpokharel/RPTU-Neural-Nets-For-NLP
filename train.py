#!/usr/bin/env python3
"""
train.py -- train word vectors and write the artefacts that `inference.py` loads.

    python train.py                         # 20k-token subset, 5 epochs, a few seconds
    python train.py --full --plot           # whole corpus, then the PCA figure
    python train.py --dim 60 --epochs 10 --name vectors_d60

Three files appear in the output directory (`--out-dir`, default `outputs/`, which
`.gitignore` already excludes):

    <name>.txt    the vectors in the word2vec text format of task 8
    <name>.pt     the state_dict checkpoint of task 8 (W_in *and* W_out)
    <name>.json   metadata: hyper-parameters, vocabulary, history and scores

The metadata carries the vocabulary on purpose: a `state_dict` is only a pile of numbers,
so without the words there would be no way to tell which row is "king".  It also stores
the history and the analogy scores, so two runs can be compared by diffing their JSON.

This file adds no new mathematics -- tasks 1-8 do the work:

    read_tokens / build_vocab / encode   task 1     text -> ids
    train_skipgram                       task 6     pairs, DataLoader, schedule
    evaluate_analogies / load_questions  task 7     the [P1] Sec. 4 report
    save_embeddings / save_checkpoint    task 8     the two on-disk formats
    word_vectors / plot_pca              task 8     W_in, and the PCA figure

Hyper-parameters and the linear decay are those of [P1] Sec. 4.2; config.py says why each
default is what it is.

Reference: [P1] Mikolov, Chen, Corrado, Dean (2013), "Efficient Estimation of Word
Representations in Vector Space", arXiv:1301.3781.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402  (after sys.path so that a local run behaves like -m)

from config import (CORPUS_PATH, DEFAULT_BATCH_SIZE, DEFAULT_DEVICE, DEFAULT_DIM,
                    DEFAULT_END_LR, DEFAULT_EPOCHS, DEFAULT_MIN_COUNT, DEFAULT_START_LR,
                    DEFAULT_WINDOW, OUTPUT_DIR, PCA_WORDS, QUESTIONS_PATH, QUICK_EPOCHS,
                    QUICK_LIMIT)  # noqa: E402
from task1_vocabulary import build_vocab, encode, read_tokens  # noqa: E402
from task6_training_loop import train_skipgram  # noqa: E402
from task7_similarity import evaluate_analogies, load_questions  # noqa: E402
from task8_io_and_pca import (plot_pca, save_checkpoint, save_embeddings,
                              word_vectors)  # noqa: E402

#: Stem of the default artefact set: outputs/word_vectors.{txt,pt,json}.
DEFAULT_NAME = "word_vectors"


def build_parser(description="train word vectors and write <out-dir>/<name>.{txt,pt,json}"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--corpus", default=CORPUS_PATH, help="text file, one sentence per line")
    parser.add_argument("--questions", default=QUESTIONS_PATH,
                        help="analogy questions for the report (task 7 format)")
    parser.add_argument("--out-dir", default=OUTPUT_DIR, help="where the artefacts are written")
    parser.add_argument("--name", default=DEFAULT_NAME, help="file name stem of the artefacts")
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM, help="D, the size of a vector")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help="C, the maximum distance of a context word")
    parser.add_argument("--epochs", type=int, default=QUICK_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT,
                        help="drop words rarer than this from the vocabulary")
    parser.add_argument("--limit", type=int, default=QUICK_LIMIT,
                        help="use only the first N tokens of the corpus")
    parser.add_argument("--full", action="store_true", help="use the whole corpus")
    parser.add_argument("--start-lr", type=float, default=DEFAULT_START_LR)
    parser.add_argument("--end-lr", type=float, default=DEFAULT_END_LR)
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="cpu, cuda or mps")
    parser.add_argument("--seed", type=int, default=42,
                        help="seeds the weights and the dynamic windows")
    parser.add_argument("--plot", action="store_true", help="write the PCA figure (task 8)")
    parser.add_argument("--no-eval", action="store_true",
                        help="skip the analogy report (faster on a large corpus)")
    return parser


def artefact_paths(out_dir, name):
    """The four paths of one artefact set, all sharing the `<out-dir>/<name>` stem."""
    stem = os.path.join(os.path.abspath(out_dir), name)
    return {"stem": stem,
            "vectors": stem + ".txt",
            "checkpoint": stem + ".pt",
            "metadata": stem + ".json",
            "plot": stem + "_pca.png"}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.dim < 1:
        parser.error("--dim must be at least 1")
    if args.limit < 0:
        parser.error("--limit must not be negative")

    # --full means "every token" and at least the reference number of epochs, exactly as in
    # run_word2vec.py, so that the quick run and the full run stay comparable.
    limit = None if args.full else args.limit
    epochs = max(args.epochs, DEFAULT_EPOCHS) if args.full else args.epochs
    paths = artefact_paths(args.out_dir, args.name)

    tokens = read_tokens(args.corpus, limit=limit)
    counts, word2idx, idx2word = build_vocab(tokens, min_count=args.min_count)
    ids = encode(tokens, word2idx)
    if limit is None or len(tokens) < limit:
        scope = "whole file"
    else:
        scope = f"first {limit:,} tokens"
    print(f"torch     : {torch.__version__} on {args.device}")
    print(f"corpus    : {len(tokens)} tokens ({scope}) from {os.path.abspath(args.corpus)}")
    print(f"vocabulary: {len(idx2word)} words with count >= {args.min_count}, "
          f"{len(ids)} tokens kept, {int(counts.sum())} in-vocabulary occurrences")
    if len(idx2word) < 2:
        parser.error(f"the vocabulary has {len(idx2word)} word(s) -- lower --min-count "
                     "or use more of the corpus")

    started = time.time()
    model, history = train_skipgram(ids, len(idx2word), dim=args.dim, window=args.window,
                                    epochs=epochs, batch_size=args.batch_size,
                                    start_lr=args.start_lr, end_lr=args.end_lr,
                                    device=args.device, seed=args.seed, verbose=False)
    seconds = time.time() - started
    print(f"training  : {seconds:.1f}s, dim={args.dim}, window={args.window}, "
          f"epochs={epochs}, batch_size={args.batch_size}, "
          f"lr {args.start_lr} -> {args.end_lr}")
    print("mean log-likelihood per epoch: "
          + ", ".join(f"{float(value):+.3f}" for value in history))

    vectors = word_vectors(model)
    report = None if args.no_eval else _evaluate(args, vectors, word2idx, idx2word)
    plot_file = _plot(args, vectors, word2idx) if args.plot else None

    save_embeddings(paths["vectors"], vectors, idx2word)
    save_checkpoint(model, paths["checkpoint"])
    metadata = _metadata(args, epochs, tokens, counts, idx2word, ids, history, seconds,
                         report, paths, plot_file)
    _write_json(paths["metadata"], metadata)

    here = os.path.dirname(os.path.abspath(__file__))
    print("\nartefacts:")
    for key in ("vectors", "checkpoint", "metadata", "plot"):
        if key == "plot" and plot_file is None:
            continue
        print(f"  {key:<10} {_display(paths[key], here)}")
    print(f"\nnext      : python inference.py similar king "
          f"--model {_display(paths['stem'], here)}")
    return 0


def _display(path, base):
    """A path relative to `base` when that stays inside it, the absolute path otherwise."""
    relative = os.path.relpath(path, base)
    return path if relative.startswith(os.pardir) else relative


def _evaluate(args, vectors, word2idx, idx2word):
    """Print the [P1] Sec. 4 analogy report; returns the report for the metadata."""
    if not os.path.exists(args.questions):
        print(f"\nno question file at {args.questions} -- skipping the analogy report")
        return None
    result = evaluate_analogies(vectors, word2idx, idx2word,
                                load_questions(args.questions))
    print(f"\nanalogy questions ([P1] Sec. 4, {result['n']} answerable):")
    print(f"  overall   {result['accuracy']:6.1%}")
    print(f"  semantic  {result['semantic']:6.1%}")
    print(f"  syntactic {result['syntactic']:6.1%}")
    for category, stats in sorted(result["per_category"].items()):
        print(f"    {category:<28} {stats['accuracy']:>6.1%} "
              f"({stats['correct']}/{stats['total']})")
    return result


def _plot(args, vectors, word2idx):
    """The PCA figure of task 8; returns its path, or None when the words are missing."""
    words = [word for word in PCA_WORDS if word in word2idx]
    if len(words) < 4:
        print("\nnot enough known words for the PCA figure -- check PCA_WORDS in config.py")
        return None
    coords = vectors[[word2idx[word] for word in words]]
    path = artefact_paths(args.out_dir, args.name)["plot"]
    plot_pca(coords, words, path)
    return path


def _metadata(args, epochs, tokens, counts, idx2word, ids, history, seconds, report,
              paths, plot_file):
    """Everything `inference.py` needs to rebuild and describe this model."""
    return {
        "script": "train.py",
        "created_at": datetime.datetime.now(datetime.timezone.utc)
                                        .isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "corpus": os.path.abspath(args.corpus),
        "limit": None if args.full else args.limit,
        "tokens": len(tokens),
        "tokens_in_vocabulary": int(len(ids)),
        "in_vocabulary_occurrences": int(counts.sum()),
        "vocab_size": len(idx2word),
        "dim": args.dim,
        "window": args.window,
        "min_count": args.min_count,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "start_lr": args.start_lr,
        "end_lr": args.end_lr,
        "device": args.device,
        "seed": args.seed,
        "train_seconds": round(seconds, 3),
        "mean_log_likelihood": [float(value) for value in history],
        "analogy": report,
        "vocabulary": list(idx2word),
        "files": {
            "vectors": os.path.basename(paths["vectors"]),
            "checkpoint": os.path.basename(paths["checkpoint"]),
            "metadata": os.path.basename(paths["metadata"]),
            "plot": os.path.basename(plot_file) if plot_file else None,
        },
    }


def _write_json(path, payload):
    """Write `payload` as indented JSON, creating the parent directory if needed."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
