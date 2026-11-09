#!/usr/bin/env python3
"""
run_word2vec.py -- the end-to-end script of the exercise (PyTorch).

It uses your eight task files and nothing else:

    python run_word2vec.py                 # 20k-token subset, a few seconds
    python run_word2vec.py --full --plot   # whole corpus + outputs/word_vectors_pca.png
    python run_word2vec.py --compare       # compare your output with solutions/

Useful knobs: --dim, --window, --epochs, --batch-size, --limit, --min-count, --device,
--seed, --out.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402  (after sys.path so that a local run behaves like -m)

from config import (CORPUS_PATH, DEFAULT_BATCH_SIZE, DEFAULT_DEVICE, DEFAULT_DIM,
                    DEFAULT_END_LR, DEFAULT_EPOCHS, DEFAULT_MIN_COUNT, DEFAULT_START_LR,
                    DEFAULT_WINDOW, EMBEDDINGS_PATH, OUTPUT_DIR, PCA_WORDS, PLOT_PATH,
                    QUESTIONS_PATH, QUICK_EPOCHS, QUICK_LIMIT)  # noqa: E402
from task1_vocabulary import build_vocab, encode, read_tokens  # noqa: E402
from task6_training_loop import train_skipgram  # noqa: E402
from task7_similarity import analogy, evaluate_analogies, load_questions, most_similar  # noqa: E402

NEIGHBOUR_WORDS = ("king", "france", "walk", "biggest")
ANALOGIES = [("man", "king", "woman"), ("france", "paris", "germany"),
             ("walk", "walking", "swim")]


def build_parser(description="train word vectors on data/corpus.txt"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--corpus", default=CORPUS_PATH)
    parser.add_argument("--questions", default=QUESTIONS_PATH)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--epochs", type=int, default=QUICK_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT)
    parser.add_argument("--limit", type=int, default=QUICK_LIMIT,
                        help="use only the first N tokens of the corpus")
    parser.add_argument("--full", action="store_true", help="use the whole corpus")
    parser.add_argument("--start-lr", type=float, default=DEFAULT_START_LR)
    parser.add_argument("--end-lr", type=float, default=DEFAULT_END_LR)
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="cpu, cuda or mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=EMBEDDINGS_PATH)
    parser.add_argument("--plot", action="store_true", help="write the PCA figure")
    parser.add_argument("--compare", action="store_true",
                        help="also train the reference solution and print both reports")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    limit = None if args.full else args.limit
    epochs = max(args.epochs, DEFAULT_EPOCHS) if args.full else args.epochs

    tokens = read_tokens(args.corpus, limit=limit)
    _counts, word2idx, idx2word = build_vocab(tokens, min_count=args.min_count)
    ids = encode(tokens, word2idx)
    scope = "whole file" if limit is None else f"first {limit:,} tokens"
    print(f"torch     : {torch.__version__} on {args.device}")
    print(f"corpus    : {len(tokens)} tokens ({scope})")
    print(f"vocabulary: {len(idx2word)} words with count >= {args.min_count}, "
          f"{len(ids)} tokens kept")

    started = time.time()
    model, history = train_skipgram(ids, len(idx2word), dim=args.dim, window=args.window,
                                    epochs=epochs, batch_size=args.batch_size,
                                    start_lr=args.start_lr, end_lr=args.end_lr,
                                    device=args.device, seed=args.seed, verbose=True)
    print(f"training  : {time.time() - started:.1f}s, dim={args.dim}, "
          f"window={args.window}, epochs={epochs}, batch_size={args.batch_size}, "
          f"lr {args.start_lr} -> {args.end_lr}")
    print("mean log-likelihood per epoch: "
          + ", ".join(f"{float(h):+.3f}" for h in history))

    vectors = model.W_in.weight.detach().cpu()
    return _report(model, vectors, word2idx, idx2word, args, ids, epochs)


def _report(model, vectors, word2idx, idx2word, args, ids, epochs):
    print("\nnearest neighbours (cosine similarity):")
    for word in NEIGHBOUR_WORDS:
        if word in word2idx:
            best = ", ".join(f"{w} ({s:.2f})"
                             for w, s in most_similar(vectors, word2idx, idx2word, word, 6))
            print(f"  {word:>9}: {best}")

    print("\nanalogies (a is to b as c is to ?):")
    for a, b, c in ANALOGIES:
        if all(word in word2idx for word in (a, b, c)):
            guesses = ", ".join(f"{w} ({s:.2f})"
                                for w, s in analogy(vectors, word2idx, idx2word, a, b, c, 3))
            print(f"  {a} -> {b} :: {c} -> {guesses}")

    result = None
    if os.path.exists(args.questions):
        result = evaluate_analogies(vectors, word2idx, idx2word,
                                    load_questions(args.questions))
        print(f"\nanalogy questions ([P1] Sec. 4, {result['n']} answerable):")
        print(f"  overall   {result['accuracy']:6.1%}")
        print(f"  semantic  {result['semantic']:6.1%}")
        print(f"  syntactic {result['syntactic']:6.1%}")
        for category, stats in sorted(result["per_category"].items()):
            print(f"    {category:<28} {stats['accuracy']:>6.1%} "
                  f"({stats['correct']}/{stats['total']})")

    _export(model, vectors, idx2word, word2idx, args)
    if args.compare:
        _compare(model, vectors, word2idx, idx2word, args, ids, epochs)
    return 0


def _export(model, vectors, idx2word, word2idx, args):
    """Save the vectors / checkpoint / figure -- task 8 is a bonus, so failures are soft."""
    from task8_io_and_pca import (plot_pca, save_checkpoint, save_embeddings,
                                  word_vectors)  # noqa: WPS433 (deliberately lazy)

    output_dir = os.path.dirname(os.path.abspath(args.out))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    try:
        save_embeddings(args.out, word_vectors(model), idx2word)
        print(f"\nvectors written to {args.out}")
    except NotImplementedError:
        print("\n(task 8 is not implemented yet -- the vectors were not saved)")
        return
    checkpoint = os.path.splitext(args.out)[0] + ".pt"
    save_checkpoint(model, checkpoint)
    print(f"checkpoint written to {checkpoint} (torch.save of the state dict)")

    if args.plot:
        words = [word for word in PCA_WORDS if word in word2idx]
        if len(words) >= 4:
            coords = vectors[[word2idx[word] for word in words]]
            plot_pca(coords, words, PLOT_PATH)
            print(f"PCA figure written to {PLOT_PATH}")
        else:
            print("not enough known words for the PCA figure -- check PCA_WORDS "
                  "in config.py")


def _compare(model, vectors, word2idx, idx2word, args, ids, epochs):
    """Train the reference solution with the same settings and print both reports."""
    print("\n" + "=" * 78)
    print("reference solution (solutions/) with identical settings")
    print("=" * 78)
    from solutions.task6_training_loop import train_skipgram as reference_train
    from solutions.task7_similarity import evaluate_analogies as reference_evaluate
    from solutions.task7_similarity import load_questions as reference_load

    reference_model, reference_history = reference_train(
        ids, len(idx2word), dim=args.dim, window=args.window, epochs=epochs,
        batch_size=args.batch_size, start_lr=args.start_lr, end_lr=args.end_lr,
        device=args.device, seed=args.seed, verbose=False)
    print("mean log-likelihood per epoch: "
          + ", ".join(f"{float(h):+.3f}" for h in reference_history))
    if os.path.exists(args.questions):
        questions = reference_load(args.questions)
        mine = evaluate_analogies(vectors, word2idx, idx2word, questions)
        theirs = reference_evaluate(reference_model.W_in.weight.detach().cpu(),
                                    word2idx, idx2word, questions)
        print(f"analogy accuracy: yours {mine['accuracy']:.1%} "
              f"| reference {theirs['accuracy']:.1%}")
        print("(a small difference is normal: the two runs draw their own random windows)")


if __name__ == "__main__":
    raise SystemExit(main())
