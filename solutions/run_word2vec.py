"""End-to-end run of the REFERENCE solution (PyTorch).

    python -m solutions.run_word2vec                      # quick subset, a few seconds
    python -m solutions.run_word2vec --full --plot        # whole corpus + PCA figure
    python -m solutions.run_word2vec --compare            # this solution vs the stub run

Use it to see the target behaviour of the exercise you are writing in the task files at
the root of this folder.
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from config import (CORPUS_PATH, DEFAULT_BATCH_SIZE, DEFAULT_DEVICE, DEFAULT_DIM,
                    DEFAULT_END_LR, DEFAULT_EPOCHS, DEFAULT_MIN_COUNT, DEFAULT_START_LR,
                    DEFAULT_WINDOW, EMBEDDINGS_PATH, OUTPUT_DIR, PCA_WORDS, PLOT_PATH,
                    QUESTIONS_PATH, QUICK_EPOCHS, QUICK_LIMIT)
from solutions.task1_vocabulary import build_vocab, encode, read_tokens
from solutions.task6_training_loop import train_skipgram
from solutions.task7_similarity import evaluate_analogies, load_questions, most_similar
from solutions.task8_io_and_pca import (plot_pca, save_checkpoint, save_embeddings,
                                        word_vectors)


def build_parser(description="word2vec reference run (PyTorch)"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--corpus", default=CORPUS_PATH)
    parser.add_argument("--questions", default=QUESTIONS_PATH)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--epochs", type=int, default=QUICK_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT)
    parser.add_argument("--limit", type=int, default=QUICK_LIMIT,
                        help="use only the first N tokens (--full for the whole corpus)")
    parser.add_argument("--full", action="store_true", help="use the whole corpus")
    parser.add_argument("--start-lr", type=float, default=DEFAULT_START_LR)
    parser.add_argument("--end-lr", type=float, default=DEFAULT_END_LR)
    parser.add_argument("--device", default=DEFAULT_DEVICE,
                        help="cpu, cuda or mps (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=EMBEDDINGS_PATH)
    parser.add_argument("--plot", action="store_true", help="also write the PCA figure")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    limit = None if args.full else args.limit
    if args.full:
        args.epochs = max(args.epochs, DEFAULT_EPOCHS)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return _run(args, limit)


def _run(args, limit):
    tokens = read_tokens(args.corpus, limit=limit)
    _counts, word2idx, idx2word = build_vocab(tokens, min_count=args.min_count)
    ids = encode(tokens, word2idx)
    print(f"torch {torch.__version__} on {args.device}")
    print(f"corpus: {len(tokens)} tokens | vocabulary: {len(idx2word)} words "
          f"(min_count={args.min_count}) | in-vocabulary tokens: {len(ids)}")

    started = time.time()
    model, history = train_skipgram(ids, len(idx2word), dim=args.dim, window=args.window,
                                    epochs=args.epochs, batch_size=args.batch_size,
                                    start_lr=args.start_lr, end_lr=args.end_lr,
                                    device=args.device, seed=args.seed, verbose=True)
    print(f"training: {time.time() - started:.1f}s, dim={args.dim}, window={args.window}, "
          f"epochs={args.epochs}, batch_size={args.batch_size}")
    print("mean log-likelihood per epoch: " + ", ".join(f"{h:+.3f}" for h in history))

    vectors = word_vectors(model)
    _report(args, vectors, word2idx, idx2word, model)
    return 0


def _report(args, vectors, word2idx, idx2word, model=None):
    print("\nnearest neighbours (cosine similarity):")
    for word in ("king", "france", "big", "walk"):
        if word in word2idx:
            neighbours = ", ".join(f"{w} ({s:.2f})"
                                   for w, s in most_similar(vectors, word2idx, idx2word,
                                                            word, 6))
            print(f"  {word:>8}: {neighbours}")

    if os.path.exists(args.questions):
        questions = load_questions(args.questions)
        result = evaluate_analogies(vectors, word2idx, idx2word, questions)
        print(f"\nanalogy task ([P1] Sec. 4, 3CosAdd), {result['n']} evaluable questions:")
        print(f"  overall   {result['accuracy']:.1%}")
        print(f"  semantic  {result['semantic']:.1%}")
        print(f"  syntactic {result['syntactic']:.1%}")
        for category, stats in sorted(result["per_category"].items()):
            print(f"    {category:<30} {stats['accuracy']:>6.1%} "
                  f"({stats['correct']}/{stats['total']})")
    else:
        result = None

    save_embeddings(args.out, vectors, idx2word)
    print(f"\nvectors written to {args.out}")
    if model is not None and hasattr(model, "state_dict"):
        checkpoint = os.path.splitext(args.out)[0] + ".pt"
        save_checkpoint(model, checkpoint)
        print(f"checkpoint (torch state_dict) written to {checkpoint}")

    if args.plot:
        words = [word for word in idx2word if word in set(PCA_WORDS)]
        if len(words) >= 4:
            coords = vectors[[word2idx[word] for word in words]]
            plot_pca(coords, words, PLOT_PATH)
            print(f"PCA figure written to {PLOT_PATH}")
        else:
            print("not enough known words for the PCA figure -- check PCA_WORDS in config.py")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
