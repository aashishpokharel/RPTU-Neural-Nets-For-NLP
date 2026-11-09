#!/usr/bin/env python3
"""
inference.py -- query a word-vector model trained by `train.py`.  Nothing here trains.

    python inference.py similar king              # 8 nearest neighbours
    python inference.py similar france -n 10
    python inference.py analogy man king woman    # man -> king :: woman -> ?
    python inference.py similarity king queen      # cosine similarity of two words
    python inference.py evaluate                  # accuracy on data/questions.txt
    python inference.py info                      # what the model was trained with
    python inference.py interactive               # a small REPL

The loader options are accepted before *or* after the command:

    python inference.py --from-checkpoint similar king
    python inference.py similar king --from-checkpoint

Both task-8 formats load, and they must agree (up to the six decimals the text format
keeps):

    default             outputs/word_vectors.txt -- the portable word2vec text format
    --from-checkpoint   outputs/word_vectors.pt  -- `load_checkpoint` rebuilds the
                        nn.Module; the vocabulary has to come from word_vectors.json,
                        because a `state_dict` holds numbers, not words

`--model` takes the stem of another artefact set, e.g. `--model outputs/vectors_d60`.

The maths is task 7 (`cosine_similarity`, `most_similar`, `analogy`,
`evaluate_analogies`) and task 8 (`load_embeddings`, `load_checkpoint`, `word_vectors`);
this file only adds argument handling and the not-in-the-vocabulary message.

Reference: [P1] Mikolov, Chen, Corrado, Dean (2013), "Efficient Estimation of Word
Representations in Vector Space", arXiv:1301.3781 -- Sec. 4 for 3CosAdd.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_DIR, QUESTIONS_PATH  # noqa: E402
from task7_similarity import (analogy, cosine_similarity, evaluate_analogies,  # noqa: E402
                              load_questions, most_similar)
from task8_io_and_pca import load_checkpoint, load_embeddings, word_vectors  # noqa: E402

#: Stem of the default artefact set written by train.py.
DEFAULT_MODEL = os.path.join(OUTPUT_DIR, "word_vectors")

HELP_TEXT = """commands:
  similar WORD [N]      the N nearest neighbours of WORD (default 8)
  analogy A B C [N]     A is to B as C is to ?          (default 5)
  similarity W1 W2      cosine similarity of two words
  info                  hyper-parameters and scores of the loaded model
  help                  this text
  quit                  leave (or Ctrl-D)"""


def normalise(word):
    """The corpus is lower-cased in task 1, so queries are lower-cased the same way."""
    return str(word).strip().lower()


class VectorSpace:
    """The `(vectors, word2idx, idx2word)` triple plus a helpful not-found message."""

    def __init__(self, vectors, word2idx, idx2word, metadata=None):
        self.vectors = vectors
        self.word2idx = word2idx
        self.idx2word = list(idx2word)
        self.metadata = metadata or {}

    def __len__(self):
        return len(self.idx2word)

    @property
    def dim(self):
        return int(self.vectors.shape[1])

    def index(self, word):
        """The row of `word`; raises KeyError with spelling suggestions if it is absent."""
        word = normalise(word)
        if word not in self.word2idx:
            guesses = difflib.get_close_matches(word, self.idx2word, n=3, cutoff=0.6)
            hint = f" Did you mean {', '.join(guesses)}?" if guesses else ""
            raise KeyError(f"{word!r} is not in the vocabulary of this model "
                           f"({len(self.idx2word)} words).{hint}")
        return self.word2idx[word]

    def nearest(self, word, topn=8):
        """`[(word, cosine)]`, best first, without the query word itself."""
        self.index(word)
        return most_similar(self.vectors, self.word2idx, self.idx2word,
                            normalise(word), topn)

    def analogy_by(self, a, b, c, topn=5):
        """`b - a + c` of [P1] Sec. 4, as `[(word, cosine)]` without the three inputs."""
        for word in (a, b, c):
            self.index(word)
        return analogy(self.vectors, self.word2idx, self.idx2word,
                       normalise(a), normalise(b), normalise(c), topn)

    def similarity(self, word, other):
        """cos(word, other), a plain float in [-1, 1] (0.0 for a zero vector)."""
        return cosine_similarity(self.vectors[self.index(word)],
                                 self.vectors[self.index(other)])


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def load_vector_space(model=DEFAULT_MODEL, vectors=None, metadata=None,
                      from_checkpoint=False):
    """Load one artefact set: `(<stem>.txt | <stem>.pt + <stem>.json)`.

    Parameters
    ----------
    model : str
        The stem `train.py` wrote, e.g. `outputs/word_vectors`.
    vectors, metadata : str or None
        Explicit paths for the text vectors / the JSON metadata.
    from_checkpoint : bool
        Rebuild the `SkipGram` module from `<stem>.pt` instead of reading the text file.
    """
    meta = _read_json(metadata if metadata is not None else model + ".json")
    if from_checkpoint:
        return _from_checkpoint(model, meta)

    path = vectors if vectors is not None else model + ".txt"
    if not os.path.exists(path):
        if meta is None:
            raise FileNotFoundError(f"no vectors at {path} -- run `python train.py` first")
        print(f"warning: {path} is missing, rebuilding the model from {model}.pt",
              file=sys.stderr)
        return _from_checkpoint(model, meta)

    table, word2idx, idx2word = load_embeddings(path)
    return VectorSpace(table, word2idx, idx2word, meta)


def _from_checkpoint(model, metadata):
    """Rebuild the module with task 8 and keep its `W_in` table."""
    if metadata is None:
        raise FileNotFoundError(
            f"{model}.json is missing.  A state_dict holds no vocabulary, so the words "
            "of the rows have to come from the metadata train.py writes next to it.")
    path = model + ".pt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"no checkpoint at {path} -- run `python train.py` first")

    vocab_size = int(metadata["vocab_size"])
    idx2word = [str(word) for word in metadata["vocabulary"]]
    if len(idx2word) != vocab_size:
        raise ValueError(f"{model}.json lists {len(idx2word)} words but says "
                         f"vocab_size={vocab_size}")
    network = load_checkpoint(path, vocab_size, int(metadata["dim"]))
    word2idx = {word: index for index, word in enumerate(idx2word)}
    return VectorSpace(word_vectors(network), word2idx, idx2word, metadata)


def _read_json(path):
    """The metadata JSON, or None when there is none next to the vectors."""
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------------------
# the answers
# --------------------------------------------------------------------------------------
def print_neighbours(space, word, topn=8):
    """`python inference.py similar king`."""
    _require_count(topn)
    neighbours = space.nearest(word, topn)      # a typo fails before the header is printed
    print(f"{normalise(word)}: {topn} nearest neighbours (cosine similarity)")
    for rank, (neighbour, score) in enumerate(neighbours, start=1):
        print(f"  {rank:>2}. {neighbour:<16} {score:+.3f}")


def print_analogy(space, a, b, c, topn=5):
    """`python inference.py analogy man king woman` -- 3CosAdd of [P1] Sec. 4."""
    _require_count(topn)
    guesses = space.analogy_by(a, b, c, topn)
    print(f"{normalise(a)} -> {normalise(b)} :: {normalise(c)} -> ?")
    for rank, (guess, score) in enumerate(guesses, start=1):
        print(f"  {rank:>2}. {guess:<16} {score:+.3f}")


def print_similarity(space, word, other):
    """`python inference.py similarity king queen`."""
    print(f"cos({normalise(word)}, {normalise(other)}) = "
          f"{space.similarity(word, other):+.4f}")


def print_evaluation(space, questions_path=QUESTIONS_PATH):
    """`python inference.py evaluate` -- the task 7 report over the question file."""
    if not os.path.exists(questions_path):
        print(f"error: no question file at {questions_path}", file=sys.stderr)
        return 1
    result = evaluate_analogies(space.vectors, space.word2idx, space.idx2word,
                                load_questions(questions_path))
    print(f"analogy questions ([P1] Sec. 4, {result['n']} answerable):")
    print(f"  overall   {result['accuracy']:6.1%}")
    print(f"  semantic  {result['semantic']:6.1%}")
    print(f"  syntactic {result['syntactic']:6.1%}")
    for category, stats in sorted(result["per_category"].items()):
        print(f"    {category:<28} {stats['accuracy']:>6.1%} "
              f"({stats['correct']}/{stats['total']})")
    return 0


def print_info(space):
    """`python inference.py info` -- what the metadata says about this model."""
    print(f"{'model':<10}: {len(space)} words x {space.dim} dimensions")
    meta = space.metadata
    if not meta:
        print(f"{'metadata':<10}: none next to the vectors (only train.py writes it)")
        return
    fields = (("corpus", meta.get("corpus")), ("tokens", meta.get("tokens")),
              ("min_count", meta.get("min_count")),
              ("dim/window", f"{meta.get('dim')} / {meta.get('window')}"),
              ("epochs", f"{meta.get('epochs')} x batch {meta.get('batch_size')}"),
              ("lr", f"{meta.get('start_lr')} -> {meta.get('end_lr')}"),
              ("device", f"{meta.get('device')}, seed {meta.get('seed')}"),
              ("trained in", f"{meta.get('train_seconds')}s"),
              ("created", meta.get("created_at")))
    for label, value in fields:
        print(f"{label:<10}: {value}")
    history = meta.get("mean_log_likelihood") or []
    if history:
        print(f"{'mean log p':<10}: {history[0]:+.3f} -> {history[-1]:+.3f} "
              f"over {len(history)} epochs")
    report = meta.get("analogy")
    if report:
        print(f"{'analogies':<10}: {report['accuracy']:.1%} overall, "
              f"{report['semantic']:.1%} semantic, {report['syntactic']:.1%} syntactic "
              f"({report['n']} answerable)")


def _require_count(topn):
    if int(topn) < 1:
        raise ValueError("N must be at least 1")


# --------------------------------------------------------------------------------------
# the REPL
# --------------------------------------------------------------------------------------
def interactive(space):
    """Read commands from stdin until `quit` / Ctrl-D; errors never end the session."""
    print(f"loaded {len(space)} words x {space.dim} dimensions -- type `help` for commands")
    while True:
        try:
            line = input("word2vec> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        parts = shlex.split(line)
        if not parts:
            continue
        command, arguments = parts[0].lower(), parts[1:]
        if command in ("quit", "exit", "q"):
            return 0
        try:
            if command in ("help", "?", "h"):
                print(HELP_TEXT)
            elif command == "similar":
                word, topn = _word_and_count(arguments, 8)
                print_neighbours(space, word, topn)
            elif command == "analogy":
                if len(arguments) < 3:
                    raise ValueError("usage: analogy A B C [N]")
                topn = int(arguments[3]) if len(arguments) > 3 else 5
                print_analogy(space, arguments[0], arguments[1], arguments[2], topn)
            elif command in ("similarity", "sim", "cos"):
                if len(arguments) != 2:
                    raise ValueError("usage: similarity WORD OTHER")
                print_similarity(space, arguments[0], arguments[1])
            elif command == "info":
                print_info(space)
            else:
                print(f"unknown command {command!r} -- try `help`")
        except KeyError as exc:
            print(f"error: {exc.args[0] if exc.args else exc}")
        except ValueError as exc:
            print(f"error: {exc}")


def _word_and_count(arguments, default):
    """`['king', '10']` -> `('king', 10)`; the count is optional."""
    if not arguments:
        raise ValueError("usage: similar WORD [N]")
    topn = int(arguments[1]) if len(arguments) > 1 else default
    return arguments[0], topn


# --------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------
def load_option_parser():
    """The loader options, shared by the top level and by every sub-command.

    `default=argparse.SUPPRESS` keeps a value given *after* the command from being
    overwritten by the default of the sub-parser, which is what makes both
    `inference.py --from-checkpoint similar king` and
    `inference.py similar king --from-checkpoint` work.
    """
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument("--model", default=argparse.SUPPRESS,
                         help="file name stem written by train.py "
                              f"(default: {os.path.relpath(DEFAULT_MODEL)})")
    options.add_argument("--vectors", default=argparse.SUPPRESS,
                         help="explicit path of a word2vec text file")
    options.add_argument("--metadata", default=argparse.SUPPRESS,
                         help="explicit path of the metadata JSON")
    options.add_argument("--from-checkpoint", action="store_true",
                         default=argparse.SUPPRESS,
                         help="rebuild the nn.Module from <model>.pt instead of "
                              "reading <model>.txt")
    return options


def build_parser(description="query the word vectors written by train.py"):
    options = load_option_parser()
    parser = argparse.ArgumentParser(description=description, parents=[options])
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    similar = commands.add_parser("similar", parents=[options],
                                  help="nearest neighbours of a word")
    similar.add_argument("word")
    similar.add_argument("-n", "--topn", type=int, default=8, help="how many (default 8)")

    analogies = commands.add_parser("analogy", parents=[options],
                                    help="A is to B as C is to ?")
    analogies.add_argument("a")
    analogies.add_argument("b")
    analogies.add_argument("c")
    analogies.add_argument("-n", "--topn", type=int, default=5, help="how many (default 5)")

    pair = commands.add_parser("similarity", parents=[options],
                               help="cosine similarity of two words")
    pair.add_argument("word")
    pair.add_argument("other")

    evaluate = commands.add_parser("evaluate", parents=[options],
                                   help="accuracy on an analogy question file")
    evaluate.add_argument("--questions", default=QUESTIONS_PATH)

    commands.add_parser("info", parents=[options],
                        help="hyper-parameters and scores of the loaded model")
    commands.add_parser("interactive", parents=[options], help="a small REPL")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        space = load_vector_space(getattr(args, "model", DEFAULT_MODEL),
                                  vectors=getattr(args, "vectors", None),
                                  metadata=getattr(args, "metadata", None),
                                  from_checkpoint=getattr(args, "from_checkpoint", False))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 1

    try:
        return _run(args, space)
    except KeyError as exc:                       # the word is not in the vocabulary
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 1
    except ValueError as exc:                     # bad count, wrong number of words
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run(args, space):
    """Dispatch the parsed command; the work is done by the matching `print_*` helper."""
    if args.command == "similar":
        print_neighbours(space, args.word, args.topn)
        return 0
    if args.command == "analogy":
        print_analogy(space, args.a, args.b, args.c, args.topn)
        return 0
    if args.command == "similarity":
        print_similarity(space, args.word, args.other)
        return 0
    if args.command == "evaluate":
        return print_evaluation(space, args.questions)
    if args.command == "info":
        print_info(space)
        return 0
    return interactive(space)


if __name__ == "__main__":
    raise SystemExit(main())
