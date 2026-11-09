"""
TASK 1 -- from a text file to word ids.

    [P1] Mikolov, Chen, Corrado, Dean (2013), "Efficient Estimation of Word
         Representations in Vector Space", arXiv:1301.3781

The model below (task 6) is trained on (center, context) pairs of *word ids*, so before
we can do anything we need the two dictionaries that every word-embedding pipeline
starts with:

    tokens   a flat list of words, in the order they appear in the corpus
    vocab    the set of words we keep, numbered 0, 1, 2, ... by frequency

[P1] Sec. 4.2 keeps only the most frequent words of the vocabulary (the long tail of
words that occur once or twice would dominate the model and be useless anyway), so
`build_vocab` gets a `min_count` argument.

Run this file on its own to check yourself:

    python task1_vocabulary.py
"""

from __future__ import annotations

from collections import Counter, OrderedDict

import numpy as np

from config import CORPUS_PATH, DEFAULT_MIN_COUNT


def read_tokens(path=CORPUS_PATH, limit=None):
    """Read the corpus and return it as a flat list of lower-cased words.

    Parameters
    ----------
    path : str
        Text file with one sentence (or document) per line.
    limit : int or None
        Keep only the first `limit` tokens; None means the whole file.

    Returns
    -------
    list of str

    Splitting on whitespace is enough -- the corpus has no punctuation that matters and
    keeping it simple means the grader can compare your list with the reference one.
    Lower-case everything so that "The" and "the" share an id.

    Hint: the natural implementation is a loop over `open(path)`, `line.lower().split()`
    and `tokens.extend(...)`, returning as soon as `len(tokens) >= limit`.
    """
    # raise NotImplementedError("Task 1: read_tokens()")

    with open (CORPUS_PATH,'r') as f:
        data = f.read()

    data_lower = data.lower()
    data_tokens = data_lower.split()
    if limit is not None:
        limit_num = limit
    else:
        limit_num = len(data_tokens)
    return data_tokens[0:limit_num]



def build_vocab(tokens, min_count=DEFAULT_MIN_COUNT):
    """Number the words of `tokens` by decreasing frequency.

    Parameters
    ----------
    tokens : list of str
    min_count : int
        Words that occur fewer than `min_count` times are dropped.

    Returns
    -------
    counts : np.ndarray, shape (V,), integer
        counts[i] is how often idx2word[i] occurs in `tokens`.
    word2idx : dict[str, int]
        word -> id.
    idx2word : list of str
        id -> word, i.e. the inverse mapping, so that idx2word[word2idx[w]] == w.

    Rules the rest of the exercise relies on
    ----------------------------------------
    * the most frequent word gets id 0 (on this corpus that is "the");
    * words with the same count are ordered alphabetically, so the vocabulary is
      reproducible;
    * `counts[l]` is the corpus frequency *before* min_count filtering, and
      `counts.sum()` equals the number of tokens you can still encode (see `encode`).
    """
    # raise NotImplementedError("Task 1: build_vocab()")
    
    word_count = {key: value for key,value in dict(Counter(tokens)).items() if value >= min_count}

    word_count_final= dict(OrderedDict(sorted(word_count.items(), key=lambda item: (-item[1], item[0]))))



    counts = np.array(list(word_count_final.values()))

    word2idx = {word: idx for idx , (word,count) in enumerate(word_count_final.items())}

    idx2word = [word  for idx , (word, count) in enumerate(word_count_final.items())  ]


    return counts, word2idx, idx2word


    


def encode(tokens, word2idx):
    """Turn a list of words into a list of ids.

    Words that are missing from `word2idx` (they were cut by `min_count`, or never
    occurred in the corpus) are silently skipped: they simply do not take part in
    training.  The order of the remaining ids must be the order of `tokens`.

    Returns
    -------
    np.ndarray, shape (n_kept,), integer dtype
    """
    valid_tokens = [token for token in tokens if token in list(word2idx.keys())]
    tokens_updated = [word2idx[token] for token in valid_tokens ]

    return np.array(tokens_updated, dtype = int)
    # raise NotImplementedError("Task 1: encode()")


if __name__ == "__main__":
    import sys

    from grader.runner import main as _grade
    from grader.runner import set_module_override

    set_module_override(1, sys.modules[__name__])
    raise SystemExit(_grade(tasks=[1]))
