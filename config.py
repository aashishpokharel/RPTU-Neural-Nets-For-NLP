"""
config.py -- shared constants and paths for the word2vec exercises.

This file is GIVEN TO YOU.  Do not change it: `check_my_code.py` and the reference
solutions import these names, so editing them will make the grader compare apples with
oranges.

Reference for every constant below:

  [P1] Mikolov, Chen, Corrado, Dean (2013), "Efficient Estimation of Word Representations
       in Vector Space", arXiv:1301.3781
  [P2] Mikolov, Sutskever, Chen, Corrado, Dean (2013), "Distributed Representations of
       Words and Phrases and their Compositionality", arXiv:1310.4546
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.txt")
QUESTIONS_PATH = os.path.join(DATA_DIR, "questions.txt")

# --------------------------------------------------------------------------------------
# Model hyper-parameters
# --------------------------------------------------------------------------------------
#: Dimensionality of the word vectors, D.  [P1] Sec. 4.2 trains D = 300 on a 1e9 word
#: corpus; on the 60k word toy corpus in data/ that would just memorise the corpus, so
#: the exercises use a small D that trains in seconds on a laptop.
DEFAULT_DIM = 40

#: C, the maximum distance of a context word from the centre word.  [P1] Sec. 3.2 uses
#: C = 5 for the Skip-gram toy example and C = 10 for the big experiments.
DEFAULT_WINDOW = 5

#: Words with fewer occurrences than this are dropped from the vocabulary.
#: [P1] Sec. 4.2 restricts the vocabulary to the most frequent words.
DEFAULT_MIN_COUNT = 3

#: Number of passes over the corpus.  Five epochs already reach ~54% on the analogy task
#: of this toy corpus (chance is ~1%), and the whole run takes about a second.
DEFAULT_EPOCHS = 5

#: Mini-batch size.  word2vec is trained with plain SGD (one pair at a time) in the C
#: implementation; the same gradients averaged over a batch give a much friendlier
#: PyTorch loop and, on this toy corpus, the same quality.
DEFAULT_BATCH_SIZE = 64

#: Where the tensors live.  "cpu" keeps the exercise bit-reproducible on every machine;
#: pass "mps" (Apple) or "cuda" (NVIDIA) to train on a GPU -- the corpus is so small that
#: you should not expect a speed-up from it.
DEFAULT_DEVICE = "cpu"

#: Linear learning-rate decay, [P1] Sec. 4.2: "we chose starting learning rate 0.025 and
#: decreased it linearly, so that it approaches zero at the end of the last training
#: epoch".
#:
#: NOTE the scale: our loss is the *mean* over a mini-batch of pairs, while [P1] takes
#: one SGD step per pair.  A mean-reduced loss gives the optimizer a gradient that is
#: `batch_size` times smaller, so the equivalent per-pair learning rate here is
#: start_lr / batch_size = 0.5 / 64 ~= 0.008.  That is the number to compare with [P1].
DEFAULT_START_LR = 0.5
DEFAULT_END_LR = 1e-4

#: Fast defaults for the driver script `run_word2vec.py`.  A full-corpus run is a few
#: seconds slower and only marginally better on this toy corpus.
QUICK_LIMIT = 20000       # use only the first N tokens of the corpus
QUICK_EPOCHS = 5

# --------------------------------------------------------------------------------------
# Output files written by run_word2vec.py
# --------------------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(HERE, "outputs")
EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "word_vectors.txt")
PLOT_PATH = os.path.join(OUTPUT_DIR, "word_vectors_pca.png")

#: The words that are labelled in the PCA figure: two clear semantic groups
#: (country -> capital) and a syntactic one (verb -> -ing form).
PCA_WORDS = [
    "france", "paris", "germany", "berlin", "italy", "rome", "spain", "madrid",
    "japan", "tokyo", "china", "beijing", "russia", "moscow", "poland", "warsaw",
    "walk", "walking", "swim", "swimming", "read", "reading", "think", "thinking",
]
