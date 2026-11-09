# word2vec with PyTorch

A hands-on reconstruction of the original word2vec papers, written with PyTorch:

* [P1] Mikolov, Chen, Corrado, Dean (2013), *Efficient Estimation of Word Representations
  in Vector Space*, arXiv:1301.3781
* [P2] Mikolov, Sutskever, Chen, Corrado, Dean (2013), *Distributed Representations of
  Words and Phrases and their Compositionality*, arXiv:1310.4546

You build vanilla Skip-gram with the **full softmax** (no hierarchical softmax, no
negative sampling -- that is what [P2] adds, and this exercise shows you *why* it was
needed), find the nearest neighbours and solve the analogy identities of [P1] Sec. 4, then
look at the PCA projection of [P1] Sec. 5.

PyTorch does the *bookkeeping*, not the thinking.  You write the model (`nn.Embedding`
tables), the loss (`-log p(context | centre)` with the full softmax), the optimizer step
and the training loop; `autograd` fills `.grad`, and the grader checks your loss against an
independent implementation **and** its gradients against central finite differences, so
"it runs" is not the same as "it is right".

## Requirements

```bash
python3 -m pip install -r requirements.txt     # torch, numpy, matplotlib
```

Tested with Python 3.15, PyTorch 2.15, NumPy 2.5 and matplotlib 3.11 (Python 3.9+ with
PyTorch 2.x is enough).  `matplotlib` is only needed for the bonus PCA figure (task 8);
everything else works without it.  Training runs on the CPU in seconds -- `--device mps` /
`--device cuda` also work, but this corpus is too small to benefit.

## Quick start

```bash
python3 check_my_code.py            # grade all eight tasks (start here, everything TODO)
python3 check_my_code.py --task 4   # grade one task
python3 task4_softmax.py            # same thing, from inside the file you are editing
python3 run_word2vec.py --plot      # train on 20k tokens and write the figure
python3 train.py --plot             # the same training, but keeps the artefacts
python3 inference.py similar king   # query the vectors train.py wrote
```

`run_word2vec.py` needs tasks 1-7; once they pass it prints something like this:

```
torch     : 2.15.0 on cpu
corpus    : 20000 tokens (first 20,000 tokens)
vocabulary: 130 words with count >= 3, 20000 tokens kept
epoch   1/5  mean log p = -4.2450  (119234 pairs, lr = 0.40070)
...
epoch   5/5  mean log p = -3.4379  (119894 pairs, lr = 0.00015)
training  : 2.9s, dim=40, window=5, epochs=5, batch_size=64, lr 0.5 -> 0.0001
mean log-likelihood per epoch: -4.245, -3.656, -3.513, -3.459, -3.438

nearest neighbours (cosine similarity):
       king: empress (0.99), queen (0.99), princess (0.98), emperor (0.98), ...
     france: portugal (0.99), greece (0.99), russia (0.99), poland (0.99), ...
    biggest: easiest (1.00), oldest (1.00), luckiest (1.00), smallest (0.99), ...

analogies (a is to b as c is to ?):
  man -> king :: woman -> queen (1.00), empress (1.00), emperor (0.98)
  france -> paris :: germany -> berlin (0.99), poland (0.98), moscow (0.98)
  walk -> walking :: swim -> swimming (1.00), speaking (0.99), reading (0.99)

analogy questions ([P1] Sec. 4, 160 answerable):
  overall    48.8%
  semantic   44.3%
  syntactic  52.2%

vectors written to .../outputs/word_vectors.txt
checkpoint written to .../outputs/word_vectors.pt (torch.save of the state dict)
PCA figure written to .../outputs/word_vectors_pca.png
```

Those numbers are the reference solution with the default settings (`--full` trains on the
whole 59k-token corpus and takes about three times longer).  See
`python3 run_word2vec.py --help` for `--dim`, `--window`, `--epochs`, `--batch-size`,
`--full`, `--device`, `--seed`, `--compare` and friends.

## Training and inference

`run_word2vec.py` trains and reports in one go; `train.py` and `inference.py` split that
in two, so the vectors can be trained once and queried many times (and from another
program):

```bash
python3 train.py                        # 20k tokens, 5 epochs, a few seconds
python3 train.py --full --plot          # whole corpus + the PCA figure
python3 train.py --dim 60 --name d60    # a second artefact set, side by side

python3 inference.py similar king           # 8 nearest neighbours
python3 inference.py similarity king queen  # cosine similarity of two words
python3 inference.py analogy man king woman # man -> king :: woman -> ?
python3 inference.py evaluate               # the analogy report of task 7
python3 inference.py info                   # hyper-parameters and scores
python3 inference.py interactive            # a small REPL
```

`train.py` writes three files into `outputs/` (git-ignored), all sharing the `--name`
stem:

| file          | what it holds                                               |
| ------------- | ----------------------------------------------------------- |
| `<name>.txt`  | the vectors in the word2vec text format of task 8            |
| `<name>.pt`   | the `state_dict` checkpoint of task 8                        |
| `<name>.json` | hyper-parameters, vocabulary, per-epoch `J`, analogy scores  |

The JSON exists because a `state_dict` is only a pile of numbers: it holds neither the
words nor the hyper-parameters, so without it there is no way to say which row is "king".
With it, `inference.py --from-checkpoint` rebuilds the `nn.Module`; without that flag it
reads the portable text file.  Both give the same answers, up to the six decimals the text
format keeps:

```bash
python3 inference.py similar king                      # reads outputs/word_vectors.txt
python3 inference.py similar king --from-checkpoint     # rebuilds word_vectors.pt
python3 inference.py similar king --model outputs/d60   # another artefact set
```

A word that is not in the vocabulary is reported as such — with spelling suggestions —
and the exit status is non-zero, so both scripts are safe to call from a Makefile or a
shell script:

```
$ python3 inference.py similar kiing
error: 'kiing' is not in the vocabulary of this model (130 words). Did you mean king, walking, kingdom?
$ echo $?
1
```

Neither script contains any new mathematics: `train.py` is `run_word2vec.py` with the
artefacts written in one place, and `inference.py` is argument handling around
`most_similar`, `analogy` and `cosine_similarity` from task 7 and `load_embeddings` /
`load_checkpoint` from task 8.

## The files

| file                      | what it holds                                             |
| ------------------------- | --------------------------------------------------------- |
| `config.py`               | hyper-parameters, paths, the words labelled in the figure  |
| `requirements.txt`        | torch, numpy, matplotlib                                   |
| `data/corpus.txt`         | 10k lines of simple English, five interleaved "topics"     |
| `data/questions.txt`      | 160 analogy questions in the format of the Google set      |
| `task1_vocabulary.py`     | read the corpus, build the vocabulary, encode it           |
| `task2_training_pairs.py` | (centre, context) pairs, the dynamic window, the `Dataset`  |
| `task3_embeddings.py`     | the `SkipGram` module: two `nn.Embedding` tables, `h`, `u`  |
| `task4_softmax.py`        | softmax, log-softmax, the Skip-gram loss, `J`              |
| `task5_learning_rate.py`  | linear lr decay, `make_optimizer`, one `train_step`        |
| `task6_training_loop.py`  | the training loop: `DataLoader`, epochs, schedule          |
| `task7_similarity.py`     | cosine similarity, nearest neighbours, analogies           |
| `task8_io_and_pca.py`     | bonus: text export, `state_dict` checkpoint, PCA figure    |
| `train.py`                | train once, write `<name>.{txt,pt,json}` (tasks 1-8)        |
| `inference.py`            | query them: neighbours, analogies, similarities, REPL       |
| `check_my_code.py`        | the grader -- runs every check and prints a report         |
| `grader/`                 | the checks themselves, plus independent reference maths    |
| `solutions/`              | reference solutions and their own `run_word2vec.py`        |

Do not edit `config.py` or `grader/`: the grader imports the constants and the checks
compare your output with them.

## The eight tasks

1. **Vocabulary** -- words to ids. Frequency ordering, `min_count` filtering, tie-breaking
   so that runs are reproducible.
2. **Training pairs** -- the dynamic window of [P1] Sec. 3.2: for every centre word draw
   `r ~ U{1..window}` and pair it with everything at distance `1..r`. Also the CBOW
   generator and a `torch.utils.data.Dataset`, so that `DataLoader` can batch the pairs.
3. **The model** -- a `SkipGram(nn.Module)` with `W_in` and `W_out` as `nn.Embedding`
   tables, initialised as [P1] Sec. 4.2 prescribes (`W_in ~ U(-0.5/dim, 0.5/dim)`,
   `W_out = 0`), a seeding helper, and the one-line hidden layer `h = W_in(ids)`.
4. **The loss** -- `J = log softmax(u)[label]` with the stable log-softmax, `-mean J` as the
   batch loss (with `sum` / `none` reductions), and an `torch.no_grad()` reporter.  This is
   the full softmax of [P1]: one pair costs `O(V * dim)`.
5. **Optimizer and step** -- the linear decay of [P1] Sec. 4.2, `torch.optim.SGD` over
   *both* tables, and the four-line `zero_grad / forward / backward / step` cycle.
6. **Training loop** -- per-epoch pairs, `DataLoader` over a `SkipGramDataset`, the
   learning-rate schedule indexed by the global step, and the average `J` of each epoch.
7. **Similarity** -- cosine similarity, nearest neighbours, `b - a + c` analogies, the
   Google-format question file and an accuracy report (semantic / syntactic / per category).
8. **Bonus** -- write and read the `word2vec` text format, save and reload a
   `state_dict` checkpoint, 2-D PCA and a PNG figure.

## How the grading works

```bash
python3 check_my_code.py            # all tasks
python3 check_my_code.py --task 6   # one task, any number of times
python3 check_my_code.py --verbose  # full traceback for ERROR / FAIL
```

Every check prints one marker:

| marker | meaning                                                              |
| ------ | -------------------------------------------------------------------- |
| `PASS` | the check passed                                                      |
| `TODO` | the function still raises `NotImplementedError` -- nothing to fix yet  |
| `FAIL` | the function returned something wrong; the message says what          |
| `ERROR` | the function crashed; the message is the last line of the traceback   |

The checks are *not* diffed against the reference implementation:

* the loss of task 4 is compared with an independent implementation, and the gradients
  `autograd` produces are compared with **central finite differences** of your own loss --
  so any correct objective passes, including one written with `F.cross_entropy`;
* tasks 5 and 6 are judged by whether `J` actually increases (with `lr = 0` nothing may
  move, which is how the grader notices a hand-written update that ignores the optimizer);
* task 6 additionally has to reach **>= 15% accuracy** on the analogy questions; chance is
  about 1% and the reference solution reaches ~50% on 20k tokens.

One consequence worth knowing: task 6's quality check trains a real model (a second or
two), which is why `check_my_code.py` takes a few seconds end to end.

## Checking the graders themselves

```bash
python3 check_my_code.py --solution   # grade solutions/ instead of your files
```

All eight tasks must pass; that is the self-test for the exercise, and it is how the
package was verified before you got it.

## Common pitfalls

* **Forgetting `optimizer.zero_grad()`.** Gradients accumulate: the second batch is
  updated with the sum of two batches' gradients and the run diverges.
* **Updating by hand instead of `optimizer.step()`.** Then the learning-rate decay is
  ignored (the grader runs one step with `lr = 0` and expects no movement).
* **Learning-rate scale.** The loss is a *mean* over the batch, so its gradient is
  `batch_size` times smaller than [P1]'s per-pair gradient: `0.5` with batch 64 is the
  equivalent of `0.025` one pair at a time.  With `lr = 0.025` and batch 64 nothing moves
  and the loss sits at `-log(V)`.
* **Plain `exp` instead of log-softmax.** `exp(1000)` overflows to `inf`; use
  `torch.logsumexp` or `F.log_softmax`.
* **`W_out` left at zero / only one row updated.** The softmax pushes the correct word up
  *and* every other word down, so every row of `W_out` gets a gradient.
* **Ids without `dtype=torch.long`.** `nn.Embedding` rejects float indices; a float32 id
  also silently truncates large vocabularies.
* **Dividing by zero** in the cosine similarity of a zero vector (clamp the denominator).
* **`python` vs `python3`.** On macOS the `python` on `PATH` is often a system interpreter
  without PyTorch; use the interpreter (or virtualenv) where you installed
  `requirements.txt`.

## References

* [P1] T. Mikolov, K. Chen, G. Corrado, J. Dean. *Efficient Estimation of Word
  Representations in Vector Space.* arXiv:1301.3781, 2013.
* [P2] T. Mikolov, I. Sutskever, K. Chen, G. Corrado, J. Dean. *Distributed
  Representations of Words and Phrases and their Compositionality.* arXiv:1310.4546, 2013.
* The original C implementation, `word2vec.c`, is worth reading next to task 2 and task 5:
  the dynamic window and the learning-rate schedule here are copied from it.
* The PyTorch tutorial *Word Embeddings: Encoding Lexical Semantics* trains the same model
  with `nn.EmbeddingBag`; this exercise takes it apart into the pieces you write yourself.
