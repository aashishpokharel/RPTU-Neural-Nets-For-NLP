#!/usr/bin/env python3
"""
check_my_code.py -- grade your eight task files.

    python check_my_code.py              # check every task
    python check_my_code.py --task 4     # check one task (repeatable)
    python check_my_code.py --list       # show what the tasks are
    python check_my_code.py --verbose    # full traceback on failures

Each task file prints a line per check with one of four markers:

    PASS   the check passed
    TODO   the function still raises NotImplementedError -- you have work to do
    FAIL   the function returned something wrong (the message says what)
    ERROR  the function crashed (the message is the last line of the traceback)

You can also grade a single file from inside it, e.g.

    python task4_softmax.py

The checks in grader/ are independent of the reference solution: the loss is compared with
an independent implementation *and* its autograd gradients with central finite differences,
and the training loop is judged by whether it learns, not by whether it matches an
implementation byte for byte.  Any correct solution passes.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import torch  # noqa: F401  (checked here so the message is friendly)
except ImportError:                                          # pragma: no cover
    sys.exit("This exercise needs PyTorch.  Install it with:\n\n"
             "    python -m pip install -r requirements.txt\n\n"
             "or (Apple/CPU only)  python -m pip install torch\n")

from grader.runner import main  # noqa: E402  (import needs HERE on sys.path)


if __name__ == "__main__":
    raise SystemExit(main())
