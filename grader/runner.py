"""
grader/runner.py -- run the per-task checks and print a Coursera-style report.

The runner knows how to find two things for every task:

    TASKS[n] = (human readable title, module name of the file you edit)

and it imports the matching checks from `grader/task<n>.py`, which exposes a list

    CHECKS = [(title, callable), ...]

Every check raises AssertionError when it fails and does nothing when it passes.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import traceback

TASKS = {
    1: ("vocabulary: words -> ids", "task1_vocabulary"),
    2: ("training pairs, dynamic window and Dataset", "task2_training_pairs"),
    3: ("the model: two embedding tables", "task3_embeddings"),
    4: ("the softmax and the Skip-gram loss", "task4_softmax"),
    5: ("learning rate, optimizer and one step", "task5_learning_rate"),
    6: ("the Skip-gram training loop", "task6_training_loop"),
    7: ("similarity, nearest neighbours and analogies", "task7_similarity"),
    8: ("I/O: export, checkpoint and the PCA figure", "task8_io_and_pca"),
}

BONUS = {8}

_STATE = {"prefix": "", "overrides": {}}


# --------------------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------------------
def set_exercise_package(prefix=""):
    """Grade `solutions/` instead of the exercise files by passing 'solutions.'."""
    _STATE["prefix"] = prefix


def set_module_override(task, module):
    """Grade an already imported module for `task` (used by `python taskN_*.py`)."""
    _STATE["overrides"][task] = module


def current_exercise_module(task):
    """The module that holds the student's implementation of `task`."""
    if task in _STATE["overrides"]:
        return _STATE["overrides"][task]
    return importlib.import_module(_STATE["prefix"] + TASKS[task][1])


def _load_checks(task):
    return importlib.import_module(f"grader.task{task}").CHECKS


def _run_check(fn, verbose):
    """Return (status, detail) with status in {'PASS', 'TODO', 'FAIL', 'ERROR'}."""
    try:
        fn()
        return "PASS", ""
    except NotImplementedError as exc:
        return "TODO", str(exc) or "not implemented yet"
    except AssertionError as exc:
        return "FAIL", str(exc)
    except Exception:                       # noqa: BLE001 - anything else is a bug
        return "ERROR", traceback.format_exc(limit=4 if verbose else 1).strip().splitlines()[-1]


def run_task(task, verbose=False, out=sys.stdout):
    """Run every check of one task.  Returns True if the task is finished."""
    title, module_name = TASKS[task]
    bonus = " (bonus)" if task in BONUS else ""
    print("", file=out)
    print("=" * 78, file=out)
    print(f"TASK {task}{bonus} -- {title}   [{module_name}.py]", file=out)
    print("=" * 78, file=out)

    try:
        checks = _load_checks(task)
    except Exception:                       # pragma: no cover
        traceback.print_exc(file=out)
        return False

    passed = todo = failed = 0
    for index, (label, fn) in enumerate(checks, start=1):
        status, detail = _run_check(fn, verbose)
        marker = {"PASS": "PASS", "TODO": "TODO", "FAIL": "FAIL", "ERROR": "ERROR"}[status]
        print(f"  {index}. [{marker}] {label}", file=out)
        if status == "PASS":
            passed += 1
        elif status == "TODO":
            todo += 1
            print(f"        -> {detail}", file=out)
        else:
            failed += 1
            print(f"        -> {detail}", file=out)

    total = len(checks)
    print(f"  ---------------------------------------------------------------", file=out)
    if passed == total:
        print(f"  {passed}/{total} checks passed -- task {task} is done.", file=out)
    else:
        extra = []
        if todo:
            extra.append(f"{todo} not implemented yet")
        if failed:
            extra.append(f"{failed} wrong")
        print(f"  {passed}/{total} checks passed ({', '.join(extra)}).", file=out)
    return passed == total


def run_all(tasks=None, verbose=False, out=sys.stdout):
    """Run the requested tasks (default: all).  Returns the list of unfinished tasks."""
    tasks = list(TASKS) if tasks is None else list(tasks)
    unfinished = [t for t in tasks if not run_task(t, verbose=verbose, out=out)]

    print("", file=out)
    print("=" * 78, file=out)
    if not unfinished:
        print(f"All {len(tasks)} task(s) passed.  Now try:  python run_word2vec.py", file=out)
    else:
        print("Unfinished tasks: " + ", ".join(str(t) for t in unfinished), file=out)
        print("Re-check a single task with:  python check_my_code.py --task <n>", file=out)
    print("=" * 78, file=out)
    return unfinished


def main(argv=None, tasks=None):
    """Command line entry point shared by `check_my_code.py` and the task files."""
    parser = argparse.ArgumentParser(
        description="grade the word2vec-from-scratch exercises")
    parser.add_argument("--task", type=int, action="append", default=None,
                        help="only grade this task (repeatable)")
    parser.add_argument("--list", action="store_true", help="list the tasks and exit")
    parser.add_argument("--verbose", action="store_true", help="more detail on failures")
    parser.add_argument("--solution", action="store_true",
                        help="grade solutions/ instead of your own code (self-test)")
    args = parser.parse_args(argv)

    if args.list:
        for number, (title, module_name) in TASKS.items():
            bonus = " (bonus)" if number in BONUS else ""
            print(f"  task {number}{bonus:<9} {module_name + '.py':<26} {title}")
        return 0

    if args.solution:
        set_exercise_package("solutions.")

    if tasks is None:
        tasks = args.task
    unfinished = run_all(tasks, verbose=args.verbose)
    return 0 if not unfinished else 1
