"""
grader -- automatic checks for the word2vec exercise (PyTorch).

`grader.runner` knows the task list, imports your file for task N and runs the checks
in `grader/taskN.py`.  Run everything with::

    python check_my_code.py
"""

from grader.runner import BONUS, TASKS, run_all, run_task, set_exercise_package  # noqa: F401

__all__ = ["TASKS", "BONUS", "run_task", "run_all", "set_exercise_package"]
