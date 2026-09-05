"""Provide a certifier that kills its process, so the crash path of the isolated tier-1 certification is testable."""

import os

from max_div.problem import MaxDivProblem


def crash(problem: MaxDivProblem) -> None:
    """Exit the process outright, the way a crashing solver library would."""
    os._exit(3)
