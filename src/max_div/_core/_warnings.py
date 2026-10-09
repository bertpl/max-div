"""Warning categories emitted by max-div.

All max-div warnings derive from `MaxDivWarning`, so users can filter the whole family with one
`warnings.filterwarnings` rule, or target a specific subclass.
"""

import inspect
import os
import warnings
from pathlib import Path

# Frames whose file lies below this directory belong to max-div.
_MAX_DIV_DIR = str(Path(__file__).absolute().parent.parent) + os.sep


# ==================================================================================================
#  Warning categories
# ==================================================================================================


class MaxDivWarning(UserWarning):
    """Base category for every warning max-div emits."""


class DistanceInputWarning(MaxDivWarning):
    """Distance input needed intervention at the problem boundary: a conversion copy or a symmetry repair."""


class ParallelSolvingWarning(MaxDivWarning):
    """Solving in parallel was configured in a way that cannot help, or that runs more workers than there are cores."""


class SolverBudgetWarning(MaxDivWarning):
    """A solve reached its optimization with no end-to-end budget left, so it returns what initialization built."""


class FeasibilityConvergenceWarning(MaxDivWarning):
    """The feasibility relaxation solve stopped at its iteration cap before converging; verdicts stay sound."""


# ==================================================================================================
#  Issuing warnings
# ==================================================================================================
def warn_outside_max_div(message: str, category: type[Warning]) -> None:
    """Issue a warning attributed to the first stack frame outside max-div.

    A fixed `stacklevel` cannot point at the user's line when entry points reach the warning through call
    stacks of different depths, as `MaxDivProblem.new()` and a direct constructor call do.  Both also pass
    through the dataclass-generated `__init__`, whose frame has the file name `<string>`, so the walk skips
    that frame too.  `warnings.warn(skip_file_prefixes=...)` does not skip it on Python 3.12.
    """
    frame = inspect.currentframe()
    stacklevel = 1  # stacklevel 1 of warnings.warn is this function's own frame
    while frame is not None:
        code = frame.f_code
        is_generated_dataclass_init = code.co_filename == "<string>" and code.co_name == "__init__"
        if not code.co_filename.startswith(_MAX_DIV_DIR) and not is_generated_dataclass_init:
            break
        frame = frame.f_back
        stacklevel += 1
    del frame  # a frame reference held by a local creates a reference cycle
    warnings.warn(message, category, stacklevel=stacklevel)
