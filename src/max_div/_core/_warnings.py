"""Warning categories emitted by max-div.

All max-div warnings derive from `MaxDivWarning`, so users can filter the whole family with one
`warnings.filterwarnings` rule, or target a specific subclass.
"""

# =================================================================================================
#  Warning categories
# =================================================================================================


class MaxDivWarning(UserWarning):
    """Base category for every warning max-div emits."""


class DistanceInputWarning(MaxDivWarning):
    """Distance input needed intervention at the problem boundary: a conversion copy or a symmetry repair."""


class ParallelSolvingWarning(MaxDivWarning):
    """Solving in parallel was configured in a way that cannot help, or that runs more workers than there are cores."""


class SolverBudgetWarning(MaxDivWarning):
    """A solve reached its optimization with no end-to-end budget left, so it returns what initialization built."""
