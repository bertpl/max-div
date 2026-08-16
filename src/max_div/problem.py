"""Public API for defining Maximum Diversity problems and constraints."""

from ._core._warnings import DistanceInputWarning, MaxDivWarning
from ._core.constraints import Constraint
from ._core.feasibility import FeasibilityResult, FeasibilityStatus
from ._core.problem import DistanceMaxDivProblem, MaxDivProblem, VectorMaxDivProblem

__all__ = [
    "Constraint",
    "DistanceInputWarning",
    "DistanceMaxDivProblem",
    "FeasibilityResult",
    "FeasibilityStatus",
    "MaxDivProblem",
    "MaxDivWarning",
    "VectorMaxDivProblem",
]


# --- module patching ---------------------------------------------------------
# This ensures that the user sees all re-exported names as belonging to this module, rather than their original
from max_div._core._utils.api import patch_modules

patch_modules(globals(), __all__, __name__)
del patch_modules
