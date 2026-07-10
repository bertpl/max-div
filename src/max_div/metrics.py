from ._core.metrics import DistanceMetric, DiversityMetric

__all__ = [
    "DistanceMetric",
    "DiversityMetric",
]


# --- module patching ---------------------------------------------------------
# This ensures that the user sees all re-exported names as belonging to this module, rather than their original
from max_div._core._utils.api import patch_modules

patch_modules(globals(), __all__, __name__)
del patch_modules
