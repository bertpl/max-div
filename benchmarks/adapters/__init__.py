"""Adapters: one per compared tool/baseline, all implementing SelectionAdapter."""

from .apricot_fl import ApricotFacilityLocation
from .base import SelectionAdapter
from .code_fdm import CodeFdmFairFlow
from .fps import FpsampleFPS, SkmatterFPS
from .greedy_maxsum import GreedyMaxSum
from .kmedoids_fasterpam import KMedoidsFasterPAM
from .qc_selector_maxmin import QcSelectorMaxMin
from .random_baseline import RandomBaseline
from .rdkit_maxmin import RdkitMaxMin

__all__ = [
    "ApricotFacilityLocation",
    "CodeFdmFairFlow",
    "FpsampleFPS",
    "GreedyMaxSum",
    "KMedoidsFasterPAM",
    "QcSelectorMaxMin",
    "RandomBaseline",
    "RdkitMaxMin",
    "SelectionAdapter",
    "SkmatterFPS",
]
