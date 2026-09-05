"""Adapters: one per compared tool/baseline, all implementing SelectionAdapter."""

from .apricot_fl import ApricotFacilityLocation
from .base import SelectionAdapter
from .code_fdm import CodeFdmFairFlow, CodeFdmSingleColor
from .dppy_kdpp import DppyKDpp
from .fps import FpsampleFPS, SkmatterFPS
from .kmedoids_fasterpam import KMedoidsFasterPAM
from .qc_selector import QcSelectorMaxMin, QcSelectorMaxSum
from .random_baseline import RandomBaseline
from .rdkit_maxmin import RdkitMaxMin

__all__ = [
    "ApricotFacilityLocation",
    "CodeFdmFairFlow",
    "CodeFdmSingleColor",
    "DppyKDpp",
    "FpsampleFPS",
    "KMedoidsFasterPAM",
    "QcSelectorMaxMin",
    "QcSelectorMaxSum",
    "RandomBaseline",
    "RdkitMaxMin",
    "SelectionAdapter",
    "SkmatterFPS",
]
