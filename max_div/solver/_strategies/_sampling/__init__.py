"""
Helper methods that centralize functionality to sample items to be removed from the selection (remove.py)
or added to the selection (add.py).

These methods bridge the gap between following 3 pieces of functionality:
    1) SolverState offering information about...
        - selected and unselected items
        - separation distances between items
        - constraints
    2) Methods to modify selectivity of an array of probabilities
    3) Sampling strategies that decide which items to sample based on the above information
"""

from ._helpers import remove_sample_from_candidates, remove_sample_from_candidates_and_p
from .add import SamplingType, select_items_to_add
from .remove import select_items_to_remove
