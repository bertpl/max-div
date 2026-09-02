"""Helper methods that centralize functionality to sample items for selection changes.

Items can be sampled to be removed from the selection (remove.py) or added to the selection (add.py).

These methods bridge the gap between following 3 pieces of functionality:
    1) SolverState offering information about...
        - selected and unselected items
        - per-point diversity contributions (higher = contributes more diversity)
        - constraints
    2) Methods to modify selectivity of an array of probabilities
    3) Sampling strategies that decide which items to sample based on the above information
"""

from ._helpers import gather_f32, remove_sample_from_candidates, remove_sample_from_candidates_and_p
from .add import SamplingType, build_add_probabilities, select_items_to_add, select_items_to_add_with_p
from .remove import select_items_to_remove
