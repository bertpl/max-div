import math

import numpy as np
from numpy.typing import NDArray

from max_div.solver._solver_state import SolverState
from max_div.solver._strategies._sampling import SamplingType, select_items_to_add

from ._base import InitializationStrategy


class InitRandomBatched(InitializationStrategy):
    """
    Initialize by taking `b` (hence: batches) random samples of ~round(k/n_batches) items.  After each batch,
      the SolverState updates distances and separations, influencing sampling probabilities of the next batch.

    When sampling a batch, we use probabilities p[i] ~= (separation of i wrt already selected items)
                                                                            + (separation of i wrt all items)

    This drives each batch to be sampled from elements that are both well-separated from the selection so far, to
      promote diversity, and also well-separated from each other, to avoid samples within a batch that are far from
      the selection but close to each other.

    As we progress through the batches, selectivity of p[i] is modified with modifier = #sampled / #to_sample.
        (see modify_p_selectivity for details)

    Suggested use: when `InitRandomOneShot` does not provide a sufficiently high-quality initialization, but when
                   e.g. `InitEager` is too slow.

    Parameters:
    - b (int): Number of batches to sample (must be > 1).
                     -> If e.g. k=100 and b=5, each batch samples 20 items.
                     -> If k is not an exact multiple of b, the first batches will be slightly larger.
    - ignore_constraints (bool): If `False`, respects problem constraints during initialization, if present.
                                 If `True`, constraints are ignored. (default: `False`)

    Time Complexity:
       - without constraints: ~O(bn)
       - with constraints:    ~O(kn + bn)
    """

    __MODIFY_P_METHOD: np.int32 = np.int32(20)  # method using fast_pow_f32(p[i], t)
    __SAMPLE_EAGER: bool = True  # always use eager sampling for this case

    def __init__(self, b: int, ignore_constraints: bool = False):
        """
        Constructor for InitRandomBatched class.
        :param b: (int) Number of batches to sample (must be > 1).
        :param ignore_constraints: (bool, default=False)
                                If `False`, respects problem constraints during initialization, if present.
                                If `True`, constraints are ignored.
        """

        # --- parameter validation --------------
        if b <= 1:
            raise ValueError("InitRandomBatched requires b > 1; for b=1 use InitRandomOneShot instead.")

        # --- settings --------------------------
        name = f"InitRandomBatched(b={b}" + (",uncon)" if ignore_constraints else ")")
        super().__init__(name)
        self.b = b
        self.ignore_constraints = ignore_constraints

        # --- initialize state ------------------
        self._batches_remaining = 0  # set appropriately when calling get_next_samples the first time

    def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
        # --- determine batch size --------------
        n_selected = state.n_selected
        if n_selected == 0:
            # first call
            self._batches_remaining = self.b

        batch_size = int(max(1, math.ceil(k_remaining / self._batches_remaining)))
        self._batches_remaining -= 1

        # --- select samples --------------------
        modifier = min(0.9, n_selected / state.k)  # proportional to progress; cap at 0.9 to avoid extremes
        return select_items_to_add(
            state=state,
            candidates=state.not_selected_index_array,
            k=batch_size,
            selectivity_modifier=modifier,
            rng_state=self._rng_state,
            sampling_type=SamplingType.GROUP,
            include_within_group_separation=True,
            ignore_constraints=self.ignore_constraints,
        )
