import numba
import numpy as np
from numpy.typing import NDArray

from max_div.internal.math.modify_p_selectivity import modify_p_selectivity
from max_div.sampling.con import randint_constrained
from max_div.sampling.uncon import randint_numba
from max_div.solver._solver_state import SolverState

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
    - constrained (bool): If `True`, respects problem constraints during initialization.
                          If `False`, constraints are ignored. (default: `True`)
    - b (int): Number of batches to sample (must be > 1).
                     -> If e.g. k=100 and b=5, each batch samples 20 items.
                     -> If k is not an exact multiple of b, the first batches will be slightly larger.

    Time Complexity:
       - without constraints: ~O(bn)
       - with constraints:    ~O(kn + bn)
    """

    __MODIFY_P_METHOD: np.int32 = np.int32(20)  # method using fast_pow_f32(p[i], t)
    __SAMPLE_EAGER: bool = True  # always use eager sampling for this case

    def __init__(self, b: int, constrained: bool = True):
        """
        Constructor for InitRandomBatched class.
        :param b: (int) Number of batches to sample (must be > 1).
        :param constrained: (bool, default=True) If `True`, respects problem constraints during initialization,
                                                 If `False`, constraints are ignored.
        """

        # --- parameter validation ----
        if b <= 1:
            raise ValueError("InitRandomBatched requires b > 1; for b=1 use InitRandomOneShot instead.")

        # --- init --------------------
        super().__init__()
        self.b = b
        self.constrained = constrained

    def initialize(self, state: SolverState):
        # --- init ------------------------------

        # settings
        k = state.k
        b = min(np.int32(self.b), k)  # number of batches cannot exceed k

        # tracking variables
        k_done = 0
        k_remaining = k

        # --- main loop -------------------------
        batch_sizes = _get_batch_sizes(k, b)
        for i_batch, batch_size in enumerate(batch_sizes):
            # --- construct p array ---
            if i_batch == 0:
                # first batch:
                #   - use global separation only
                #   - no selectivity modification
                p = state.global_separation_array
            else:
                # later batches:
                #   - use combined separation (wrt selection so far + global)
                #   - modify selectivity based on progress
                p = state.global_separation_array[state.not_selected_index_array]  # this creates a new array
                p += state.not_selected_separation_array  # so we can add in-place

                modifier = min(0.9, k_done / k)  # proportional to progress; cap at 0.9 to avoid extremes
                modify_p_selectivity(
                    p=p,
                    modifier=np.float32(modifier),
                    method=self.__MODIFY_P_METHOD,
                    p_out=p,  # in-place
                )

            # --- sample ---
            if self.constrained and state.has_constraints:
                # NOTE: A) at this point, 'p' is of size 'n_not_selected', hence potentially smaller than 'n'
                #       B) also, con_indices refers to 'original' indices in [0,n) (not to 'not selected' indices)
                #       --> Since these 2 properties are not compatible, we need to fix either 'p' or 'con_indices'.
                #       --> `con_indices` is complicated to rewrite in terms of 'not selected' indices,
                #           hence we extend 'p' to size 'n' and set the `i_forbidden` argument of `randint_constrained`
                #           to exclude already selected indices.
                p_full = np.zeros(state.n, dtype=np.float32)
                p_full[state.not_selected_index_array] = p
                samples = randint_constrained(
                    n=state.n,
                    k=batch_size,
                    con_values=state.con_values,
                    con_indices=state.con_indices,
                    p=p_full,
                    seed=self.next_seed(),
                    eager=self.__SAMPLE_EAGER,
                    k_context=k_remaining,
                    i_forbidden=state.selected_index_array,
                )
            else:
                # NOTE: i_samples are indices into state.not_selected_index_array
                i_samples = randint_numba(
                    n=np.int32(p.size),
                    k=batch_size,
                    replace=False,
                    p=p,
                    seed=self.next_seed(),
                )
                samples = state.not_selected_index_array[i_samples]

            # --- update state ---
            for sample in samples:
                state.add(sample)

            # --- update tracking variables ---
            k_done += batch_size
            k_remaining -= batch_size


# =================================================================================================
#  Initial helpers
# =================================================================================================
@numba.njit(numba.types.int32[:](numba.types.int32, numba.types.int32))
def _get_batch_sizes(k: np.int32, b: np.int32) -> NDArray[np.int32]:
    """
    Get sizes of each batch when splitting k items into b batches.
    If k is not an exact multiple of b, the first batches will be slightly larger.

    Example:
        k=10, b=3  ->  [4, 3, 3]
        k=100, b=6 ->  [17, 17, 17, 17, 16, 16]
    """
    batch_sizes = np.zeros(b, dtype=np.int32)
    for i in range(k):
        batch_sizes[i % b] += 1

    return batch_sizes
