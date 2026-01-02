import numpy as np
from numpy._typing import NDArray

from max_div.internal.math.modify_p_selectivity import exponential_selectivity
from max_div.sampling.con import randint_constrained
from max_div.sampling.uncon import randint_numba
from max_div.solver._solver_state import SolverState

from ._base import InitializationStrategy


class InitEager(InitializationStrategy):
    """
    Initialize by adding 1 sample at a time (`k` iterations), where each for each added sample we...
      - start from a set of `nc` randomly sampled candidates
      - we take the candidate which results in the highest score

    After each iteration, the SolverState updates distances and separations, influencing sampling probabilities
      of the next batch of candidates.

    When sampling a batch, we use probabilities p[i] ~= (separation of i wrt already selected items)
                                                                            + (separation of i wrt all items)

    This drives each batch to be sampled from elements that are both well-separated from the selection so far, to
      promote diversity, and also well-separated from each other, to avoid samples within a batch that are far from
      the selection but close to each other.

    As we progress through the batches, selectivity of p[i] is modified with modifier = #sampled / #to_sample.
        (see modify_p_selectivity for details)

    Suggested use: when highest quality results are desired and time permits.  This method is computationally more
                   expensive than most other methods.

    Parameters:
    - nc (int): Number of candidates (>1) to sample in each iteration, the best of which will be added.
    - ignore_constraints (bool): If `False`, respects problem constraints during initialization, if present.
                                 If `True`, constraints are ignored. (default: `False`)

    Time Complexity:
       - without constraints: ~O(nc * k * n)
       - with constraints:    ~O(nc * k * n)
    """

    __MODIFY_P_METHOD: np.int32 = np.int32(20)  # method using fast_pow_f32(p[i], t)
    __SAMPLE_EAGER: bool = True  # always use eager sampling for this case

    def __init__(self, nc: int, ignore_constraints: bool = False):
        """
        Constructor for InitRandomBatched class.
        :param nc: (int) Number of candidates to sample in each iteration.
        :param ignore_constraints: (bool, default=False)
                        If `False`, respects problem constraints during initialization, if present.
                        If `True`, constraints are ignored when sampling candidates and when comparing scores.
        """

        # --- parameter validation --------------
        if nc <= 1:
            raise ValueError("InitEager requires nc > 1; for nc=1 use InitRandomBatched with b=k instead.")

        # --- init ------------------------------
        name = f"InitEager(nc={nc}" + (",uncon)" if ignore_constraints else ")")
        super().__init__(name)
        self.nc = np.int32(nc)
        self.ignore_constraints = ignore_constraints

    def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
        # --- initialize ------------------------
        k = state.k
        n_selected = state.n_selected
        nc = min(self.nc, state.n - n_selected)  # number of candidates cannot be larger than remaining items

        # --- construct p array -----------------
        if n_selected == 0:
            # first batch of candidates:
            #   - use global separation only
            #   - no selectivity modification
            p = state.global_separation_array
        else:
            # later batches:
            #   - use separation of not-selected wrt selected items
            #       --> as opposed to RandomBatched, we ignore global separation here; we only add 1 item at a time,
            #           so we don't run the risk of adding multiple similar items in one batch as in RandomBatched.
            #   - modify selectivity based on progress
            p = state.not_selected_separation_array

            modifier = min(0.9, n_selected / k)  # proportional to progress; cap at 0.9 to avoid extremes
            exponential_selectivity(
                p_in=p,
                p_out=p,  # in-place
                modifier=np.float32(modifier),
            )

        # --- sample nc times ---
        best_sample: np.int32 = np.int32(-1)
        best_score_tuple: tuple | None = None

        for j in range(nc):
            # --- sample once ---
            if state.has_constraints and (not self.ignore_constraints):
                # NOTE: A) at this point, 'p' is of size 'n_not_selected', hence potentially smaller than 'n'
                #       B) also, con_indices refers to 'original' indices in [0,n) (not to 'not selected' indices)
                #       --> Since these 2 properties are not compatible, we need to fix either 'p' or 'con_indices'.
                #       --> `con_indices` is complicated to rewrite in terms of 'not selected' indices,
                #           hence we extend 'p' to size 'n' and set the `i_forbidden` argument of `randint_constrained`
                #           to exclude already selected indices.
                p_full = np.zeros(state.n, dtype=np.float32)
                p_full[state.not_selected_index_array] = p
                sample = randint_constrained(
                    n=state.n,
                    k=np.int32(1),
                    con_values=state.con_values,
                    con_indices=state.con_indices,
                    p=p_full,
                    seed=self.next_seed(),
                    eager=self.__SAMPLE_EAGER,
                    k_context=k_remaining,
                    i_forbidden=state.selected_index_array,
                )[0]
            else:
                # NOTE: i_samples are indices into state.not_selected_index_array
                i_sample = randint_numba(
                    n=np.int32(p.size),
                    k=np.int32(1),
                    replace=False,
                    p=p,
                    seed=self.next_seed(),
                )[0]
                sample = state.not_selected_index_array[i_sample]

            # --- try out sample --

            # take snapshot -> add and remember score -> revert
            state.set_snapshot()
            state.add(sample)
            score = state.score
            state.restore_snapshot()

            # see if sample is better
            if self.ignore_constraints:
                score_tuple = score.as_tuple(soft=1.0)  # ignore constraint score
            else:
                score_tuple = score.as_tuple()  # consider full score, including constraints

            if (best_score_tuple is None) or (score_tuple > best_score_tuple):
                best_score_tuple = score_tuple
                best_sample = sample

        # --- return best sample ----------------
        return np.array([best_sample], dtype=np.int32)
