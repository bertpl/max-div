"""Construction of the add-candidate pool for swap-based optimization strategies.

Every swap iteration needs a pool of candidate items to sample additions from. The natural pool is
all non-selected items (n-k items), but passing all of them makes each sampling call cost O(n),
which dominates early-run time on large problems. This module centralizes the alternative: each
iteration, the pool is reduced to a uniform random subset of min(cap, n-k) items, where

    cap = CAP_INITIAL + CAP_GROWTH_PER_ITER * iteration

The cap thus starts small and grows linearly with the strategy's iteration count, saturating at
the full n-k pool. Iteration-indexing is deliberate: runs that converge (tens of thousands of
iterations) reach the full pool while meaningful iterations remain — preserving end-of-run
quality, which needs the best few candidates once improving swaps become rare — while runs whose
budget affords only a few hundred iterations at very large n stay capped for their whole life and
never pay for exploration breadth they cannot use.
"""

import numpy as np
from numpy.typing import NDArray

from max_div._core._random import P_UNIFORM, choice
from max_div._core.solver._solver_state import SolverState

# Cap on the add-candidate pool: CAP_INITIAL + CAP_GROWTH_PER_ITER * iteration, saturating at the
# full pool. The initial value sits on the measured plateau of short-budget quality gains (flat
# roughly between 100 and 1000, with smaller values slightly ahead); the growth rate is chosen so
# runs at sizes where convergence is reachable uncap around mid-run.
CAP_INITIAL = 250
CAP_GROWTH_PER_ITER = 1


def candidate_samples_to_add(
    state: SolverState,
    iteration: int,
    rng_state: NDArray[np.uint64],
) -> NDArray[np.int32]:
    """Return the add-candidate pool for one swap: all non-selected items, under a growing cap.

    When the pool fits under the cap, it is returned as-is and **no RNG is consumed** — small
    problems and late-run large problems behave exactly as an uncapped pool would. Above the cap,
    a uniform without-replacement subsample of cap size is drawn from the strategy's RNG.

    :param state: (SolverState) current solver state; supplies the non-selected items.
    :param iteration: (int) the strategy's iteration counter, driving the cap growth.
    :param rng_state: (NDArray[np.uint64]) the strategy's RNG state (updated in place on draw).
    """
    pool = state.not_selected_index_array
    cap = CAP_INITIAL + CAP_GROWTH_PER_ITER * iteration
    if pool.size <= cap:
        return pool
    return choice(pool, np.int32(cap), False, P_UNIFORM, rng_state)
