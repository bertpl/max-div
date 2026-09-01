import numba
import numpy as np
from numpy.typing import NDArray

from max_div._core._random import P_UNIFORM, randint
from max_div._core.metrics import DiversityContributionFamily, DiversityMetric
from max_div._core.metrics._distance import DISTANCE_STORE_TYPE, DistanceStore, get_distance
from max_div._core.solver._solver_state import SolverState

from ._base import InitializationStrategy


@numba.njit(
    numba.void(numba.int32[:], numba.float32[:], numba.int64, numba.int64),
    cache=True,
    fastmath={"reassoc", "contract"},
)
def _move_top_to_front(
    cand_idx: NDArray[np.int32], cand_val: NDArray[np.float32], n_top: np.int64, n_remaining: np.int64
) -> None:
    """Reorder the first `n_top` pool slots to hold the highest values among the first `n_remaining`, descending."""
    for s in range(n_top):
        best = s
        for c in range(s + 1, n_remaining):
            if cand_val[c] > cand_val[best]:
                best = c
        if best != s:
            cand_val[s], cand_val[best] = cand_val[best], cand_val[s]
            cand_idx[s], cand_idx[best] = cand_idx[best], cand_idx[s]


@numba.njit(
    numba.int64(
        numba.int32[:],
        numba.float32[:],
        numba.int32,
        numba.float32,
        numba.int64,
        DISTANCE_STORE_TYPE,
        numba.uint64[:],
        numba.int32[:],
    ),
    cache=True,
    fastmath={"reassoc", "contract"},
)
def _draw_round(
    cand_idx: NDArray[np.int32],
    cand_val: NDArray[np.float32],
    top_k: np.int32,
    alpha: np.float32,
    b_target: np.int64,
    store: DistanceStore,
    rng_state: NDArray[np.uint64],
    out_batch: NDArray[np.int32],
) -> np.int64:
    """Draw up to `b_target` items from the candidate pool into `out_batch`; return the count drawn.

    `alpha` sets one acceptance rule, applied twice: a candidate may be drawn while its value is
    at least `alpha` times the pool's current best, and the round ends once even that best falls
    below `alpha` times the round's opening best. Each draw samples uniformly among the acceptable
    candidates within the top `top_k`, then refreshes the remaining pool against the drawn item, so
    every drawn item's value is exact wrt the full selection at its draw moment.

    Filtering the draw window matters because refreshes can leave a just-drawn item's neighbors far
    down the window: drawing one of those would put a near-duplicate pair in the selection, which a
    separation objective scores by its worst pair.
    """
    count = len(cand_idx)
    p_uniform = np.zeros(0, dtype=np.float32)  # module globals freeze to readonly inside njit
    v_round_open = np.float32(-1.0)
    n_drawn = np.int64(0)
    for bi in range(b_target):
        k_eff = min(np.int64(top_k), np.int64(count - bi))
        if k_eff <= 0:
            break
        _move_top_to_front(cand_idx, cand_val, k_eff, np.int64(count - bi))
        if v_round_open < 0:
            v_round_open = cand_val[0]
        elif cand_val[0] < alpha * v_round_open:
            break
        # the front k_eff are sorted descending, so acceptable candidates form the leading run
        n_acceptable = np.int64(1)
        while n_acceptable < k_eff and cand_val[n_acceptable] >= alpha * cand_val[0]:
            n_acceptable += 1
        # a single candidate needs no draw, matching `InitFarthestPoint`'s argmax path
        pick_pos = (
            0 if n_acceptable == 1 else randint(np.int32(n_acceptable), np.int32(1), False, p_uniform, rng_state)[0]
        )
        x = cand_idx[pick_pos]
        out_batch[n_drawn] = x
        n_drawn += 1
        # remove the drawn candidate (swap in the last remaining one), then refresh the rest
        last = count - bi - 1
        cand_idx[pick_pos] = cand_idx[last]
        cand_val[pick_pos] = cand_val[last]
        for c in range(count - bi - 1):
            cand_val[c] = min(cand_val[c], get_distance(store, x, cand_idx[c]))
    return n_drawn


class InitFarthestPointBatched(InitializationStrategy):
    """Initialize by farthest-point sampling in self-sizing rounds, not one pick at a time.

    Per round: the top `batch_max * top_k` not-selected items by diversity contribution form a
    candidate pool; each draw samples among the top `top_k` candidates whose value is still within
    `alpha` of the pool's best, refreshes the remaining pool against the drawn item, and the round
    ends when the pool's best remaining value falls below `alpha` times the round's opening best.
    One round is one `get_next_samples` call, so the state applies each batch in a single tracker
    pass — several times faster than `InitFarthestPoint` at large n, while every drawn item's
    contribution is exact at its draw moment.

    The construction heuristics — most notably the round-stop rule — are tailored to
    separation-family diversity metrics, whose contributions only decrease within a round; other
    metric families are rejected when the solver is built. Constraints are ignored by design,
    like `InitFarthestPoint` — whose picks and randomness this strategy does not reproduce.

    Parameters:
    - top_k (int): every draw samples uniformly among the `top_k` best remaining candidates by
                   current value — the meaning it has on `InitFarthestPoint`; 1 keeps the exact
                   greedy pick. (default: 8)
    - alpha (float): acceptance threshold in (0, 1] — a candidate is drawable while its value is
                     at least `alpha` times the pool's current best, and the round ends once that
                     best falls below `alpha` times the round's opening best. Higher values
                     approach the per-pick construction at the cost of smaller rounds.
                     (default: 0.9)
    - batch_max (int): upper bound on items drawn per round; the pool holds
                       `batch_max * top_k` candidates. (default: 256)
    """

    def __init__(self, top_k: int = 8, alpha: float = 0.9, batch_max: int = 256) -> None:
        """Create the strategy.

        Raises:
            ValueError: If `top_k` or `batch_max` is below 1, or `alpha` outside (0, 1].
        """
        super().__init__()
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if batch_max < 1:
            raise ValueError(f"batch_max must be >= 1, got {batch_max}")
        self._top_k = top_k
        self._alpha = alpha
        self._batch_max = batch_max

    def validate_diversity_metric(self, diversity_metric: DiversityMetric) -> None:
        """Reject metric families the round heuristics are not tailored to."""
        if diversity_metric.contribution_family != DiversityContributionFamily.SEPARATION:
            raise ValueError(
                f"InitFarthestPointBatched does not support diversity metric {diversity_metric}: "
                "the heuristics of the algorithm are tailored to separation-based diversity metrics. "
                "Use InitializationStrategy.farthest_point() instead."
            )

    def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
        if state.n_selected == 0:
            return randint(n=state.n, k=np.int32(1), replace=False, p=P_UNIFORM, rng_state=self._rng_state)
        b_target = min(self._batch_max, int(k_remaining))
        cand_idx, cand_val = state.top_not_selected_contributions(self._batch_max * self._top_k)
        out_batch = np.empty(b_target, dtype=np.int32)
        n_drawn = _draw_round(
            cand_idx,
            cand_val,
            np.int32(self._top_k),
            np.float32(self._alpha),
            np.int64(b_target),
            state.distance_store,
            self._rng_state,
            out_batch,
        )
        return out_batch[:n_drawn]
