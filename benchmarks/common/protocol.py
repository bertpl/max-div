"""Define the head-to-head comparison protocol's shared constants.

The three comparison tiers share these so that their pages read as one campaign, aligned with the
solver-scaling protocol where the two overlap: the same T_max, the same 1-2-5 grids for budgets
and sizes, the same worker count. The seed count deviates from the scaling protocol's
`QUALITY_SEEDS` to fit the three tiers into one night; `docs/benchmarks/third_party/scaling/protocol.md`
states that once.
"""

from benchmarks.solver_scaling.grid import EXTENDED_BUDGET_SEC, REFERENCE_BUDGET_SEC

from .budget_series import grid_budget_series

T_MAX_SEC = REFERENCE_BUDGET_SEC  # the budget every quoted result is judged at
SEEDS = (0, 1, 2)
N_WORKERS = 12  # the multi-worker series' worker count: the reference machine's performance cores

# max-div runs two budget series per cell: one worker from 1 ms, and N_WORKERS from 1 s — spawning
# the workers costs about a second, so a smaller multi-worker budget would only show start-up.
SINGLE_WORKER_BUDGETS_SEC = grid_budget_series(0.001, T_MAX_SEC)
MULTI_WORKER_BUDGETS_SEC = grid_budget_series(1.0, T_MAX_SEC)

# How many single-worker solves run side by side: the single-worker series packs across the cores
# the multi-worker series uses one at a time.
SINGLE_WORKER_CONCURRENCY = N_WORKERS

# The budgets the result tables quote: the point where the two series first coincide, and T_max.
QUOTED_BUDGETS_SEC = (1.0, T_MAX_SEC)

# The exact solvers' certification cap: the scaling protocol's extended budget.
CERTIFICATION_CAP_SEC = EXTENDED_BUDGET_SEC
