"""Define the head-to-head comparison protocol's shared constants.

The three comparison tiers share these so that their pages read as one campaign, aligned with the
solver-scaling protocol where the two overlap: the same T_max, the same 1-2-5 grids for budgets
and sizes, the same worker count. Seeds deviate (3 here, 5 there) to fit the three tiers into one
night; the protocol page states that once.
"""

from .budget_series import grid_budget_series

T_MAX_SEC = 60.0  # the budget every quoted result is judged at; equals the scaling protocol's T_max
SEEDS = (0, 1, 2)
N_WORKERS = 12  # the multi-worker series' worker count: the reference machine's performance cores

# max-div runs two budget series per cell: one worker from 1 ms, and N_WORKERS from 1 s — spawning
# the workers costs about a second, so a smaller multi-worker budget would only show start-up.
SINGLE_WORKER_BUDGETS_SEC = grid_budget_series(0.001, T_MAX_SEC)
MULTI_WORKER_BUDGETS_SEC = grid_budget_series(1.0, T_MAX_SEC)

# The budgets the result tables quote: the point where the two series first coincide, and T_max.
QUOTED_BUDGETS_SEC = (1.0, T_MAX_SEC)

# The exact solvers' certification cap: the scaling protocol's extended budget, 15 x T_max.
CERTIFICATION_CAP_SEC = 15 * T_MAX_SEC
