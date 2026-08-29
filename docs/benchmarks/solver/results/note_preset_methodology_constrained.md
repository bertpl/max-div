Each preset is run for a series of increasing time budgets, one run per budget point with a fresh seed, so the spread across seeds shows up as scatter alongside the time trend.

Timing is end-to-end: it includes computing the distances, the cost a caller actually pays.

The uncertainty band is a q10-q90 estimate from monotone cubic-spline quantile regression through that scatter, approximating the best of about ten seeded runs (around q90). It covers the informative metric only — constraint score while a constrained problem is infeasible, diversity score otherwise.

A separate black-circle series is a parallel run with the default parallel settings, shown as a reference for the best result the workers jointly reach. It omits the lowest budgets, where the fixed start-up cost of the parallel workers dominates.
