Each preset is run for a ladder of increasing time budgets, one run per budget point with a fresh seed, so the spread across seeds shows up as scatter alongside the time trend.

Timing is end-to-end: it includes building the solver's distance store, the cost a caller actually pays.

The uncertainty band is a q10-q90 estimate from monotone cubic-spline quantile regression through that scatter, approximating the best of about ten seeded runs (around q90). It covers the informative metric only — constraint score while a constrained problem is infeasible, diversity score otherwise.

A separate black-circle series is the default parallel invocation: the SMART preset on the machine's default set of cooperative workers, a best-of-N reference. It runs only the longer budgets, since a parallel run's start-up cost dominates the shortest, and carries no band because it is one reference series rather than a preset sweep.
