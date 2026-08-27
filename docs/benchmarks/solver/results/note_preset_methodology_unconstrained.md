Each preset is run for a series of increasing time budgets, one run per budget point with a fresh seed, so the spread across seeds shows up as scatter alongside the time trend.

Budgets are end-to-end: a run's time budget covers the whole solve, computing the distances and initialization included, so a budget point shows what a caller gets for that much time.

The uncertainty band is a q10-q90 estimate from monotone cubic-spline quantile regression through that scatter, approximating the best of about ten seeded runs (around q90). It covers the diversity score.

A separate black-circle series is the default parallel invocation: the SMART preset on the machine's default set of cooperative workers, shown as a reference for the best result the workers jointly reach. It runs its own budget series and carries no band, because it is one reference series rather than a preset sweep.

The parallel series' shortest budget points sit below the serial presets: an end-to-end budget includes the roughly one second the parallel solver spends spawning its worker processes.
