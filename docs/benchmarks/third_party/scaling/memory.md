# Solver Scaling — Memory

The **largest n within memory** is the largest problem size a configuration handles within the memory budget `M_max`. The values below come from the memory sweep of the [measurement protocol](protocol.md) (section IV.B), which also defines the budget, the fit, and its trust conditions.

![Memory footprint against problem size, per solver configuration, with fitted growth curves](images/scaling_memory.webp)

Dots are recorded footprints; dashed curves are each configuration's fitted growth, drawn up to the memory cap. A flat left end is the solver process's fixed startup cost, which the fit's intercept absorbs.

--8<-- "generated/scaling_memory.md"

## Per-configuration fits

The combined chart's log scale flattens most series, so each fitted configuration is shown again on its own adaptive linear scale — its footprints and fitted curve, with the fitted coefficients and `R²`. Click a chart for the full-size version.

--8<-- "generated/scaling_memory_fits.md"
