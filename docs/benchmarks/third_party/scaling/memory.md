# Solver Scaling — Memory

The **largest n within memory** is the largest problem size a configuration handles within the memory budget `M_max` — 32 GB of machine memory in use above the level right before the solver process starts, so every process a solver spawns is counted, shared memory is counted once, and the input vectors are included. It comes from a dedicated memory sweep, independent of the [time page](time.md)'s: each grid size runs for up to one minute as an observation window — completion is not required — while the runner records the memory footprint reached. A sweep that crosses the cap, or hits a size the solver outright fails at, is bracketed at its last size; otherwise it walks upward until the recorded footprints carry a trustworthy trend, and the fitted growth curve is read off at the cap. Configurations that spawn worker processes are not measured (their recorded footprints would miss the workers; a solver's memory-bound size is reached by its single-process configurations, since workers only add memory). The [measurement protocol](protocol.md) defines the sweep, the fit, and its trust conditions.

*Measured with max-div v0.14.2. Only the smoke subset — max-div, RDKit MaxMinPicker, and DPPy — is measured; the remaining solver configurations are not yet.*

![Memory footprint against problem size, per solver configuration, with fitted growth curves](images/scaling_memory.webp)

Dots are recorded footprints; dashed curves are each configuration's fitted growth, drawn up to the memory cap. A flat left end is the solver process's fixed startup cost, which the fit's intercept absorbs.

--8<-- "generated/scaling_memory.md"

## Per-configuration fits

The combined chart's log scale flattens most series, so each fitted configuration is shown again on its own adaptive linear scale — its footprints and fitted curve, with the fitted coefficients and `R²`. Click a chart for the full-size version; open markers are windows whose footprint had not settled, which do not feed the fit.

--8<-- "generated/scaling_memory_fits.md"
