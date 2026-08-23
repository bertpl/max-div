# Solver Scaling — Memory

The **largest n within memory** is the largest problem size a configuration handles within the memory budget `M_max` — 32 GB of machine memory in use above the level right before the solver process starts, so every process a solver spawns is counted, shared memory is counted once, and the input vectors are included. It is derived from the same runs as the [time page](time.md): a configuration whose size sweep ended by crossing the cap is bracketed at its last completed size, and every other configuration is extrapolated — a constrained fit of its process's recorded memory footprint against `n` over its largest completed sizes, read off at the cap. Configurations that spawn worker processes are excluded from extrapolation (their recorded footprints miss the workers; a solver's memory-bound size is reached by its single-process configurations, since workers only add memory). The [measurement protocol](protocol.md) defines the measurement and the fit.

*Measured with max-div v0.14.2. Only the smoke subset — max-div, RDKit MaxMinPicker, and DPPy — is measured; the remaining solver configurations are not yet.*

![Memory footprint against problem size, per solver configuration, with fitted growth curves](images/scaling_memory.webp)

Dots are recorded footprints; dashed curves are each configuration's fitted growth, drawn up to the memory cap. A flat left end is the solver process's fixed startup cost, which is why only the largest completed sizes feed the fit.

--8<-- "generated/scaling_memory.md"

## Per-configuration fits

The combined chart's log scale flattens most series, so each fitted configuration is shown again on its own adaptive linear scale — just its footprints and fitted curve, with the fitted coefficients and `R²`.

--8<-- "generated/scaling_memory_fits.md"
