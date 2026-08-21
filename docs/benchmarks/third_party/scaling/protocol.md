# Solver Scaling — Measurement Protocol

This protocol defines how the three problem-size values — the [memory](memory.md), [time](time.md), and [quality](quality.md) sizes — are measured, identically for every tool in the comparison suite. Each value answers a different question about the same tool on the same problem family:

- **Memory** — how large a problem still fits in memory at all?
- **Time** — how large a problem still yields *a valid answer* within the reference budget?
- **Quality** — how large a problem still yields *a good answer* within the reference budget?

## Shared context

Every value is defined against one pinned context:

- **Problem:** `U1` (the fixed-$d=2$ clustered problem family), sized per the same k→n contract every other published `U1` benchmark uses (n = 10·k). A value at size n is a claim about the same problem every other published `U1` number describes.
- **Candidate sizes:** the 1-2-5 grid in n (100, 200, 500, 1000, ...), floor 100. A published value is always a grid value.
- **No result-binding size cap.** Each value terminates by its own mechanism (below). The only bound is operational and derived from the protocol's own memory budget: instances are generated only while the raw float32 vectors array fits 32 GB (n·d·4 bytes ≤ 32 GB; at d=2, n ≈ 4×10⁹) — beyond that, the input itself cannot exist under the protocol.
- **Machine:** the named benchmark machine.
- **Memory budget:** 32 GB peak, enforced during runs by an RSS watchdog; a watchdog kill is a fail and costs no more than the kill.
- **Timing:** end-to-end from raw vectors — a tool's distance/setup work is part of its cost.
- **Quality metric:** the max-min objective — the smallest pairwise distance in the selection — chosen for maximum tool compatibility: no other objective is supported by as many suite tools (it is not universal — DPPy pursues no diversity objective, apricot-select pursues max-sum). Tools that optimize a different objective are still scored on max-min, with the mismatch disclosed; a tool that clears the 90% line at no size renders as a dash with a footnote. At d=2 the harness scores selections with a spatial index (min separation via nearest-neighbor queries, O(k log k)), so evaluation stays cheap at any generatable size.

## Constants

| Constant | Value | Reading |
|---|---|---|
| T (reference budget) | 60 s | "within one minute" — a human unit, not a tuned number |
| M (long-budget multiple) | 15 | long-budget reference runs get M·T = 15 min, "a quarter of an hour" |
| Gap threshold | 0.9 | "closes at least 90% of the random-to-best gap" |
| Q_rand sample | 32 | uniform random size-k selections, median objective |
| Seeds S | 3 / 5 | 3 for deterministic-or-nearly tools, 5 for stochastic ones (max-div) |
| Memory cap | 32 GB | shared by the run watchdog and the memory-size definition |

T and M are the two computability knobs: T bounds every within-T probe, M·T bounds every long-budget reference run.

## The three definitions

- **Memory** — the largest grid size whose peak memory stays within 32 GB in the tool's most memory-efficient configuration. Determined by calibrated extrapolation as the standard method: a per-tool memory model, checked against the tool's documented structure, is fitted to measured peak RSS from runs at smaller sizes (two completed sizes minimum) and read off at the 32 GB crossing, rounded down to the grid. No run is executed at the value to confirm it — at full memory most tools are far too slow to run at all — and because extrapolation is the standard method for every tool, the value carries no special provenance mark.
- **Time** — the largest grid size at which the tool returns a valid size-k selection within T, under the memory cap, on the median seed — in its fastest valid configuration: the fastest standard, user-configurable setting that still produces a valid selection (max-div: purely random initialization, no optimization steps; exact solvers: stop at first feasible solution; one-shot tools: their only mode). Each tool's pinned configuration is listed on the [Time](time.md) page. Expected to land near the memory size for most tools — the binding cost is mandatory per-item overhead, not search.
- **Quality** — the largest grid size (≤ the time size by construction) at which the tool's median quality, in its standard quality configuration (the same configuration the head-to-head pages measure) at budget T, clears `Q_rand + 0.9 · (Q_best − Q_rand)`. `Q_rand` is the median objective over 32 uniform random size-k selections; `Q_best` is defined below.

**Nesting.** No memory → no solution → no good solution: each value can only be at or below the previous, so a tool's row reads as a sequence of tightening claims. The third ≤ the second is structural: a tool that returns a good answer within T in its quality configuration also returns *some* answer within T in its fastest valid configuration.

## The best-known solution

**`Q_best(n)` = the best solution any tool in the suite produces within M·T at size n.**

No tool is excluded by category — only by producing nothing within M·T. The candidate pool is the union of the dedicated long-budget runs (below) and every within-T quality run (T < M·T, so those results are trivially candidates); the final value is taken at analysis time.

**Scheduling rule** — a dedicated M·T run is scheduled wherever it could improve on what the tool's within-T runs can produce:

- **Budget-responsive tools** (max-div with parallel default workers; the exact solvers, whose anytime incumbents improve with budget): every grid size they reach.
- **One-shot deterministic tools**: only the completion band — sizes where the within-T run does not finish but an M·T run might (for an O(nk) tool at fixed k/n, ×15 budget buys roughly ×4 in n: one or two grid steps).
- **Samplers** (DPPy, qc-selector): no dedicated runs — best-of-restarts at M·T would be our wrapper, not the tool's standard usage. Their within-T results still enter the candidate pool.

**Judge independence.** Every long-budget run uses a single pre-registered seed, and the schedule (which tools, which sizes, which seeds) is frozen before any quality run is examined. No run is added or repeated in reaction to candidate results. A single seed is safe as well as clean: parallel best-of-N is internally variance-reduced, and the candidate-pool union floors the best-known solution at the within-T field's best, so an unlucky long-budget run cannot drag it below.

**Stated properties:**

- The best-known solution's long-budget term is usually produced by max-div itself. The pool definition keeps this honest — wherever any competitor beats it within M·T, the competitor sets the reference — and where the exact solvers prove optimality within M·T, the reference is a certified optimum, which also validates the 0.9 line at the small end.
- At sizes where no M·T run betters the within-T field, the per-size winner defines the line it is scored against and passes by construction — the degeneracy the M·T term exists to remove returns locally there.
- A re-measure that improves the best-known solution lowers every tool's quality size retroactively.
- At sizes where every tool scores near-random, the gap collapses and the 90% line is trivially cleared; the quality size degenerates toward the time size there.
- The per-size gap-closure fractions are published in the record table, so readers see margins, not just pass/fail against the 0.9 line.
- The best-known solutions themselves are published as a per-size provenance table — best result obtained, which tool produced it, under what configuration and budget (within-T quality run or M·T run) — on the [Quality](quality.md) page. The "usually max-div's own" property above is thereby a checkable fact, not an assertion.

## Campaign structure — four sequential stages

The measurement campaign splits into four dedicated stages; each stage prunes or feeds the next, and each is independently resumable.

1. **Time measurements.** Per tool: ascend the grid from the floor in the fastest valid configuration, budget T per run, stop at the first fail (monotonicity in n at fixed k/n is assumed). Peak RSS is recorded on every run. Passes are fast by construction; fails cost T. → time values, which bound stage 4's grids.
2. **Memory-model fits.** Desk work, no runs: per tool, fit the memory model to the recorded peaks at the largest sizes that tool completed, checked against its documented memory structure, and read off the 32 GB crossing → memory values. Exception rule: a tool whose memory-optimal configuration differs from its fastest valid one gets a few dedicated small runs in the memory-optimal configuration.
3. **Long-budget reference runs.** The frozen pre-registered schedule per the rule above, each run killed at M·T. Runs before stage 4, so judge independence is visible in the protocol order.
4. **Quality runs.** Per tool: S seeded runs at budget T in the quality configuration, at grid sizes up to its time value; median quality per size → quality values against Q_rand and the best-known-solution pool. A tool clearing the line at no size publishes a dash with a footnote.

**Pass criteria** everywhere use the median seed. All kills (T, M·T, RSS) are hard kills counted as fails.
