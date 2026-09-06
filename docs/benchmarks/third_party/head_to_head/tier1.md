# Head-to-Head — vs. Exact Solvers

## I. Goal and reading guide

How close does `max-div` get to a *proven* optimum, and what does proving one cost? Three exact solvers — CP-SAT, SCIP and HiGHS — serve as the optimality reference on the two smallest built-in [benchmark problems](../../solver/test_problems.md). They are not competitors on speed: an exact solver's result counts only where it certified optimality within its cap, and the sizes no solver certifies are not judged.

Every chart on this page reads the same way:

- the **black curves** are `max-div`: solid with one worker, dashed with 12; the band around each is the min/max over seeds, the line the mean;
- the **dotted horizontal line** is the certified optimum;
- each **cross** is one exact solver at (time to certify, optimum), in that solver's color.

The distance between a black curve and the dotted line is the gap to the optimum at that budget; the position of a cross says what certifying it cost. The gaps and proof times are tabulated on the [tables page](tier1_tables.md).

## II. Protocol

The tier follows the [solver-scaling protocol](../scaling/protocol.md): the same T_max of 60 s, the same 1-2-5 size grid, the same [solver configurations](../scaling/solver_configs.md), the same reference machine (Apple M3 Max, 12 performance cores). It runs 3 seeds per cell instead of the scaling protocol's 5, so that the three head-to-head tiers fit one night of measurement.

### II.A. Entrants and configurations

- **CP-SAT**, the `optimal` configuration: 12 portfolio workers. Certifies the max-min objective through its max-min model and the mean and geomean objectives through the nearest-neighbor assignment model — it is the only solver strong enough on that model to certify any grid size.
- **SCIP** and **HiGHS**, their `optimal` configurations: the big-M max-min MIP, SCIP single-threaded, HiGHS with parallel branch-and-bound. Max-min only.
- **max-div**, `DEFAULT` preset, `L2` distance, run in two budget series per cell:
    - one worker, budgets 1 ms → 60 s on the 1-2-5 grid (15 points);
    - 12 workers with the default dynamic grouping, budgets 1 s → 60 s (6 points) — spawning the workers costs about a second, so smaller budgets would only show start-up.

Every `max-div` point is one independent solve per budget and seed, timed end to end around the call; the charts plot *measured* wall-clock, never the nominal budget. Each exact certification runs in its own process with a cap of 900 s, the scaling protocol's extended budget; a solve that does not certify within the cap ends that solver's column.

### II.B. Problems, sizes and objectives

- **U1** (unconstrained) and **C1** (constrained), k = n/10, on the 1-2-5 grid from n = 20 upward until certification stops.
- **Minimum separation** on all three solvers.
- **Mean** and **geomean separation** on CP-SAT only; SCIP and HiGHS certify the nearest-neighbor assignment model less far and are left out of those columns.

### II.C. Budgets and seeds

- exact solvers: up to 900 s per instance to certify;
- `max-div`: both budget series above, 3 seeds per cell; the tables quote the median over seeds at 1 s and 60 s.

## III. Minimum separation

Where certification stopped, per solver: CP-SAT certifies the max-min optimum up to n = 1,000 on U1 (13 s) and n = 500 on C1 (26 s); HiGHS up to n = 200 on both problems (30–40 s); SCIP up to n = 100 on both (10–160 s). Each column ends at the next grid size, which that solver did not certify within 900 s. Near its limit, every solver's proof time climbs by at least a factor 5 per grid step.

How `max-div` fares against those optima, median over seeds:

- **up to n = 100** both series reach the optimum within 20 ms on C1; on U1 at n = 100 the single worker sits 3.7 % short until 2 s, the 12-worker series is on the optimum from its first budget;
- **from n = 200** a gap remains at 60 s: 5–11 % with one worker, 0–8 % with 12. The 12-worker series is ahead of, or level with, the single worker at every budget both run, and its lead grows with size;
- the curves **improve in steps, not smoothly**: on U1 at n = 200 the single worker's median stays at 12.8 % from 20 ms to 60 s while 12 workers reach the optimum by 2 s; on C1 at n = 200 both series plateau, at 4.8 % and 1.6 %. More workers close such a gap where more time on one worker does not.

### III.A. U1

--8<-- "docs/benchmarks/third_party/head_to_head/results/tier1_charts_min_separation_u1.md"

### III.B. C1

--8<-- "docs/benchmarks/third_party/head_to_head/results/tier1_charts_min_separation_c1.md"

## IV. Mean separation

CP-SAT certifies the nearest-neighbor assignment model at n = 20 and n = 50 on both problems, within seconds; n = 100 is not certified within 900 s. `max-div` reaches every certified mean-separation optimum within 100 ms in both series.

--8<-- "docs/benchmarks/third_party/head_to_head/results/tier1_gallery_mean_separation.md"

## V. Geomean separation

The same two sizes certify on both problems, and C1 at n = 100 certifies as well — in 863 s, just inside the cap — the only n = 100 cell any solver proves on the assignment model. `max-div` reaches every certified geomean optimum within 100 ms, except C1 at n = 100, where the single worker is 0.1 % short at 1 s and on the optimum from 5 s.

--8<-- "docs/benchmarks/third_party/head_to_head/results/tier1_gallery_geomean_separation.md"

## VI. Tables

The [tables page](tier1_tables.md) lists, per objective, the certified optimum of every judged cell with `max-div`'s median gap at 1 s and 60 s for both series, and per solver where certification stopped.
