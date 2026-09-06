# Head-to-Head — vs. Exact Solvers

## I. Goal and reading guide

Three exact solvers — CP-SAT, SCIP and HiGHS — serve as the optimality reference on the two smallest built-in [benchmark problems](../../solver/test_problems.md): the page measures how close `max-div` gets to their proven optima, and what each proof cost. The exact solvers are not competitors on speed. A solver's result counts only where it certified optimality within its cap, and the sizes no solver certifies are not judged.

Every chart on this page reads the same way:

- the **black curves** are `max-div`: solid with one worker, dashed with 12; the band around each is the min/max over seeds, the line the mean;
- the **dotted horizontal line** is the certified optimum;
- each **cross** is one exact solver at (time to certify, optimum), in that solver's color.

The distance between a black curve and the dotted line is the gap to the optimum at that budget; the position of a cross says what certifying it cost.

## II. Protocol

The tier follows the [solver-scaling protocol](../scaling/protocol.md): its time budget, size grid, reference machine, and its [solver configurations](../scaling/solver_configs.md) for the exact solvers. It runs 3 seeds per cell, not 5 ([why](../scaling/protocol.md#iii-fundamental-constants-invariants)).

### II.A. Entrants

- **CP-SAT** certifies the max-min objective through its max-min model, and the mean and geomean objectives through the nearest-neighbor assignment model.
- **SCIP** and **HiGHS** certify max-min through the big-M MIP. On the nearest-neighbor assignment model they certify less far than CP-SAT, so they are left out of the mean and geomean cells.
- **max-div**, `DEFAULT` preset, `L2` distance, runs two budget series per cell:
    - one worker, budgets 1 ms → 60 s on the 1-2-5 grid (15 points);
    - 12 workers with the default dynamic grouping, budgets 1 s → 60 s (6 points) — spawning the workers costs about a second, so smaller budgets would only show start-up.

Every `max-div` point is one independent solve per budget and seed, timed end to end around the call; the charts plot *measured* wall-clock, never the nominal budget. Each exact certification runs in its own process with a cap of 900 s, the scaling protocol's extended budget; a solve that does not certify within the cap ends that solver's column.

### II.B. Problems, sizes and objectives

- **U1** (unconstrained) and **C1** (constrained), k = n/10, on the 1-2-5 grid from n = 20 upward until certification stops.
- **Minimum separation** on all three solvers; **mean** and **geomean separation** on CP-SAT only.

## III. Minimum separation

Certification stops at a different size per solver, and the size after it is not certified within 900 s:

- CP-SAT certifies up to n = 1,000 on U1 (13 s) and n = 500 on C1 (26 s);
- HiGHS up to n = 200 on both problems (30–40 s);
- SCIP up to n = 100 on both (10–160 s).

Near its limit, every solver's proof time climbs by at least a factor 5 per grid step. Against those optima, `max-div` (median over seeds) shows three patterns:

- **up to n = 100** both series reach the optimum within 20 ms on C1; on U1 at n = 100 the single worker sits 3.7 % short until 2 s, the 12-worker series is on the optimum from its first budget;
- **from n = 200** a gap remains at 60 s: 5–11 % with one worker, 0–8 % with 12. The 12-worker series is ahead of, or level with, the single worker at every budget both run, and its lead grows with size;
- the curves **improve in steps, not smoothly**: on U1 at n = 200 the single worker's median stays at 12.8 % from 20 ms to 60 s while 12 workers reach the optimum by 2 s. More workers close such a gap where more time on one worker does not.

### III.A. U1

--8<-- "docs/benchmarks/third_party/head_to_head/results/tier1_charts_min_separation_u1.md"

### III.B. C1

--8<-- "docs/benchmarks/third_party/head_to_head/results/tier1_charts_min_separation_c1.md"

## IV. Mean separation

CP-SAT certifies the nearest-neighbor assignment model at n = 20 and n = 50 on both problems, within seconds; n = 100 is not certified within 900 s. `max-div` reaches every certified mean-separation optimum within 100 ms in both series.

--8<-- "docs/benchmarks/third_party/head_to_head/results/tier1_gallery_mean_separation.md"

## V. Geomean separation

n = 20 and n = 50 certify on both problems, and C1 at n = 100 certifies as well — in 863 s, just inside the cap — the only n = 100 cell any solver proves on the assignment model. `max-div` reaches every certified geomean optimum within 100 ms, except C1 at n = 100, where the single worker is 0.1 % short at 1 s and on the optimum from 5 s.

--8<-- "docs/benchmarks/third_party/head_to_head/results/tier1_gallery_geomean_separation.md"

## VI. Tables

The [tables page](tier1_tables.md) holds the gap tables per objective and the certification table per solver.
