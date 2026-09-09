# Head-to-Head — vs. Python Heuristics

## I. Goal and reading guide

The [scaling pages](../scaling/protocol.md) say how large a problem each Python subset-selection tool handles, and at what quality. This page adds what they cannot show: how `max-div`'s quality evolves with its time budget next to the fixed answer of each one-shot tool. Two questions per size:

- at what budget does `max-div` pass the best one-shot tool?
- how do `max-div`'s single-worker and 12-worker series compare?

Every chart reads the same way:

- the **black curves** are `max-div`: solid with one worker, dashed with 12; the band around each is the min/max over seeds, the line the mean;
- each **dot** is one one-shot tool at its own measured time and quality (mean over seeds);
- the **dotted horizontal line** is the best one-shot result at that size — where a black curve crosses it is the budget at which `max-div` overtakes.

## II. Protocol

The tier reuses the time budget and reference machine of the [solver-scaling protocol](../scaling/protocol.md), and the [solver configurations](../scaling/solver_configs.md) measured there for the one-shot tools. The tier runs 3 seeds per cell, not 5 ([why](../scaling/protocol.md#iii-fundamental-constants-invariants)).

- **Problem**: U1 — the scaling pages' problem, so both describe the same instances — at n = 100, 1,000, 10,000 and 100,000, k = n/10. Constrained problems are not on this page: no one-shot tool in the registry handles the constraints the harder problems carry.
- **Objective**: minimum separation under the `L2` distance, scored identically for every tool by `max-div`'s own metric code. Tools that optimize a different objective enter as different-objective references, not as dispersion competitors ([solver configurations](../scaling/solver_configs.md)):
    - `apricot-select` (facility location);
    - `kmedoids` (representativeness);
    - `DPPy` (a determinantal sample);
    - the max-sum picker of `qc-selector`.
- **Entrants**: every non-exact registry tool, at the sizes its scaling time limit covers; one run per seed where the tool is seeded. A tool's time includes any conversion it needs. Exact solvers are compared on the [exact-solver tier](tier1.md), not here.
- **max-div**: `DEFAULT` preset with lazy distance storage, one independent solve per budget and seed, one solve at a time, timed end to end around the call; charts plot *measured* wall-clock, never the nominal budget. Two budget series per size:
    - one worker, 1 ms to 60 s;
    - 12 workers with the default dynamic grouping, 1 s to 60 s.

    A budget is charted only when it exceeds the measured time of the previous charted budget, so the small budgets that all end at the set-up cost appear once.

    Lazy distance storage computes distances on demand instead of building a distance store first, and this tier runs it at every size:

    - this page covers the largest problem sizes of the three tiers;
    - at the largest sizes, building the store takes an unreasonable share of the time budget, and a knowledgeable user would choose lazy distances there;
    - for simplicity the whole page runs on the lazy backend, although at small n it is not expected to help `max-div`'s results.

## III. Results

The best one-shot tool is a farthest-point picker at every size: `RDKit` up to n = 10,000, `fpsample` at n = 100,000, where the other pickers reach the same value within 0.5 %. `max-div`'s `DEFAULT` preset starts from the same farthest-point construction, so the comparison is about what its optimization adds on top, and at what fixed cost:

- **n ≤ 1,000**: `max-div` passes the best picker within 50 ms and keeps improving to 60 s — at n = 100 up to the [certified optimum](tier1.md), at n = 1,000 to 9 % (one worker) and 11 % (12 workers) above the picker.
- **n = 10,000**: `max-div`'s first budgets sit 0.5 % below the picker line, the optimization overtakes at 1 s in both series, and at 60 s the series end 8 % (one worker) and 9 % (12 workers) above it. The picker itself takes 2 ms (`fpsample[kdline]`) to 4 s (`RDKit`) for its one answer.
- **n = 100,000**: the single-worker series stays on its farthest-point start, 0.02 % above the picker at every budget up to 60 s; the 12-worker series starts 0.3 % above it and gains a further 0.05 % by 60 s. Both pay a set-up before their first point, 2 s with one worker and 3 s with 12, against 50 ms for `fpsample[kdline]`.

### III.A. Problem U1 at size n = 100

![tier2_U1_100_min_separation](./images/tier2_U1_100_min_separation.webp)

### III.B. Problem U1 at size n = 1,000

![tier2_U1_1000_min_separation](./images/tier2_U1_1000_min_separation.webp)

### III.C. Problem U1 at size n = 10,000

![tier2_U1_10000_min_separation](./images/tier2_U1_10000_min_separation.webp)

### III.D. Problem U1 at size n = 100,000

![tier2_U1_100000_min_separation](./images/tier2_U1_100000_min_separation.webp)

## IV. Tables

The [tables page](tier2_tables.md) holds, per size, every tool's quality and time in one ordering, with `max-div` at its quoted budgets, and the overtake budgets.
