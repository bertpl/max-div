# Head-to-Head — vs. MDPLIB Best-Known Values

## I. Goal and reading guide

This tier is the only one run on instances `max-div` did not generate: the MMDP instance sets of the [MDPLIB library](https://www.uv.es/rmarti/paper/mdp.html) (Glover, Geo, Ran), the literature's shared max-min diversity benchmark. The question is how far `max-div` is from the best value anyone has published per instance, as a function of budget, and where it matches or exceeds that value.

Every chart covers one instance group (a set, a size n and a selection size k, ten instances) and reads the same way:

- the **y axis** is the gap to the best-known value in percent, positive below it; the **dotted line** at zero is the best-known value itself;
- the **black curves** are `max-div`: solid with one worker, dashed with 12; the line is the mean over the group's instances and seeds, the band the min/max;
- each **dot** is one one-shot tool at its own measured time and mean gap.

## II. Protocol

The tier reuses the time budget and reference machine of the [solver-scaling protocol](../scaling/protocol.md), and the [solver configurations](../scaling/solver_configs.md) measured there. The tier runs 3 seeds per cell, not 5 ([why](../scaling/protocol.md#iii-fundamental-constants-invariants)).

### II.A. Instances, entrants and budgets

- **Instances**: the 120 published (instance, k) pairings of the Geo and Ran sets — n ∈ {100, 250, 500}, two k values per n, ten instances each.
    - Geo instances come with coordinates (d = 5 at n = 100, d = 13 at n = 250 and 500); Ran instances are given as a distance matrix only.
    - The 75 Glover pairings (n ≤ 30) are measured but not charted: one sentence on the tables page states the match count.
    - Instances are fetched at run time and never redistributed; the maintainers' site states no license ("all rights reserved") and asks that the library be cited as Martí, Duarte, Martínez-Gavara & Sánchez-Oro (2021), *The MDPLIB 2.0 Library of Benchmark Instances for Diversity Problems*.
- **Objective**: minimum separation, the MMDP objective, scored identically for every tool.
- **Entrants**: the registry tools whose input form the instance provides; one run per seed where the tool is seeded, and a tool's time includes any conversion it needs.
    - On Ran, the tools that accept a distance matrix: `qc-selector` (max-min and max-sum), `kmedoids`.
    - On Geo, additionally the tools that take vectors: `fpsample`, `skmatter`, `RDKit`, `apricot-select`, `DPPy`, `code-FDM`.
    - Exact solvers are compared on the [exact-solver tier](tier1.md), not here.
- **max-div**: `DEFAULT` preset, the single-worker budget series from 1 ms to 60 s and the 12-worker series from 1 s to 60 s on every pairing.
    - One independent solve per budget and seed, timed end to end; charts plot measured wall-clock.
    - A budget is charted only when it exceeds the measured time of the previous charted budget, so the small budgets that all end at the set-up cost appear once.

### II.B. Reference values

The best-known value of a pairing is the largest value published by any of three sources; the tables page lists every value with its source.

- **RMGD2010** — Resende, Martí, Gallego & Duarte (2010), *GRASP and path relinking for the max–min diversity problem*, Computers & Operations Research 37(3): the values distributed with the instances, best over the paper's six algorithms.
- **PHG2011** — Porumbel, Hao & Glover (2011), *A simple and effective algorithm for the MaxMin diversity problem*, Annals of Operations Research 186(1), appendix tables 6–7 ([author manuscript](https://cedric.cnam.fr/~porumbed/papers/paperAOR11.pdf)).
- **DCGL2009** — Della Croce, Grosso & Locatelli (2009), *A heuristic approach for the max–min diversity problem based on max-clique*, Computers & Operations Research 36(8), as republished in PHG2011.

The 2010 values alone sit below a later source on 68 of the 120 Geo and Ran pairings, so a comparison against the 2010 values alone mostly measures how far later work moved them.

A value is flagged as a **proven optimum** only where a published exact method certified it:

- every n = 100 pairing (DCGL2009, per PHG2011);
- the n = 250 and 500 pairings CPLEX proved in Saboonchi (2013), *Solving the p-dispersion problem*, [HEC Montréal thesis](http://biblos.hec.ca/biblio/theses/2013NO10.PDF), tables 3.V and 3.VI. <!-- codespell:ignore theses -->

That is 93 pairings in all. On those, matching the reference is the most any method can reach.

Ran's distances are integers in [1, 200], so on that set one integer step below a small reference value is a gap of several percent (20 % where the reference is 5), which the charts show as a staircase.

## III. The Geo set: instances given as coordinates

No Geo value is exceeded. The best-known value is reached on nearly every n = 100 instance, on fewer at n = 250, and on one instance per group at n = 500; the remaining gap at 60 s is a few percent at most on average per group, and 12 workers cut it further. The per-group counts and gaps are on the [tables page](tier3_tables.md).

The farthest-point pickers (`fpsample`, `skmatter`, `RDKit`, `qc-selector` max-min, `code-FDM`) sit 3.5–8.5 % short in a few milliseconds; `max-div` passes them within about 100 ms of budget and keeps improving. The tools with a different objective (`apricot-select`, `kmedoids`, `DPPy`, `qc-selector` max-sum) sit 31–49 % short.

### III.A. Geo instances at n = 100, k = 10

![tier3_geo_100_10](./images/tier3_geo_100_10.webp)

### III.B. Geo instances at n = 100, k = 30

![tier3_geo_100_30](./images/tier3_geo_100_30.webp)

### III.C. Geo instances at n = 250, k = 25

![tier3_geo_250_25](./images/tier3_geo_250_25.webp)

### III.D. Geo instances at n = 250, k = 75

![tier3_geo_250_75](./images/tier3_geo_250_75.webp)

### III.E. Geo instances at n = 500, k = 50

![tier3_geo_500_50](./images/tier3_geo_500_50.webp)

### III.F. Geo instances at n = 500, k = 150

![tier3_geo_500_150](./images/tier3_geo_500_150.webp)

## IV. The Ran set: instances given as a distance matrix

The Ran picture is the same, except for Ran 500 with k = 150: the best-known value is reached on every n = 100 instance and on fewer from n = 250 on, never exceeded, with the per-group counts and gaps on the [tables page](tier3_tables.md).

On that group, every best-known value is 5, `max-div` reaches 4 on every instance with one worker — the integer distances make that one step a 20 % gap — and 12 workers reach 5 on three instances within 60 s.

Only the distance-matrix tools enter: `qc-selector` max-min sits 2–11 % short (39 % on the k = 150 group), `kmedoids` and `qc-selector` max-sum 4–31 % (80 % on that group).

### IV.A. Ran instances at n = 100, k = 10

![tier3_ran_100_10](./images/tier3_ran_100_10.webp)

### IV.B. Ran instances at n = 100, k = 30

![tier3_ran_100_30](./images/tier3_ran_100_30.webp)

### IV.C. Ran instances at n = 250, k = 25

![tier3_ran_250_25](./images/tier3_ran_250_25.webp)

### IV.D. Ran instances at n = 250, k = 75

![tier3_ran_250_75](./images/tier3_ran_250_75.webp)

### IV.E. Ran instances at n = 500, k = 50

![tier3_ran_500_50](./images/tier3_ran_500_50.webp)

### IV.F. Ran instances at n = 500, k = 150

![tier3_ran_500_150](./images/tier3_ran_500_150.webp)

## V. Tables

The [tables page](tier3_tables.md) holds the numbers behind the charts.
