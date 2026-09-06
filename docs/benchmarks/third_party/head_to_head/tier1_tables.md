# Head-to-Head — vs. Exact Solvers — Tables

The tables behind the [exact-solver comparison](tier1.md). Gaps are `max-div`'s median over seeds at the quoted budget, in percent below the certified optimum; a dash marks a budget the series did not run.

## I. Gap to the certified optimum

### I.A. Minimum separation

--8<-- "docs/benchmarks/third_party/head_to_head/results/tier1_gap_min_separation.md"

### I.B. Mean separation

--8<-- "docs/benchmarks/third_party/head_to_head/results/tier1_gap_mean_separation.md"

### I.C. Geomean separation

--8<-- "docs/benchmarks/third_party/head_to_head/results/tier1_gap_geomean_separation.md"

## II. Certification

Per solver, problem and objective: the largest size certified within the 900 s cap, and the size at which certification stopped.

--8<-- "docs/benchmarks/third_party/head_to_head/results/tier1_certification.md"
