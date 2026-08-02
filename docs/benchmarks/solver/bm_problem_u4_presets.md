# Benchmark Results - Problem U4 - Solver Presets

!!! note "Measurement vintage"
    These results were measured with max-div v0.5.4 (February 2026). Later releases have
    changed solver speed and behavior, so the figures below describe that version's
    solver, not the current one.

## I. Introduction

We present results of the different built-in solver presets on problem U4, size=100.  We run each preset for
increasing durations and evaluate final constraint & diversity score of the solution.  Each run is performed with a
different seed, so also the influence of the seed is evaluated.

As the different seeds cause some randomness in the results, we estimate q10-q90 uncertainty bounds, by performing
quantile regression through the data points of each preset using a cubic spline with monotonicity constraints.

The resulting uncertainty bounds give an idea of the result that can be expected by e.g. taking the best result out of 10 runs
(with different seeds) as this is expected to lie around ~q90.  Uncertainty bounds are only estimated and shown
for the relevant metric (constraint score if the problem is constrained and infeasible; diversity score otherwise)

## II. Results

### A. Figures

![Problem U4 - Size 100 - Preset results](./images/preset_results_U4_100.webp)

### B. Tables

--8<-- "docs/benchmarks/solver/results/preset_result_quantiles_U4_100.md"
