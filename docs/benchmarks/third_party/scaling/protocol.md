# Solver Scaling — Measurement Protocol

## I. Introduction

With _solver scaling_ we want to establish the **maximum problem size each solver can practically solve**.  Unfortunately, this question cannot be unambiguously answered:

- What is the _**size**_ of a problem? (dimensions, item count, selection size, number of constraints, ...)
- What is _**practical**_?
  - a solver might theoretically be able to solve a problem, but might require excessive amounts of time (days, months, years, ...)
  - a solver might be able to produce a solution within a reasonable amount of time, but with very low quality (e.g. diversity comparable with a random selection)

Answering these questions in a nuanced way and with all corner cases taken into account would be an extensive study in itself.  

Hence, this page aims at describing a **practical, unambiguous protocol that can answer the _solver scaling_ question** in a sufficiently nuanced way in order to be practical and informative, while still keeping the protocol executable **in a reasonable amount of time** (hours or days, not weeks or months).

## II. The Three Nested Axes

We aim to determine **maximum solvable problem size** under three different constraint regimes, each tighter than the previous:

- **Memory**: maximum problem size a solver can handle _**within a given memory budget**_.
- **Time**: maximum problem size a solver can handle within the same memory budget _**and within a given (wall clock) time budget**_. (on a specified reference machine)
- **Quality**: maximum problem size a solver can handle within the same memory and time budget, while _**achieving optimality (=diversity) sufficiently close to the best known solution**_.

These three descriptions are consciously kept qualitative in nature.  The next section nails down the quantitative aspects of these axes and establishes what we mean by _problem size_.

## III. Fundamental Constants & Invariants

- **problem size**: defined as item count (`n`) of built-in benchmarking problem `U1`:
  - see the [Problem U1 description](../../solver/problem_u1.md): `d=2`, `k=n/10`, clustered item distribution + background distribution + outliers.
  - _**unconstrained**_: since most 3rd party solvers don't handle constrained problems, we choose an unconstrained (no fairness constraints) problem, so we don't exclude any solvers from the start.
  - _**fixed dimensionality**_: some solvers explicitly build a full pairwise distance matrix at solver start (O(n^2) in time and memory), others compute distances lazily as needed (O(n.d) in memory).  In order to be able to distinguish both types we choose to keep `d` fixed and not let it scale proportionally with `n`.
  - _**granularity**_: we only evaluate a fixed set of problem sizes **N = {20, 50, 100, 200, 500, ..., 2e9}** — a 1-2-5 grid — in order to keep total benchmarking time limited and to explicitly acknowledge the accuracy limitations inherent to a time-constrained testing protocol.  The lower end (`n=20`) is the smallest size our benchmark problems build at; the upper end (`n=2e9`) is the largest grid value for which the raw vector data alone (8 bytes per item: `d=2` in float32) fits the memory budget `M_max` defined below (`5e9` no longer does).
  - _**distance metric**_: L2 distance, as this has widest support among competing solvers.
  - _**diversity metric**_: `max-min` (or `minimal separation`), as this has the widest support among competing solvers.  Some solvers can only be configured to optimize for a different metric (or don't explicitly optimize at all), in which case this is explicitly indicated.
- **memory**: we define the memory budget as **M_max = 32GB** in line with typical memory of current high-end desktop machines.  Memory budget is defined as the **peak RSS memory usage** of the solver, excluding any memory used for problem construction before triggering the solver itself.
- **time**: we define time budget as **T_max = 1min**, mostly driven by keeping the overall protocol executable.  Time is measured **end-to-end**: the clock runs from handing the raw input vectors to the solver until it returns a selection, so any distance computation or other setup work the solver performs is included in its cost.
- **machine**: all measurements are executed on the same reference machine: an Apple M3 Max (12 performance cores, 4 efficiency cores).
- **seeding**: every solver run uses the fixed seed `42`, except where a phase explicitly enumerates multiple seeds (the quality runs of IV.D.3).
- **quality**: the minimum solution quality (=diversity) threshold is defined as **Q_threshold = 0.1 Q_random + 0.9 Q_best_known**, i.e. the solver reached at least 90% of the quality delta between a purely random selection and the best known solution.

## IV. Measurement Protocol

### IV.A. Solver Configurations

Different solvers come with different tuning knobs to influence their runtime, memory consumption, result quality.  Knowledgeable users should therefore be assumed to use a **solver tuned optimally for the scenario** at hand: memory-bound, time-constrained or quality-focussed.

Our testing protocol implements the same spirit in order to avoid testing a 3rd party solver in a specific configuration that favors some scenarios above others:

- Each **solver** can potentially be represented by multiple **solver configurations** in our testing protocol.  
- Each solver configuration will be tested independently
- For any of the three axes and any considered problem size `n`, the best result for each solver across its configurations will be taken, assuming a user would also pick the most optimal one for the use case at hand
- Solver configurations considered in our testing protocol are listed in detail [here](solver_configs.md).

### IV.B. Memory-Bound Setting

#### IV.B.1. Considerations

We assume...
- memory usage increases monotonically with increasing `n` and does so either linearly or quadratically
- memory usage is sufficiently deterministic to not require multiple runs with different seeds

Since the memory-bound setting is the least restrictive and the only one that does not consider time budget constraints, care needs to be taken to keep the protocol practical.  

Therefore we will...
- limit actual measurements to our earlier defined time budget T_max
- extrapolate observations beyond T_max, if needed, to determine the largest `n` with the memory budget M_max.

#### IV.B.2. Protocol

```
For each solver configuration:
  - For each `n` in `N`:
    - Run the solver configuration under hard **M_max** and **T_max** 
      constraints (killing the solver when exceeding either)
    - Each run resulting in `success`, `T_exceeded` or `M_exceeded` outcome
    - Record memory usage M(n) for each evaluated n and stop after 
      the first `*_exceeded` outcome
  - if the final run resulted in `M_exceeded` or `success`:
    - Record the last successful `n` (or None if there is none) as the result for this solver configuration
  - if the final run resulted in `T_exceeded`:
    - if # successes >= 3
      - Perform a least-squares regression f(n) = c0 + c1*n + c2*(n^2) through 
        all obtained observations M(n), enforcing c0>=0, c1>=4d=8 (assuming 8 bytes per vector minimal memory usage), c2 >= 0.
      - Find the largest `n` in `N` for which f(n) <= M_max
      - Record this `n` as this solver configuration's
        maximum problem size in the memory-bound setting
    - if # successes = 2
      - Same procedure but using a linear regression
    - if # successes = 1
      - Record that single `n` as the result for this solver configuration
    - if # successes = 0
      - Record 'None' as the result
```

### IV.C. Time-Bound Setting

#### IV.C.1. Considerations

The time-bound setting adds the time budget T_max on top of the memory budget — but the measurement runs of the memory-bound setting (IV.B.2) already enforce both budgets: every run there was executed under hard M_max and T_max kills.  No additional runs are therefore needed; the time-bound result is derived from the runs already executed.

We assume...
- runtime increases monotonically with increasing `n`, so the first `*_exceeded` outcome bounds all larger sizes
- a single run per (configuration, `n`) decides the pass/fail verdict: runtime noise can at worst shift a result by one step in `N`, an inaccuracy the granularity of `N` already accepts

#### IV.C.2. Protocol

```
For each solver configuration:
  - Take the run outcomes of the memory-bound protocol (IV.B.2)
  - Record the largest `n` in `N` with a `success` outcome as this solver
    configuration's maximum problem size in the time-bound setting
    (`None` if no run succeeded)
```

### IV.D. Quality-Bound Setting

The additional criterion that comes into play here is `median(Q_observed) >= 0.1 Q_random + 0.9 Q_best_known` (the median taken over seeds, see IV.D.4).  So we need 3 elements here

- `Q_observed`: regular solver executions within `T_max` and `M_max` but now using different seeds for non-deterministic solvers that support seeding (as opposed to memory and time usage, quality _is_ expected to be strongly influenced by random seeds)
- `Q_random`: determined as the median quality (diversity) of 31 random selections of size `k` for each relevant `n`
- `Q_best_known`: taken as the best (for each relevant `n`) observed quality...
  - over all `Q_observed` (any solver configuration) 
  - AND over all `Q_extended`: results of extended-budget runs (`T_extended = 15 T_max`, see IV.D.2) (any solver configuration)

#### IV.D.1. Determining `Q_random`

> `Q_random` is recorded per `n`.

```
- determine n_max = largest n for any solver under the time-bound setting
- for each n <= n_max in `N`:
  - determine 31 randomized selections of size `k`
  - compute the selection quality (diversity) of each such random selection
  - the median value is recorded as `Q_random(n)` 
```

#### IV.D.2. Determining `Q_extended`

> `Q_extended` is recorded per `(solver_config, n)`.

```
- determine n_max = largest n for any solver under the time-bound setting
- for each solver configuration:
  - for each n <= n_max in `N`:
    - execute the solver under `M_max` memory constraint and `T_extended = 15 T_max` time budget
    - if the solver finished successfully within memory and (extended) time budget, record the selection quality (diversity) as `Q_extended` for this `(solver_config, n)`
    - if the solver exceeded the memory or (extended) time budget, proceed with the next solver config
```

#### IV.D.3. Determining `Q_observed`

> `Q_observed` is recorded per `(solver_config, n, seed)`.

```
- for each solver configuration:
  - determine `n_max` for this solver config as the largest `n` under the time-bound setting
  - if `n_max = None`, record `None` for all `Q_observed` for this solver config
  - if `n_max != None`:
    - determine `n_seeds`:
      - if the solver is deterministic and/or does not support seeds: `n_seeds = 1`
      - otherwise: `n_seeds = 3`
    - for each n <= n_max in `N`:
      - for each `seed = 1,...,n_seeds`:
        - execute the solver and record the selection quality (diversity) as `Q_observed` for this `(solver_config, n, seed)`
```

#### IV.D.4. Determining the quality-bound problem size limits

> The quality-bound problem size limit is recorded per `solver_config`.  No solver runs are involved in this phase; it only combines the quantities recorded in IV.D.1–IV.D.3.

```
- determine n_max = largest n for any solver under the time-bound setting
- for each n <= n_max in `N`:
  - `Q_best_known(n)` = the maximum over all recorded `Q_observed(*, n, *)`
    (all solver configs, all seeds) and all recorded `Q_extended(*, n)`
  - `Q_threshold(n)` = 0.1 `Q_random(n)` + 0.9 `Q_best_known(n)`
- for each solver configuration:
  - for each n <= n_max of this solver config (as in IV.D.3):
    - `Q_median(solver_config, n)` = the median over seeds of
      `Q_observed(solver_config, n, *)`
  - record the largest `n` for which `Q_median(solver_config, n) >= Q_threshold(n)`
    as this solver configuration's maximum problem size in the quality-bound
    setting (`None` if no such `n` exists)
```

Note that a `Q_observed` value competes for `Q_best_known` per seed (a lucky draw is still a known solution), while the pass/fail verdict uses the per-config **median** over seeds — so a lucky seed can raise the bar for everyone, but can never carry its own configuration over it.
