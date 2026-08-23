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

Different solvers come with different tuning knobs to influence their runtime, memory consumption, result quality.  Knowledgeable users should therefore be assumed to use a **solver tuned optimally for the scenario** at hand: memory-bound, time-constrained or quality-focused.

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

!!! note "Pseudo-code"

    ```text
    FOR EACH solver configuration:

        FOR EACH n in N (smallest to largest):
            run under hard M_max and T_max          # kill on either
            outcome is one of: success | T_exceeded | M_exceeded
            record peak memory M(n)
            STOP after the first *_exceeded

        # memory-bound size, from the final run's outcome:
        IF success or M_exceeded:
            RECORD largest n that succeeded          (None if never)

        IF T_exceeded:                               # too slow to reach M_max
            extrapolate from the successful M(n), by how many there are:
                >= 3   ->  fit f(n) = c0 + c1*n + c2*n^2   (c0,c2 >= 0; c1 >= 4d = 8)
                           IF c2 < 0.1                # < 1 byte per k*n entry: no real
                               refit with c2 = 0      # allocation can grow this slowly
                           RECORD largest n in N with f(n) <= M_max
                   2   ->  same, but a linear fit
                   1   ->  RECORD that single n
                   0   ->  RECORD None
    ```

### IV.C. Time-Bound Setting

#### IV.C.1. Considerations

The time-bound setting adds the time budget T_max on top of the memory budget — but the measurement runs of the memory-bound setting (IV.B.2) already enforce both budgets: every run there was executed under hard M_max and T_max kills.  No additional runs are therefore needed; the time-bound result is derived from the runs already executed.

We assume...

- runtime increases monotonically with increasing `n`, so the first `*_exceeded` outcome bounds all larger sizes
- a single run per (configuration, `n`) decides the pass/fail verdict: runtime noise can at worst shift a result by one step in `N`, an inaccuracy the granularity of `N` already accepts

#### IV.C.2. Protocol

!!! note "Pseudo-code"

    ```text
    FOR EACH solver configuration:
        take the run outcomes recorded by the memory-bound protocol (IV.B.2)

        time-bound size = largest n in N with a 'success' outcome   (None if none)
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

!!! note "Pseudo-code"

    ```text
    n_max = largest n reached by any solver under the time-bound setting

    FOR EACH n in N up to n_max:
        draw 31 random selections of size k
        Q_random(n) = median quality (diversity) over those 31 selections
    ```

#### IV.D.2. Determining `Q_extended`

> `Q_extended` is recorded per `(solver_config, n)`.

!!! note "Pseudo-code"

    ```text
    n_max = largest n reached by any solver under the time-bound setting

    FOR EACH solver configuration:
        FOR EACH n in N up to n_max:
            run under M_max and the extended budget T_extended = 15 * T_max
            IF success within both budgets:
                Q_extended(solver_config, n) = quality (diversity)
            ELSE:                                   # exceeded memory or time
                STOP and move to the next configuration
    ```

#### IV.D.3. Determining `Q_observed`

> `Q_observed` is recorded per `(solver_config, n, seed)`.

!!! note "Pseudo-code"

    ```text
    FOR EACH solver configuration:
        n_max = this configuration's largest n under the time-bound setting
        IF n_max = None:
            Q_observed = None for this configuration
        ELSE:
            n_seeds = 1 if the solver is deterministic or unseeded, else 3
            FOR EACH n in N up to n_max:
                FOR seed = 1 .. n_seeds:
                    run the solver
                    Q_observed(solver_config, n, seed) = quality (diversity)
    ```

#### IV.D.4. Determining the quality-bound problem size limits

> The quality-bound problem size limit is recorded per `solver_config`.  No solver runs are involved in this phase; it only combines the quantities recorded in IV.D.1–IV.D.3.

!!! note "Pseudo-code"

    ```text
    n_max = largest n reached by any solver under the time-bound setting

    FOR EACH n in N up to n_max:
        Q_best_known(n) = max over all Q_observed(*, n, *) and all Q_extended(*, n)
                          (every configuration, every seed)
        Q_threshold(n)  = 0.1 * Q_random(n) + 0.9 * Q_best_known(n)

    FOR EACH solver configuration:
        FOR EACH n it reached (as in IV.D.3):
            Q_median(solver_config, n) = median over seeds of Q_observed(solver_config, n, *)
        quality-bound size = largest n with Q_median >= Q_threshold(n)   (None if never)
    ```

Note that a `Q_observed` value competes for `Q_best_known` per seed (a lucky draw is still a known solution), while the pass/fail verdict uses the per-config **median** over seeds — so a lucky seed can raise the bar for everyone, but can never carry its own configuration over it.
