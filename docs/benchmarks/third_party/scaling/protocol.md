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
- **memory**: we define the memory budget as **M_max = 32GB** in line with typical memory of current high-end desktop machines.  Two measurements serve two purposes:
    - **enforcement is machine-level**: a run is killed once the machine's memory in use rises more than M_max above its level right before the solver process started.
        - This counts every process a solver spawns, counts shared memory once, and requires no knowledge of any solver's internals.
        - Its noise (unrelated OS activity) is negligible at the cap's scale, and an otherwise-quiet machine is assumed, as the timing measurements already do.
    - **the recorded memory footprint** — the input to the extrapolating fit of IV.B.2 — is the **peak RSS of the solver process**: precise where machine-level readings are far too noisy to fit growth rates from.
        - RSS fully covers a single process, threads included, but not worker *processes* — so a solver observed spawning worker processes is excluded from memory extrapolation.
        - No information is lost by that exclusion: worker processes only ever add memory, so a solver's memory-bound size is reached by its single-process configurations.

    Memory used for problem construction — which happens before the solver process starts — is excluded from both.
- **time**: we define time budget as **T_max = 1min**, mostly driven by keeping the overall protocol executable.  Time is measured **end-to-end**: the clock runs from handing the raw input vectors to the solver until it returns a selection, so any distance computation or other setup work the solver performs is included in its cost.
- **machine**: all measurements are executed on the same reference machine: an Apple M3 Max (12 performance cores, 4 efficiency cores).
- **seeding**: every solver run uses the fixed seed `42`, except where a phase explicitly enumerates multiple seeds (the quality runs of IV.D.3).
- **quality**: judged at two gap-closure fractions, 50% and 90%.  For a fraction `b`, the minimum solution quality (=diversity) threshold is defined as **Q_threshold(b) = (1-b) Q_random + b Q_best_known**, i.e. the solver closed at least the fraction `b` of the quality delta between a purely random selection and the best known solution.  Each fraction yields its own recorded size limit; the 90% one marks near-best quality.

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

The memory sweep is **independent of the time sweep** (IV.C), and runs first — the stages read memory → time → quality, matching the nesting of section II.

The key relaxation is that a run here does not need to *complete* — only to *allocate*.  Each size runs for up to `T_max` as an **observation window**; a run that outlives its window is killed and still yields a footprint.  This decouples memory observation from the time limit, which would otherwise truncate the sweep at sizes where a solver's memory growth is still invisible under its process's fixed startup cost.

We assume...

- memory usage increases monotonically with increasing `n` and does so either linearly or quadratically
- memory usage is sufficiently deterministic to not require multiple runs with different seeds

The extrapolating fit is only trusted once its **trust conditions** all hold:

- the recorded footprints **span a 3x range** — the growth term dominates the fixed baseline within the data;
- the fitted model explains them, **R² >= 0.95**;
- there are **at least 5 measured sizes** — a high-R² fit over only three or four points extrapolates to the cap on too little evidence.

Together these mean the extrapolation extends a measured trend, not an assumption.

A solver can also end its size sweep for a non-resource reason: it cannot express the instance at some size at all (a `failed` outcome — e.g. a sampler whose kernel rank is exceeded).  The failure is recorded and disclosed with its reason, and ends the sweep like a memory kill does — the previous size is the result, since nothing larger runs at all.  The one exception is a failure **before any size has succeeded**: some solvers fail the tiny smallest instance yet work above it, so a failure with no measurement yet is skipped and the sweep tries the next size.

Each configuration gets one discarded warm-up run before its sweep: the first process after a fresh environment install pays a one-off import/bytecode-compilation cost that would otherwise land in its first measurement.

#### IV.B.2. Protocol

!!! note "Pseudo-code"

    ```text
    FOR EACH solver configuration:

        IF worker processes are observed on its first run:
            RECORD not measured                   # RSS misses the workers; the solver's
            CONTINUE                              # single-process configurations carry
                                                  # its memory-bound size

        FOR EACH n in N (smallest to largest):
            run for up to T_max                   # observation window: completion not
                                                  # required; M_max kill (machine-level)
            record footprint M(n)                 # solver-process RSS

            STOP when one of:
                M_max crossed  ->  RECORD previous n                  # measured directly
                solver failed AND >= 1 size already succeeded:
                                   RECORD previous n                  # measured directly; disclosed
                                                                      # (a failure before any
                                                                      #  success is skipped)
                >= 5 sizes  AND  M(n) span >= 3x  AND  fit R^2 >= 0.95:
                    fit f(n) = c0 + c1*n + c2*n^2   (c0,c2 >= 0; c1 >= 4d = 8)
                    IF c2 < 0.1                   # < 1 byte per k*n entry: no real
                        refit with c2 = 0         # allocation can grow this slowly
                    RECORD largest n in N with f(n) <= M_max
    ```

### IV.C. Time-Bound Setting

#### IV.C.1. Considerations

The time sweep is its own pass over the grid, independent of the memory sweep: here a run must **complete** — return a valid selection — within `T_max`, measured end-to-end.  Every run still executes under the machine-level `M_max` kill, so a time verdict is genuinely "within both budgets" whatever the solver's process tree looks like.

We assume...

- runtime increases monotonically with increasing `n`, so the first `*_exceeded` outcome bounds all larger sizes
- a single run per (configuration, `n`) decides the pass/fail verdict: runtime noise can at worst shift a result by one step in `N`, an inaccuracy the granularity of `N` already accepts

A run may be allowed to finish somewhat past `T_max` instead of being killed exactly at it; it then counts as `T_exceeded` for the verdicts, while its completed measurements remain usable.

#### IV.C.2. Protocol

!!! note "Pseudo-code"

    ```text
    FOR EACH solver configuration:
        FOR EACH n in N (smallest to largest):
            run under T_max and M_max               # both kills active
            PASS = completed within T_max
            STOP at the first non-pass, UNLESS it is a non-resource failure
                with no size passed yet — then skip it and try the next size
                (as in the memory sweep)

        time-bound size = largest passing n          (None if none)
    ```

### IV.D. Quality-Bound Setting

The additional criterion that comes into play here is `median(Q_observed) >= (1 - b) Q_random + b Q_best_known`, judged once per gap-closure fraction `b` (50% and 90%; the median taken over seeds, see IV.D.4).  So we need 3 elements here

- `Q_observed`: regular solver executions within `T_max` and `M_max` but now using different seeds for solver configurations whose seed can influence the result (a literal seed, or a seeded initial pick) (as opposed to memory and time usage, quality _is_ expected to be strongly influenced by random seeds)
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
            run under M_max with the extended budget T_extended = 15 * T_max
            IF the run completes:                   # however long it took
                Q_extended(solver_config, n) = quality (diversity)
            ELSE:                                   # killed on memory or time, or crashed
                STOP and move to the next configuration
    ```

The kill allows a small overrun past `T_extended`, so a completed-but-late run can occur; it counts, because the extended runs feed only the best-known reference: a late solution can only raise that reference, and the same reference is applied to every solver.  The published best-known table reports each entry's measured time.

The time-bound setting (IV.C) keeps its strict criterion: there, completion within `T_max` is itself the published value.

#### IV.D.3. Determining `Q_observed`

> `Q_observed` is recorded per `(solver_config, n, seed)`.

!!! note "Pseudo-code"

    ```text
    FOR EACH solver configuration:
        n_max = this configuration's largest n under the time-bound setting
        IF n_max = None:
            Q_observed = None for this configuration
        ELSE:
            seeds = [the fixed seed] if the seed cannot influence the result, else [1, 2, 3, 4, 5]
            FOR EACH n in N up to n_max:
                FOR EACH seed IN seeds:
                    run the solver
                    Q_observed(solver_config, n, seed) = quality (diversity)
    ```

#### IV.D.4. Determining the quality-bound problem size limits

> The quality-bound problem size limit is recorded per `solver_config` and per gap-closure fraction.  A per-solver limit is also recorded, judging the best configuration at each size (the best-result-across-configurations rule of section III).  No solver runs are involved in this phase; it only combines the quantities recorded in IV.D.1–IV.D.3.

!!! note "Pseudo-code"

    ```text
    n_max = largest n reached by any solver under the time-bound setting

    FOR EACH n in N up to n_max:
        Q_best_known(n) = max over all Q_observed(*, n, *) and all Q_extended(*, n)
                          (every configuration, every seed)

    FOR EACH gap-closure fraction b IN {50%, 90%}:
        Q_threshold(b, n) = (1 - b) * Q_random(n) + b * Q_best_known(n)

        FOR EACH solver configuration:
            FOR EACH n it reached (as in IV.D.3):
                Q_median(solver_config, n) = median over seeds of Q_observed(solver_config, n, *)
            quality-bound size = largest n such that Q_median >= Q_threshold(b) at it and
                                 at every smaller judged n   (None if the smallest judged n fails)

        FOR EACH solver:
            FOR EACH n reached by any of its configurations:
                Q_median(solver, n) = best Q_median(solver_config, n) over its configurations
            quality-bound size = the same passing-prefix rule over these
    ```

Note that a `Q_observed` value competes for `Q_best_known` per seed (a lucky draw is still a known solution), while the pass/fail verdict uses the per-config **median** over seeds — so a lucky seed can raise the bar for everyone, but can never carry its own configuration over it.

A failing size ends the passing range even when larger sizes pass again. At the largest sizes only a few configurations still complete, so `Q_best_known` there rests on fewer solutions — a configuration meeting the threshold in that regime may simply be measured against a weaker reference, so such a pass cannot lift it over failures at smaller, better-referenced sizes.
