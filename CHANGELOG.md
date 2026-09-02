# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

### Changed
- Optimization iteration throughput improved: reads of the selected items' contributions no longer scan all n items, and the add-sampling probabilities are built once per iteration

### Deprecated

### Removed
- Python 3.11 is no longer supported; the minimum is now 3.12

### Fixed

### Security

## 0.17.0 (2026-09-01)

### Added
- A batched farthest-point initialization, `InitializationStrategy.farthest_point_batched()`: the same greedy construction with one pass over the dataset per batch of picks instead of per pick — several times faster at large n, offering each pick the same candidates as `farthest_point()` (separation-family diversity metrics only)

### Changed
- The SMART and THOROUGH presets initialize unconstrained problems with the batched farthest-point construction — several times faster at large n, offering each pick the same candidates as before (the per-pick construction remains under `MEAN_PAIRWISE_DISTANCE`)
- Random one-shot initialization updates its distance bookkeeping in one batched pass — an order of magnitude faster at large n, with a `parallel` option that speeds it up further when the solve runs single-process

## 0.16.2 (2026-08-31)

### Fixed
- Invalid constraint definitions (empty groups, negative counts, `min_count > max_count`, out-of-range item indices) are rejected with a `ValueError` at construction (an out-of-range index is reported with the constraint's position) — previously they crashed or silently mis-solved
- `k = n` problems now return the forced full selection immediately (also usable to score an existing selection) — previously they failed
- Out-of-bounds accesses in low-level selection primitives are guarded — previously invalid input could corrupt memory
- Failed parallel workers are reported (a warning naming them on partial failure, the original traceback on total failure) — previously they died silently; empty custom worker groups are rejected at configuration
- Unconverged feasibility relaxations are surfaced (a `converged` field on the result, a warning on the initialization path) and their marginals renormalized so they sum to k

## 0.16.1 (2026-08-31)

### Changed
- Solver-scaling quality benchmarks re-measured under the dynamic worker-group default; the best-known and gap-closure results are now shown as charts

## 0.16.0 (2026-08-30)

### Changed

- `most_feasible()` initialization and `check_feasibility()` now solve their underlying relaxation exactly via an interior-point method: exact certified violation floors, and genuinely different feasible initializations per seed (parallel workers no longer start identically on easily feasible problems).

### Deprecated

- The `max_iter` parameter of `check_feasibility()` and `most_feasible()`: the exact relaxation solve has no iteration budget to set; the parameter is ignored and will be removed.

## 0.15.0 (2026-08-30)

### Added
- Minkowski distance with caller-chosen power `p` via `DistanceMetric.minkowski(p, root=True)`; `p` = 1, 2, inf resolve to the existing dedicated metrics, and `p` = 0.5, 0.25 and 0.125 get specialized fast implementations

### Changed
- Parallel solving now adapts its worker grouping during the solve — starting from independent per-worker groups and consolidating toward one group — instead of using fixed groups of about four
- Configuring parallel workers is split over two builder methods: `with_workers()` takes just a worker count and uses the dynamic grouping, and `with_custom_worker_groups()` takes per-worker configurations and keeps a fixed grouping
- The end-to-end budget option moved from an `end_to_end_budget` argument on `with_preset()` and `with_workers()` to a `with_end_to_end_budget()` method on both builders
- `DistanceMetric` values are now created via factory methods (`DistanceMetric.l2_euclidean()`, ...), not accessed as enum members

## 0.14.3 (2026-08-27)

### Changed
- The benchmark CLI's `--target-max-minutes` option, which picked a `--speed` to fit a target duration, is removed; preview a run's estimated duration with `--dry-run` and pick `--speed` directly

## 0.14.2 (2026-08-26)

### Added
- A time budget can now cover the whole solve end to end, distance computation and initialization included: `end_to_end_budget=True` on `with_preset()` and `with_workers()`, and `--total-seconds` on `max-div solve`

### Changed
- The comparison tables' single `max practical n` column is replaced by four measured solver-scaling values — memory, time, and quality at two gap-closure fractions — each with a published definition

### Fixed
- Re-solving the same solver with the SMART or THOROUGH preset now reproduces the first solve's selection for a given seed, instead of carrying adaptive state across solves

## 0.14.1 (2026-08-21)

### Changed
- Constraint membership on constrained problems is now stored as one packed array instead of a dict of per-item arrays, cutting ~150 bytes per item of object overhead and building ~10x faster

### Fixed
- Unconstrained problems no longer allocate per-item constraint bookkeeping (~280 bytes per item), previously the bulk of the solver's memory footprint at large n

## 0.14.0 (2026-08-19)

### Changed
- Parallel solving now defaults to cooperative worker groups of about four workers instead of one all-worker group; explicit `n_groups` values are unaffected

## 0.13.3 (2026-08-18)

### Added
- `most_feasible()` now appears in the published initialization benchmarks for the constrained problems

### Changed
- The initialization, optimization, and feasibility benchmark pages share one `k`-based size series
- The Benchmarks documentation is reorganized by benchmark type

## 0.13.2 (2026-08-18)

### Changed
- SMART and THOROUGH presets now initialize constrained problems with `most_feasible()` instead of the farthest-point construction; unconstrained problems are unchanged

## 0.13.1 (2026-08-17)

### Added
- `max-div benchmark solver feasibility` prints and regenerates the certified per-size feasibility verdicts now shown on the constrained benchmark problem docs pages
- `MaxDivProblem.check_feasibility()` accepts `max_iter` to set the search budget; a larger budget can only move an unknown verdict toward a proof

## 0.13.0 (2026-08-17)

### Added
- Opt-in `InitializationStrategy.most_feasible()`: starts a constrained solve from the most nearly feasible selection it can construct — one satisfying every constraint where such a selection is found, and the least-violating one found otherwise
- `MaxDivProblem.check_feasibility()`: reports whether the constraints can be satisfied at all — provably yes with a satisfying selection, provably no with a re-checkable certificate and a certified violation floor, or an explicit unknown; alongside a per-constraint breakdown of the best selection found

## 0.12.1 (2026-08-16)

### Changed
- Benchmark problem suite restructured, not backward compatible: problems take the size `n` directly (all other dimensions derived), `U1` (new clustered-2D) and `C1` (exact quotas, previously `C2`) are fixed-d=2 reference problems every third-party tool can run, and the log-uniform problem is removed

## 0.12.0 (2026-08-15)

### Added
- Cooperative parallel solving: parallel workers now form worker groups whose members share their best solution mid-run; set the group count via `with_workers(..., n_groups=...)`, or pass nested worker-config lists for explicit groups

### Changed
- Parallel solving defaults to a single cooperative worker group (`n_groups=1`) on 3/4 of the logical cores (previously: fully independent workers on half); a fully independent portfolio remains available via `n_groups` equal to the worker count, and stays the reproducible mode
- Cooperative workers exchange their best selection roughly every 0.05s (was: every 0.5s), improving short-budget results; single solves and fully independent portfolios are unaffected

## 0.11.2 (2026-08-13)

### Added
- `InitializationStrategy.farthest_point()` accepts an optional `top_k`: each greedy pick samples uniformly among the `top_k` best candidates (default 1 keeps the exact greedy construction)

### Changed
- The tunable entropy mechanism of the farthest-point initialization procedure has changed from a pure-random fraction to choosing randomly from top_k-farthest-points in each step: short-budget results improve further, while maintaining the effect that seeds sufficiently diversify the search process

## 0.11.1 (2026-08-12)

### Added
- Parallel solving can report live progress at the same verbosity levels as a single solve: one non-interleaved view showing overall progress and the best result found so far, tagged with the worker that produced it
- A `Verbosity` enum names the verbosity levels for both solvers (plain integers keep working)

### Changed
- SMART and THOROUGH now begin from a lightly randomized farthest-point construction: short-budget results improve and constrained problems reach competitor-level quality sooner; a given seed selects different but equally diverse items than before
## 0.11.0 (2026-08-11)

### Added
- Parallel solving: configure several workers with `ParallelMaxDivSolverBuilder` (defaulting to half the logical cores) and they solve one problem at once, sharing a single copy of the distances; the best result comes back with a summary of what every worker found
## 0.10.3 (2026-08-08)

### Added
- A farthest-point-sampling initialization strategy (`InitializationStrategy.farthest_point()`): the greedy construction farthest-point-sampling tools use, computed from the solver's own distance store, selectable via `MaxDivSolverBuilder.set_initialization_strategy`

### Changed
- Distance-store construction is multi-threaded, roughly 7× faster (depending on CPU architecture) at large problem sizes; parallel builds are bit-identical to the single-threaded ones, and `MAXDIV_PARALLEL_BUILD=0` disables it
- Constrained problems set up roughly 9× faster when the constraints together contain many item indices, via numpy vectorization of the internal constraint-array construction
## 0.10.2 (2026-08-08)

### Changed
- Published comparison benchmarks re-measured against the current solver; the conclusions drawn from them updated accordingly
## 0.10.1 (2026-08-07)

### Changed
- Solving is up to about 10% faster on stored distance backends, from cheaper bookkeeping of which items are selected
- Faster startup: importing the package and running the first solve in a process together take roughly a quarter of the time they did, because routines that were rebuilt on every run are now compiled once and reused
- Swap candidates are drawn from the full pool of non-selected items, so a given seed now selects different items than before; solution quality is unchanged once a run has converged
## 0.10.0 (2026-08-05)

### Changed
- Substantially faster distance lookups and computations, most of all on large problems; the faster computation rounds differently across storage backends and across CPU architectures, so the same input can now yield different but equally diverse selections — results repeat exactly only within one machine, build and backend
## 0.9.1 (2026-08-02)

### Changed
- Solver setup no longer computes dataset-wide diversity contributions (requiring all pairwise distances) up front; they are computed on demand by the strategies that use them
- Swap candidates are sampled from a bounded, gradually growing pool and no longer weighted by dataset-wide contributions, substantially improving results at short time budgets and on large problems
## 0.9.0 (2026-07-31)

### Added
- More flexible distance storage and computation: how pairwise distances are held during solving is now selectable on the solver builder (`DistanceStorage`), with an automatic default —
  - `FULL_MATRIX` stores the full n×n matrix: roughly 2× faster solving at large problem sizes, for twice the memory
  - `LAZY` computes distances on demand from the vectors: removes the O(n²) memory requirement, so much larger problems become feasible
  - `CONDENSED` remains the memory-lean stored layout (previously the only option)
  - `AUTO` (default) picks the fastest layout that fits in memory for vector problems, and keeps the provided format for distance-input problems; the solution summary reports the resolved choice
  - all layouts produce bit-identical distances, so backend choice never changes which items get selected
- Linf (Chebyshev) distance metric: `DistanceMetric.LINF_CHEBYSHEV` measures each pair by its largest per-dimension difference

### Changed
- Square distance inputs are now kept in full-matrix form rather than condensed (zero-copy when possible), and asymmetric input is repaired to exact symmetry with a warning instead of rejected
## 0.8.5 (2026-07-30)

### Changed
- Due to internal optimizations, solver iterations run more than 2× faster on separation-based metrics, so time-budgeted runs reach better selections
## 0.8.4 (2026-07-28)

### Added
- Reference page for each compared third-party tool, covering what it solves, its problem formulation, and its feature set

### Changed
- The comparison page now covers every capability shown in the README table, with footnotes explaining partial support

### Fixed
- Documentation site: the README capability table now renders instead of showing a broken image
## 0.8.3 (2026-07-27)

### Added
- Support for free-threaded Python 3.14 (3.14t)
- Glossary page defining the terms used across the documentation
- At-a-glance capability-comparison table at the top of the README

### Changed
- Solution summaries now say "items selected" rather than "vectors selected"

### Removed
- Roadmap page from the documentation site
## 0.8.2 (2026-07-22)

### Added
- Comparison benchmarks against third-party tools (Python heuristics, exact solvers, and MDPLIB best-known values) in the documentation
## 0.8.1 (2026-07-16)

### Fixed
- `SolverPreset.SMART` no longer crashes on very small problems where an adaptive swap can remove all but one selected item
- Documentation site: removed the defunct polyfill.io script (which triggered login popups) and fixed the broken splash image on Read the Docs and PyPI
## 0.8.0 (2026-07-14)

### Added
- New `MEAN_PAIRWISE_DISTANCE` diversity metric: maximize the mean pairwise distance among selected vectors (the classical max-sum diversity objective), alongside the existing separation-based metrics
## 0.7.3 (2026-07-13)

### Added
- Cosine distance metric (`DistanceMetric.COSINE`) for embedding-style workloads
- Problems can now be constructed from a precomputed distance matrix (square or condensed) via `MaxDivProblem.from_distances`, enabling custom and non-Euclidean metrics
## 0.7.2 (2026-07-13)

### Changed
- Solver iterations skip redundant selection-score computations (when reverting a rejected swap and for intermediate scores that are never read), measured up to ~18% faster iterations depending on preset and problem size
## 0.7.1 (2026-07-12)

### Changed
- Pairwise distances are now computed directly in float32 instead of via a float64 matrix, cutting peak setup memory roughly 3x->1x on large problems

### Fixed
- Pairwise-distance index lookups overflowed 32-bit arithmetic on large problems (n ≳ 46k), causing wrong distances or crashes; indexing is now 64-bit
## 0.7.0 (2026-07-11)

### Added
- Per-constraint weights: each `Constraint` can be weighted (default 1) to scale its influence on feasibility scoring
- Constraint-violation penalty is now configurable as linear (default) or quadratic via `MaxDivSolverBuilder.with_constraint_penalty`
## 0.6.2 (2026-07-10)

### Added
- GitHub Releases with changelog notes and SLSA build provenance for every release
## 0.6.1 (2026-07-10)

### Added
- PEP 561 `py.typed` marker: the package's type annotations are now visible to downstream type checkers
## 0.6.0 (2026-07-10)

### Added
- PyPI classifiers and license metadata; the PyPI page now renders the README with tag-pinned links

### Changed
- README badges are now served via shields.io and the splash image is version-stamped at release time (both previously hosted on gh-pages)
- Releases are now driven by a local validated release flow instead of CI-side version bumping
- Changelog migrated to the Keep a Changelog format (whole history restructured)
- docs: initial structure for API reference
- docs: minor improvements to figures layout
- docs: minor improvements to changelog layout
## 0.5.5 (2026-02-23)

### Changed
- Improve usage ergonomics of `DiversityMetric` by making it a plain StrEnum
- Reorganize internal, non-public-API functionality into `max_div._core`
- Create top-level `max_div.solver`, ... modules for user-facing API
- docs: extend solver preset benchmarking data (U1-U4, C1-C4)
- docs: minor improvements to figures layout

## 0.5.4 (2026-02-16)

### Added
- Add `--max-run-duration-minutes` option to `benchmark solver presets` CLI command

### Changed
- Simplify: remove `glightbox` mkdocs plugin, due to limited zoom functionality and awkward ergonomics on mobile
- Improved q10-q90 solver preset results uncertainty estimation
- Extend solver preset benchmarking data (U3, U4)

## 0.5.3 (2026-02-14)

### Added
- Add tabular solver preset benchmark results to docs

### Changed
- Tweaks to docs figures layout

## 0.5.2 (2026-02-11)

### Added
- docs: add solver preset benchmarking results

### Changed
- Tweaks to docs figures layout
- Improve how solver preset benchmarks are set up and executed

## 0.5.1 (2026-02-09)

### Added
- docs: add illustrations to all 8 solver benchmark problems & restructure for clarity

### Changed
- Use `InitRandomOneShot` instead of `InitFast` for all presets, to allow seed to more strongly affect outcome, increasing the chance to get good results across multiple seeds
- Solver preset benchmarking: improved scope calculations (higher accuracy for target duration; avoid too short durations)
- Solver preset benchmarking: improved result reporting (case-specific number formatting; no constraint score for unconstrained problems)

## 0.5.0 (2026-02-01)

### Added
- Add solver preset benchmarking capabilities to CLI

### Changed
- Rename CLI command `benchmark solver run` to `benchmark solver strategies` to accommodate future additions
- Refactor Markdown formatting functionality for easier reuse

### Removed
- Remove platform benchmarking functionality (formerly used to estimate duration of initialization strategies in presets)

## 0.4.7 (2026-01-27)

### Changed
- General solver (`SolverState`) low-level speed optimizations (factor ~2-3x): selection tracked as a boolean numpy array instead of a SortedSet, constraint membership as numpy arrays instead of Python lists, and new `SolverState.add_many` / `SolverState.remove_many` methods for more efficient batch modifications
- Streamline RANDOM & GUIDED presets to always use 'fast' initialization, like all other presets

## 0.4.6 (2026-01-26)

### Added
- Make adaptive Poisson-based sampling (used for swap size and # candidates meta-parameter sampling) cost-aware, by penalizing adapting towards larger, more expensive values
- Extend solver benchmark problems A1-A5 to extended set of 4 unconstrained (U1-U4) and 4 constrained (C1-C4) problems

### Changed
- Tweaks to SMART & THOROUGH presets for faster convergence: increase adaptive sampling learning rate by ~10x and make adaptive sampling cost-aware (SMART: 0.5, THOROUGH: 0.1)
- Make solver debug info reporting (verbosity=25) more informative (a.o., added swap success %)
- Make all numba.njit-decorated functions cached

## 0.4.5 (2026-01-20)

### Added
- Add ability to ignore diversity if constraints are not yet satisfied, the first 'f' fraction of optimization step progress
- Add new optimization strategy 'OptimSmartSwaps', leading to more robust & faster convergence across the board
- Add new solver presets: `RANDOM` (baseline strategy for reference, using `OptimRandomSwaps`), `GUIDED` (former default), `SMART` (new default), and `THOROUGH` (same as `SMART` but with significantly wider sampling ranges for `swap_size`, `nc_add` & `nc_remove`)

## 0.4.4 (2026-01-18)

### Added
- Support sampled parameters (using `AdaptiveSampler` class) in optimization strategies, similar to `ParameterSchedule`, but stochastic in nature

### Changed
- Full refactor of sampling functionality, for improved flexibility, performance & future-readiness
- `MaxDivSolver.solver()`: improve granularity of `verbosity` parameter wrt reporting frequency, including debug mode
- `solve` CLI command: add ability to specify which preset to use

### Removed
- Remove unneeded `randint_constrained_robust` function and associated functionality

## 0.4.3 (2026-01-02)

### Added
- Implement `verbosity` parameter in `MaxDivSolver.solver()` and add option for more detailed tabular progress reporting

### Changed
- ci: show slow tests in CI when running tests or generating coverage (badge generation)
- cli: add `benchmark platform` command to benchmark platform speed
- cli: add `solve` command for easy test-running the solver on test problems
- Initialization strategies: update internal calling API in preparation of improved progress reporting

## 0.4.2 (2026-01-02)

### Changed
- `TargetDuration` now implements `__eq__` and more informative `__repr__` and `__str__` for time-based durations
- `MaxDivSolverBuilder`: add config options to default preset + improve test coverage
- `estimate_platform_speed`: move to `internal.benchmarking` and improve implementation for maintainability & encapsulation

### Fixed
- Fix slowest & a few flaky tests

## 0.4.1 (2025-12-31)

### Added
- Notebooks for automated estimation of time models of init. and optim. strategies for different problem sizes

### Changed
- Make default preset in `MaxDivSolverBuilder` smart, taking into account expected duration of strategies
- Make solver benchmarks explicitly tied to presets, reducing duplication of code

### Fixed
- Corner case where `randint_constrained` could return duplicate samples due to `int32` overflow/underflow wrap-around

## 0.4.0 (2025-12-28)

### Added
- Implement initial `default` preset in `MaxDivSolverBuilder`, allowing low-threshold initialization of a solver for any given problem

## 0.3.9 (2025-12-28)

### Added
- Implement `OptimGuidedSwaps` optimization strategy, the intended workhorse strategy for the solver
- Extend solver benchmarking implementation & docs with new strategy

### Changed
- Switch to more generic and faster `exponential_selectivity` in initialization strategies
- Streamline solver benchmarking implementation
- Solver benchmarking: add options `--optimization-only` and `--initialization-only` for targeted tests
- Sort vectors in solver benchmark problems such as to penalize degenerate/naive problem initialization (i.e. make the first `k` vectors not randomly scattered, but clustered together)

## 0.3.8 (2025-12-25)

### Added
- Implement `SwapBasedOptimizationStrategy` base class for swap-based optimization strategies
- Implement `InitDummy` baseline initialization strategy
- Implement `OptimRandomSwaps` baseline optimization strategy
- Extend solver benchmarks with testing of optimization strategies + test up to larger problem sizes

### Changed
- Speed improvement of score computation in case of soft constraints
- Improve unit test coverage & depth of `SolverState`
- Speed improvement of fast exponential and power function approximations by up to ~40%, leading to speed improvements in modify_p_selectivity methods of up to ~25-30%

## 0.3.7 (2025-12-24)

### Added
- Extend `OptimizationStrategy` implementation to support scheduled parameters, that evolve according to a schedule during multi-iteration execution of a strategy
- Add `exponential` p-selectivity modification method
- Add truncated Poisson sampling

### Changed
- docs: tweak layout of solver benchmark docs
- Speed improvement of `geomean_separation_approx` (~60%)
- Improve sampling heuristics of `InitEager`, leading to slightly higher (~5%) diversity scores and slightly faster execution (~10%)
- Reduce unnecessary array copying in `InitRandomBatched` and `InitRandomEager`

## 0.3.6 (2025-12-22)

### Added
- Implement `InitEager` initialization strategy & extend solver benchmarking implementation & docs with new strategy

## 0.3.5 (2025-12-22)

### Added
- Implement `InitRandomBatched` initialization strategy & extend solver benchmarking implementation & docs with new strategy
- `randint_constrained`: add `k_context` parameter to enable sampling in batches
- `randint_constrained`: add `i_forbidden` parameter to have more flexibility on how to sample from non-selected subsets of ranges
- Refactor `modify_p_selectivity` methods: unify various implementations under single API, allow in-place modification for optimal speed, add fast-power based implementation (method=20), add accuracy estimation to `modify_p_selectivity` benchmark

### Changed
- Rename initialization strategy `InitOneShotRandom` -> `InitRandomOneShot`, in preparation of other random init strategies
- `benchmark internal` CLI command: redirect output to file using `--file` flag, add `all` benchmark that runs all in one, add `make` command to automatically regenerate benchmark results in docs
- Improve accuracy of `modify_p_selectivity` method 20 (recalibrated coefficients) & unify calibration methods
- Speed up `modify_p_selectivity` method 100 by ~10%

## 0.3.4 (2025-12-13)

### Added
- Add 5th solver benchmark problem with simpler constraints, to complete the spectrum
- Add benchmarking results to documentation & add `make` command to automatically regenerate benchmark results

### Changed
- `benchmark solver run` CLI command: add constraint score reporting, add uncertainty ranges to reported metrics, add `--turbo` flag and `--speed` parameter, redirect output to file using `--file` flag, allow benchmarking `all` test problems with single command, improve test coverage

## 0.3.3 (2025-12-12)

### Added
- Implement `InitOneShotRandom` initialization strategy
- Make `SolverStep`, `InitializationStrategy` & `OptimizationStrategy` instances have a settable random seed
- Add distinction between `internal` and `solver` benchmarks in CLI & implement `benchmark solver` commands

### Changed
- Add initialization of `SolverState` to the reported results
- Improve reliability & testing of DiversityMetric implementations for sub-standard array sizes
- Improve docs structure

## 0.3.2 (2025-12-11)

### Added
- Framework for built-in benchmark suite for `MaxDivSolver` problems & strategies
- First 4 benchmark problems (A1, A2, A3, A4)

### Changed
- Improve structure of CLI implementation; improving encapsulation of internal, low-level, CLI-only benchmarking functionality
- Various low-level algorithmic tweaks & test coverage improvements

## 0.3.1 (2025-12-10)

### Added
- Add support for Python 3.14
- `MaxDivProblem` class, so problems can be formulated outside of the `MaxDivSolver` and `MaxDivSolverBuilder` classes, paving the way for a built-in benchmark suite

### Changed
- Improve consistency of dimension notations `n`, `d`, `k`, `m` across codebase

## 0.3.0 (2025-12-10)

### Added
- Methods for modifying 'selectivity' of an array of probabilities (more or less uniform): `modify_p_selectivity_power` & `modify_p_selectivity_pwl2`, with benchmarking in CLI & docs
- Make strategies aware of progress fraction within execution of a `SolverStep` to allow scheduling various parameters (constraint softness, wide vs selective search, ...)

### Changed
- Add caching of `Score` object in `SolverState` to avoid redundant score computations when state hasn't changed

## 0.2.6 (2025-12-09)

### Added
- Full support for structured multi-component scoring & diversity tie-breaker metrics: structured `Score` object with optional `div_tie_breakers` field, support for manual or preset tie-breaker metrics in `MaxDivSolverBuilder`, integration of updated `Score` class with `SolverState` & `Solver` implementations
- Add initial support for soft constraint handling
- Make a `SolverStep` return score checkpoints correctly via `SolverStepResult`
- Make `MaxDivSolver` return extensive meta-data via `MaxDivSolution`
- Add `non_zero_separation_frac` to `DiversityMetric` implementations and benchmarking suite

### Changed
- ~20% speed improvement of `min_separation` diversity metric

## 0.2.5 (2025-12-06)

### Changed
- Improve progress tracking implementation for future use in `MaxDivSolver` overall progress reporting
- Introduction of `SolverStep`, combining a strategy & a duration
- Introduction of `Score` class for more structured scoring enabling more advanced future strategies

## 0.2.4 (2025-12-05)

### Added
- `SolverStrategy` progress bar support
- First version of main `MaxDivSolver` loop

### Changed
- Improve test coverage of `SolverState` and `SolverStrategy` modules

## 0.2.3 (2025-12-04)

### Added
- Add `SolverState` snapshot management, so `SolverStrategy` modules can perform trial-modifications to the state
- Implement first version of `SolverStrategy` API

## 0.2.2 (2025-12-03)

### Added
- Add `DiversityMetric.approx_geomean_separation()` based on `fast_log2` and `fast_exp2`
- Add benchmarking of different `DiversityMetric` implementations to CLI & docs

### Changed
- Further improve test coverage of `fast_log*` & `fast_exp*` functions
- Make `DiversityMetric` implementation more flexible

## 0.2.1 (2025-12-02)

### Added
- `fast_exp*` functions in the same spirit as the updated `fast_log*` functions

### Changed
- `fast_log_*` & `fast_log2_*`: simplify & speedup by only supporting degree 2 (we don't need more accuracy), and update computation of coefficients to guarantee continuity and smoothness of the approximation

## 0.2.0 (2025-11-30)

### Added
- Initial `MaxDivSolver` framework

## 0.1.3 (2025-11-27)

### Changed
- Further speed up test coverage generation
- Add ReadTheDocs build status badge to README
- Add License badge to README

## 0.1.2 (2025-11-25)

### Added
- Implement `randint_constrained_robust`

### Changed
- Speed up test coverage generation
- Improve visual clarity of `randint_constrained` benchmarking output
- Add benchmarking for `randint_constrained_robust` to `benchmark randint_constrained` CLI command

### Removed
- Remove excessively slow pure-Python `randint_constrained` & promote `randint_constrained_numba` -> `randint_constrained`

## 0.1.1 (2025-11-23)

### Added
- Implement `Constraints` class
- Implement `sampling.con.randint_constrained_numba` function, offering a 10-100x speedup over the non-`numba` version

## 0.1.0 (2025-11-22)

### Added
- Add `benchmark randint_constrained` command to CLI

### Changed
- Various minor improvements to docs
- Improve test coverage of CLI functionality

## 0.0.9 (2025-11-21)

### Added
- Add `sampling.con.randint_constrained`, which is like `randint` but supporting constraints

### Changed
- Make naming more consistent with other known functionality: `sampling.discrete.sample_int` -> `sampling.uncon.randint`
- Ensure `randint` also works with non-normalized probabilities (with minimal loss of efficiency, especially in `numba`-flavor)
- Update local dev tooling to declutter project root

## 0.0.8 (2025-11-18)

### Changed
- `sampling.discrete.sample_int`: speedup for non-uniform sampling without replacement (custom min-heap implementation)

## 0.0.7 (2025-11-17)

### Changed
- Improve test coverage of internal rng functionality
- Improve speed of generating `int32` arrays in `sampling.discrete.sample_int`
- Add `pytest-xdist` and `pytest-rerunfailures` to improve test performance & robustness

## 0.0.6 (2025-11-16)

### Added
- Make `numba` and non-`numba` versions of `sampling.discrete.sample_int` public and self-sustained (full-featured)
- CLI command `numba-status` to report on `numba` installation and configuration status

### Changed
- `sampling.discrete.sample_int`: significant speedup, improved benchmarking + notes on optimization efforts
- Improve coverage statistics for `numba`-decorated functions
- Make `numba` a non-optional dependency for the sake of code simplicity
- Improve granularity of speed-vs-accuracy trade-off in benchmarking functionality (add `--speed` argument)

## 0.0.5 (2025-11-08)

### Added
- Implement `max-div` CLI, allowing package to be installed as tool to define system-wide command 'max-div'

### Changed
- Allow benchmarking functionality to output results in Markdown format
- Switch docs to use `readthedocs` theme

## 0.0.4 (2025-11-08)

### Changed
- Further improvements to CI/CD pipeline

## 0.0.3 (2025-11-07)

### Added
- Move to trunk-based development workflow with release branches

## 0.0.2 (2025-11-02)

### Added
- `sampling.discrete.sample_int`
- `benchmark.benchmark_sample_int`

## 0.0.1 (2025-11-01)

### Added
- Initial project setup & framework
