# Change Log

<!------------------------------------------------------------------------------------------------->
> ## v0.4.2
> *(under development)*
<!------------------------------------------------------------------------------------------------->

/

<!------------------------------------------------------------------------------------------------->
> ## v0.4.1
> *(2025-12-31)*
<!------------------------------------------------------------------------------------------------->

- **new**:
    - notebooks for automated estimation of time models of init. and optim. strategies for different problem sizes

- **improved**:
    - make default preset in `MaxDivSolverBuilder` smart, taking into account expected duration of strategies
    - make solver benchmarks explicitly tied to presets, reducing duplication of code

- **fixed**:
    - corner case where `randint_constrained` could return duplicate samples due to `int32` overflow/underflow wrap-around

<!------------------------------------------------------------------------------------------------->
> ## v0.4.0
> *(2025-12-28)*
<!------------------------------------------------------------------------------------------------->

- **new**:
    - implement initial `default` preset in `MaxDivSolverBuilder`, allowing low-threshold initialization of a solver for any given problem.

<!------------------------------------------------------------------------------------------------->
> ## v0.3.9
> *(2025-12-28)*
<!------------------------------------------------------------------------------------------------->

- **new**:
    - solver strategies
       - implement `OptimGuidedSwaps` optimization strategy, the intended workhorse strategy for the solver 
    - extend solver benchmarking implementation & docs with new strategy

- **improved**:
    - solver strategies
        - switch to more generic and faster `exponential_selectivity` in initialization strategies     
    - solver benchmarking
        - streamline solver benchmarking implementation
        - add options `--optimization-only` and `--initialization-only` for targeted tests
        - sort vectors in solver benchmark problems such as to penalize degenerate/naive problem initialization
          (i.e. make the first `k` vectors not randomly scattered, but clustered together)

<!------------------------------------------------------------------------------------------------->
> ## v0.3.8
> *(2025-12-25)*
<!------------------------------------------------------------------------------------------------->

- **new**:
    - solver framework
       - implement `SwapBasedOptimizationStrategy` base class for swap-based optimization strategies
    - solver strategies 
       - implement `InitDummy` baseline initialization strategy
       - implement `OptimRandomSwaps` baseline optimization strategy
    - benchmarks
       - extend solver benchmarks with testing of optimization strategies + test up to larger problem sizes

- **improved**: 
    - solver framework
      - speed improvement of score computation in case of soft constraints
      - improve unit test coverage & depth of `SolverState` 
    - low-level math
      - speed improvement of fast exponential and power function approximations by up to ~40%, leading to speed improvements in modify_p_selectivity methods of up to ~25-30% 

<!------------------------------------------------------------------------------------------------->
> ## v0.3.7
> *(2025-12-24)*
<!------------------------------------------------------------------------------------------------->

- **new**:
    - solver framework
      - extend `OptimizationStrategy` implementation to support scheduled parameters, that evolve according to a schedule during multi-iteration execution of a strategy 
    - low-level math
      - add `exponential` p-selectivity modification method  
      - add truncated Poisson sampling

- **improved**:
    - docs
      - tweak layout of solver benchmark docs 
    - diversity metrics
      - speed improvement of `geomean_separation_approx` (~60%)
    - initialization strategies
      - sampling heuristics of `InitEager`, leading to slightly higher (~5%) diversity scores and slightly faster execution (~10%)
      - reduce unnecessary array copying in `InitRandomBatched` and `InitRandomEager`

<!------------------------------------------------------------------------------------------------->
> ## v0.3.6
> *(2025-12-22)*
<!------------------------------------------------------------------------------------------------->

- **new**:
    - implement `InitEager` initialization strategy
      - core implementation
      - extend solver benchmarking implementation & docs with new strategy

<!------------------------------------------------------------------------------------------------->
> ## v0.3.5
> *(2025-12-22)*
<!------------------------------------------------------------------------------------------------->

- **new**:
    - implement `InitRandomBatched` initialization strategy
      - core implementation
      - extend solver benchmarking implementation & docs with new strategy
    - extended functionality for `randint_constrained`
      - add `k_context` parameter to enable sampling in batches
      - add `i_forbidden` parameter to have more flexibility on how to sample from non-selected subsets of ranges
    - refactor `modify_p_selectivity` methods 
      - unify various implementations under single API
      - allow in-place modification, for optimal speed
      - add fast-power based implementation (method=20)
      - add accuracy estimation to `modify_p_selectivity` benchmark

- **improved**:
    - rename initialization strategy `InitOneShotRandom` -> `InitRandomOneShot`, in preparation of other random init strategies
    - `benchmark internal` CLI command
        - Redirect output to file using `--file` flag 
        - add `all` benchmark that runs all in one
        - add `make` command to automatically regenerate benchmark results in docs
    - improve accuracy of `modify_p_selectivity` method 20  (recalibrated coefficients) & unify calibration methods
    - speed up `modify_p_selectivity` method 100 by ~10%

<!------------------------------------------------------------------------------------------------->
> ## v0.3.4
> *(2025-12-13)*
<!------------------------------------------------------------------------------------------------->

- **new**:
    - Add 5th solver benchmark problem with simpler constraints, to complete the spectrum
    - Add benchmarking results to documentation & add `make` command to automatically regenerate benchmark results

- **improved**:
    - `benchmark solver run` CLI command
        - Add constraint score reporting 
        - Add uncertainty ranges to reported metrics
        - Add `--turbo` flag and `--speed` parameter
        - Redirect output to file using `--file` flag
        - Allow benchmarking `all` test problems with single command
        - Improve test coverage

<!------------------------------------------------------------------------------------------------->
> ## v0.3.3
> *(2025-12-12)*
<!------------------------------------------------------------------------------------------------->

- **new**:
    - implement `InitOneShotRandom` initialization strategy
    - make `SolverStep`, `InitializationStrategy` & `OptimizationStrategy` instances have a settable random seed
    - add distinction between `internal` and `solver` benchmarks in CLI & implement `benchmark solver` commands

- **improved**:
    - add initialization of `SolverState` to the reported results  
    - reliability & testing of DiversityMetric implementations for sub-standard array sizes
    - improve docs structure

<!------------------------------------------------------------------------------------------------->
> ## v0.3.2
> *(2025-12-11)*
<!------------------------------------------------------------------------------------------------->

- **new**:
    - Framework for built-in benchmark suite for `MaxDivSolver` problems & strategies 
    - First 4 benchmark problems (A1, A2, A3, A4)

- **improved**:
    - structure of CLI implementation; improving encapsulation of internal, low-level, CLI-only benchmarking functionality
    - various low-level algorithmic tweaks & test coverage improvements

<!------------------------------------------------------------------------------------------------->
> ## v0.3.1
> *(2025-12-10)*
<!------------------------------------------------------------------------------------------------->

- **new**:
    - add support for Python 3.14 
    - `MaxDivProblem` class, so problems can be formulated outside of the `MaxDivSolver` and `MaxDivSolverBuilder` classes, paving the way for a built-in benchmark suite.

- **improved**:
    - improve consistency of dimension notations `n`, `d`, `k`, `m` across codebase

<!------------------------------------------------------------------------------------------------->
> ## v0.3.0
> *(2025-12-10)*
<!------------------------------------------------------------------------------------------------->

- **new**:
    - methods for modifying 'selectivity' of an array of probabilities (more or less uniform) 
        - `modify_p_selectivity_power` & `modify_p_selectivity_pwl2`
        - add benchmarking of these methods to CLI & docs
    - make strategies aware of progress fraction within execution of a `SolverStep` to allow scheduling various
       parameters (constraint softness, wide vs selective search, ...)
  
- **improved**:
    - add caching of `Score` object in `SolverState` to avoid redundant score computations when state hasn't changed

<!------------------------------------------------------------------------------------------------->
> ## v0.2.6
> *(2025-12-09)*
<!------------------------------------------------------------------------------------------------->

- **new**:
    - full support for structured multi-component scoring & diversity tie-breaker metrics 
        - implement structured `Score` object with optional `div_tie_breakers` field
        - add support for manual or preset tie-breaker metrics in `MaxDivSolverBuilder`
        - integrate update `Score` class with `SolverState` & `Solver` implementations
    - add initial support for soft constraint handling
    - make a `SolverStep` return score checkpoints correctly via `SolverStepResult`
    - make `MaxDivSolver` return extensive meta-data via `MaxDivSolution`
    - add `non_zero_separation_frac` to `DiversityMetric` implementations and benchmarking suite

- **improved**:
    - ~20% speed improvement of `min_separation` diversity metric  


<!------------------------------------------------------------------------------------------------->
> ## v0.2.5
> *(2025-12-06)*
<!------------------------------------------------------------------------------------------------->

- **improved**:
    - Improve progress tracking implementation for future use in `MaxDivSolver` overall progress reporting
    - Introduction of `SolverStep`, combining a strategy & a duration
    - Introduction of `Score` class for more structured scoring enabling more advanced future strategies

<!------------------------------------------------------------------------------------------------->
> ## v0.2.4
> *(2025-12-05)*
<!------------------------------------------------------------------------------------------------->

- **new**:
    - `SolverStrategy` progress bar support 
    - First version of main `MaxDivSolver` loop

- **improved**:
    - test coverage of `SolverState` and `SolverStrategy` modules

<!------------------------------------------------------------------------------------------------->
> ## v0.2.3
> *(2025-12-04)*
<!------------------------------------------------------------------------------------------------->

- **new**:
    - add `SolverState` snapshot management, so `SolverStrategy` modules can perform trial-modifications to the state.
    - implement first version of `SolverStrategy` API


<!------------------------------------------------------------------------------------------------->
> ## v0.2.2
> *(2025-12-03)*
<!------------------------------------------------------------------------------------------------->

- **new**:
    - add `DiversityMetric.approx_geomean_separation()` based on `fast_log2` and `fast_exp2`
    - add benchmarking of different `DiversityMetric` implementations to CLI & docs

- **improved**:
    - further improve test coverage of `fast_log*` & `fast_exp*` functions
    - make `DiversityMetric` implementation more flexible 

<!------------------------------------------------------------------------------------------------->
> ## v0.2.1
> *(2025-12-02)*
<!------------------------------------------------------------------------------------------------->

- **new**:
    - `fast_exp*` functions in the same spirit as the updated `fast_log*` functions

- **improved**:
    - `fast_log_*` & `fast_log2_*`
        - simplify & speedup by only supporting degree 2 (we don't need more accuracy)
        - update computation of coefficients to guarantee continuity and smoothness of the approximation

<!------------------------------------------------------------------------------------------------->
> ## v0.2.0
> *(2025-11-30)*
<!------------------------------------------------------------------------------------------------->

- **new**:
    - initial `MaxDivSolver` framework

<!------------------------------------------------------------------------------------------------->
> ## v0.1.3
> *(2025-11-27)*
<!------------------------------------------------------------------------------------------------->

- **improved**:
    - further speed up test coverage generation
    - add ReadTheDocs build status badge to README
    - add License badge to README

<!------------------------------------------------------------------------------------------------->
> ## v0.1.2
> *(2025-11-25)*
<!------------------------------------------------------------------------------------------------->

- **new**:
    - implement `randint_constrained_robust`
    - remove excessively slow pure-Python `randint_constrained` & promote `randint_constrained_numba` -> `randint_constrained`

- **improved**:
    - speed up test coverage generation
    - improve visual clarity of `randint_constrained` benchmarking output
    - add benchmarking for `randint_constrained_robust` to `benchmark randint_constrained` CLI command


<!------------------------------------------------------------------------------------------------->
> ## v0.1.1
> *(2025-11-23)*
<!------------------------------------------------------------------------------------------------->

- **new**:
    - implement `Constraints` class
    - implement `sampling.con.randint_constrained_numba` function, offering a 10-100x speedup over the non-`numba` version

<!------------------------------------------------------------------------------------------------->
> ## v0.1.0
> *(2025-11-22)*
<!------------------------------------------------------------------------------------------------->

- **new**:
    - add `benchmark randint_constrained` command to CLI

- **improved**:
    - various minor improvements to docs 

- **internal**:
    - improve test coverage of cli functionality

<!------------------------------------------------------------------------------------------------->
> ## v0.0.9
> *(2025-11-21)*
<!------------------------------------------------------------------------------------------------->

- **new**:
    - add `sampling.con.randint_constrained`, which is like `randint` but supporting constraints

- **improved**:
    - make naming more consistent with other known functionality: `sampling.discrete.sample_int` -> `sampling.uncon.randint`
    - ensure `randint` also works with non-normalized probabilities (with minimal loss of efficiency, especially in `numba`-flavor)

- **internal**:
    - update local dev tooling to declutter project root 

<!------------------------------------------------------------------------------------------------->
> ## v0.0.8
> *(2025-11-18)*
<!------------------------------------------------------------------------------------------------->

- **improved**:
    - `sampling.discrete.sample_int`: speedup for non-uniform sampling without replacement (custom min-heap implementation)

<!------------------------------------------------------------------------------------------------->
> ## v0.0.7
> *(2025-11-17)*
<!------------------------------------------------------------------------------------------------->

- **improved**:
    - test coverage of internal rng functionality
    - improve speed of generating `int32` arrays in `sampling.discrete.sample_int`
  
- **internal**:
    - add `pytest-xdist` and `pytest-rerunfailures` to improve test performance & robustness

<!------------------------------------------------------------------------------------------------->
> ## v0.0.6
> *(2025-11-16)*
<!------------------------------------------------------------------------------------------------->

- **new**:
    - make `numba` and non-`numba` versions of `sampling.discrete.sample_int` public and self-sustained (full-featured)
    - CLI command `numba-status` to report on `numba` installation and configuration status

- **improved**:
    - `sampling.discrete.sample_int`: 
        - significant speedup
        - improve benchmarking + add notes on optimization efforts
    - improve coverage statistics for `numba`-decorated functions
    - make `numba` a non-optional dependency for the sake of code simplicity
    - improve granularity of speed-vs-accuracy trade-off in benchmarking functionality (add `--speed` argument)


<!------------------------------------------------------------------------------------------------->
> ## v0.0.5
> *(2025-11-08)*
<!------------------------------------------------------------------------------------------------->

- **new**:
    - implement `max-div` CLI, allowing package to be installed as tool to define system-wide command 'max-div'

- **improved**:
    - allow benchmarking functionality to output results in Markdown format 
    - switch docs to use `readthedocs` theme
  
<!------------------------------------------------------------------------------------------------->
> ## v0.0.4
> *(2025-11-08)*
<!------------------------------------------------------------------------------------------------->

- **improved**: 
    - further improvements to CI/CD pipeline

<!------------------------------------------------------------------------------------------------->
> ## v0.0.3
> *(2025-11-07)*
<!------------------------------------------------------------------------------------------------->

- **new**: 
    - move to trunk-based development workflow with release branches

<!------------------------------------------------------------------------------------------------->
> ## v0.0.2
> *(2025-11-02)*
<!------------------------------------------------------------------------------------------------->

- **new**: 
    - `sampling.discrete.sample_int`
    - `benchmark.benchmark_sample_int` 

<!------------------------------------------------------------------------------------------------->
> ## v0.0.1
> *(2025-11-01)*
<!------------------------------------------------------------------------------------------------->

- Initial project setup & framework
