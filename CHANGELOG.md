# Change Log

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
