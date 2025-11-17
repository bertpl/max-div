# Change Log

<!------------------------------------------------------------------------------------------------->
> ## v0.0.7
> *(under development)*
<!------------------------------------------------------------------------------------------------->

- **improved**:
  - test coverage of internal rng functionality
  
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
