# Roadmap

## v0.0.x

- Initial project setup
- Implement & optimize unconstrained sampling (`randint`)
- Implement benchmarking

## v0.1.0

- First version with constrained sampling (`randint_constrained`) + benchmarking functionality

## v0.1.x

- Tweaks & optimizations to constrained sampling

## v0.2.0

- First version of the `MaxDivSolver` framework all core classes (metrics, distances, constraints, solver state, ...)

## v0.2.x
 
- Implement all remaining parts of framework
  - multi-component prioritized constraints
  - soft constraint handling
  - tie-breaker diversity scores
  - full `MaxDivSolverBuilder` class for `MaxDivSolver` instances
  - probability selectivity modification functionality
  - support for `SolverSteps` with scheduled (=progress-dependent) parameters

## v0.3.0

- First version of the `MaxDivSolver` framework & all core classes (metrics, distances, constraints, solver state, scheduling, ...)

## v0.3.x

- Implementation of various initialization & optimization strategies
- Implementation of convenient strategy presets in `MaxDivSolverBuilder` class.

## v0.4.0

- First version with full-featured `MaxDivSolver`, including a variety of initialization & optimization strategies.

## v0.4.x

- Documentation push
  - Getting started / Tutorial
  - Use cases with benchmarks & visualizations
  - Package architecture

## v0.9.0

- First full-featured release with full docs

## v0.9.x

- Bug fixes & minor improvements based on initial usage

## v1.0.0

- First stable release
- To be released before downstream use cases start relying on the package