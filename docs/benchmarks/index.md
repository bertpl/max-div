# Benchmark Results

This section contains three categories of benchmark results:

- [**3rd Party Solvers**](./comparison/overview.md) — `max-div` against other freely available
  subset-selection tools: Python heuristics, exact MIP/CP solvers, and the MDPLIB
  best-known values from the literature.
- [**Built-in problems**](./solver/test_problems.md) — `max-div`'s own benchmark suite: the
  built-in test problems, and its initialization/optimization strategies and presets measured on
  them.
- [**Internal Primitives**](./internal/overview.md) — micro-benchmarks of internal building
  blocks (diversity-metric kernels, random samplers, probability selectivity).

The solver and internal-primitive benchmarks are triggerable via the CLI:
```
max-div benchmark <category> <sub-command>
```

All benchmarks are run on a 16" MacBook Pro with M3-class CPU.
