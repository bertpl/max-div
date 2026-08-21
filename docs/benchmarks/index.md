# Benchmark Results

This section contains three categories of benchmark results:

- [**3rd Party Solvers**](./third_party/comparison.md) — `max-div` against other freely available
  subset-selection tools: the qualitative capability comparison, the solver-scaling
  protocol and results, the measured head-to-head benchmarks, and per-solver profiles.
- [**Built-in Problems**](./solver/test_problems.md) — `max-div`'s own benchmark suite: the
  built-in test problems, and its initialization/optimization strategies and presets measured on
  those problems.
- [**Internal Primitives**](./internal/overview.md) — micro-benchmarks of internal building
  blocks (diversity-metric kernels, random samplers, probability selectivity).

The built-in-problem (`solver`) and internal-primitive (`internal`) benchmarks are triggerable via the CLI:
```
max-div benchmark <category> <sub-command>
```

All benchmarks are run on a 16" MacBook Pro with M3-class CPU.
