# `randint_constrained`

([API reference][max_div.sampling.con.randint_constrained])

Command:
```bash
uv tool install max-div
max-div benchmark --markdown randint_constrained
```
or 
```bash
uv run max-div benchmark --markdown randint_constrained
```

We compare the following situations:
- use `randint`, i.e. we simply sample without taking constraints into account (this is `max_div`'s numba-accelerated implementation)
- use `randint_constrained`, i.e. we sample taking constraints into account

We test both speed (absolute time of a single call) and accuracy (% of cases the generated samples satisfied the constraints).

--8<-- "docs/benchmarks/internal/results/benchmark_randint_constrained_1.md"
--8<-- "docs/benchmarks/internal/results/benchmark_randint_constrained_2.md"
--8<-- "docs/benchmarks/internal/results/benchmark_randint_constrained_3.md"
--8<-- "docs/benchmarks/internal/results/benchmark_randint_constrained_4.md"
