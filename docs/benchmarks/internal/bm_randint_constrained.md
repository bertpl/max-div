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
- use `randint_numba`, i.e. we simply sample without taking constraints into account
- use `randint_constrained`, i.e. we sample taking constraints into account
- use `randint_constrained_robust`, i.e. we sample taking constraints into account with multiple trials to improve success rate

We test both speed (absolute time of a single call) and accuracy (% of cases the generated samples satisfied the constraints).

{% include-markdown "./results/benchmark_randint_constrained_1.md" %}
{% include-markdown "./results/benchmark_randint_constrained_2.md" %}
{% include-markdown "./results/benchmark_randint_constrained_3.md" %}
{% include-markdown "./results/benchmark_randint_constrained_4.md" %}
