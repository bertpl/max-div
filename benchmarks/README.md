# Comparison-benchmark harness

Tooling for benchmarking max-div against other freely available subset-selection tools
(see the docs' *Comparison* page for the landscape). This directory is repo-tracked for
transparency and reproducibility, but it is **not part of the published package** — its
dependencies live in the `benchmarks` dependency group, never in package metadata.

## Layout

- `common/` — shared infrastructure: budget ladders, subset-quality evaluation,
  run records, benchmark problem construction.
- `adapters/` — one adapter per competing tool/baseline, all implementing
  `SelectionAdapter`.
- `runners/` — drivers that execute max-div (anytime ladder) or an adapter
  (single-shot) against a problem and emit run records.
- `figures/` — plotting of anytime curves (max-div) vs. single-shot dots (competitors).
- `tier2/` — benchmark scenarios vs. Python subset-selection heuristics.

## Running

```bash
uv sync --group benchmarks
uv run --group benchmarks python -m benchmarks.tier2.smoke
```

Outputs (JSONL records + figures) are written under `./reports/benchmarks/` (gitignored);
only curated result tables/figures are ever promoted into `docs/`.

## Measurement protocol

- max-div runs a **budget ladder** (2× steps); every record stores the *measured*
  wall-clock reported by the solver, never the nominal budget.
- Single-shot competitors run once per seed; their measured runtime is recorded the
  same way.
- Selection quality is always evaluated by `common/quality.py` under max-div's own
  diversity metrics, for every tool alike.
