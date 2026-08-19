# Comparison-benchmark harness

Tooling for benchmarking max-div against other freely available subset-selection tools
(see the docs' *Comparison* page for the landscape). This directory is repo-tracked for
transparency and reproducibility, but it is **not part of the published package** — its
dependencies live in the `benchmarks` dependency group, never in package metadata.

## Layout

- `common/` — shared infrastructure: budget series, subset-quality evaluation,
  run records, benchmark problem construction.
- `adapters/` — one adapter per competing tool/baseline, all implementing
  `SelectionAdapter`.
- `ceilings/` — the size-ceilings campaign: candidate-size grid, per-tool run
  configurations, a budget/memory-enforcing subprocess runner, and the stage drivers
  behind the capability table's measured ceiling columns.
- `runners/` — drivers that execute max-div (anytime budget series) or an adapter
  (single-shot) against a problem and emit run records.
- `figures/` — plotting of anytime curves (max-div) vs. single-shot dots (competitors).
- `tier1/` — benchmark scenarios vs. exact solvers (CP-SAT, SCIP).
- `tier2/` — benchmark scenarios vs. Python subset-selection heuristics.
- `tier3/` — benchmark scenarios vs. MDPLIB best-known values.
- `mdplib/` — MDPLIB instance loader plus the vendored best-known-value table.

## Running

```bash
uv sync --group benchmarks
uv run --group benchmarks python -m benchmarks.tier2.smoke
```

Outputs (JSONL records + figures) are written under `./reports/benchmarks/` (gitignored);
only curated result tables/figures are ever promoted into `docs/`. The exception is
third-party reference data: competitor and exact-solver results are tracked under
`tier1/data/` and `tier2/data/` (alongside `mdplib/data/`), because the published
comparison pages keep those values fixed across re-measurements of max-div —
republishing must not depend on files that exist only on the machine of the original
run.

To re-measure max-div alone (after solver changes), run the max-div-only drivers and then
the reports — the third-party side comes from the tracked reference data:

```bash
uv run --group benchmarks python -m benchmarks.tier1.rerun
uv run --group benchmarks python -m benchmarks.tier2.rerun
uv run --group benchmarks python -m benchmarks.tier3.full   # has no third-party runs, so it is already max-div-only
uv run --group benchmarks python -m benchmarks.tier1.report
uv run --group benchmarks python -m benchmarks.tier2.report
uv run --group benchmarks python -m benchmarks.tier3.report
```

Refreshing the tracked reference data itself is a deliberate act: run the full drivers and
copy the fresh `third_party_*` / exact-solver outputs from `reports/benchmarks/` over
`tier1/data/` and `tier2/data/`.

## Measurement protocol

- max-div runs a **budget series** (2× steps); every record stores the *measured*
  wall-clock reported by the solver, never the nominal budget.
- Single-shot competitors run once per seed; their measured runtime is recorded the
  same way.
- Selection quality is always evaluated by `common/quality.py` under max-div's own
  diversity metrics, for every tool alike.
