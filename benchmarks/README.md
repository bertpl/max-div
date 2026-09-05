# Comparison-benchmark harness

Tooling for benchmarking max-div against other freely available subset-selection tools
(see the docs' *Comparison* page for the landscape). This directory is repo-tracked for
transparency and reproducibility, but it is **not part of the published package** — its
dependencies live in the `benchmarks` dependency group, never in package metadata.

## Layout

- `common/` — shared infrastructure: budget series and the comparison protocol's constants,
  subset-quality evaluation, run records, benchmark problem construction, registry names.
- `adapters/` — one adapter per compared tool/baseline, all implementing
  `SelectionAdapter`; each converts the problem to the input its tool needs inside its timed
  `select`, so the conversion counts toward the tool's measured time.
- `runners/` — drivers that execute max-div (over a budget series, with one or several
  workers) or an adapter (single-shot) against a problem and emit run records.
- `figures/` — the shared chart styling and the anytime-curve plotter (budget-series curves,
  single-shot dots, reference lines and markers).
- `tier1/` — the comparison against the exact solvers' certified optima.
- `tier2/` — the comparison against the one-shot tools on U1, the uniform unconstrained
  benchmark problem.
- `tier3/` — the comparison against the MDPLIB best-known values.
- `mdplib/` — MDPLIB instance loader plus the vendored best-known-value table with provenance.
- `solver_scaling/` — the solver-scaling measurement stages behind the scaling pages.
- `campaign.py` — runs the three tiers back to back as one overnight campaign.

## Running

```bash
uv sync --group benchmarks
uv run --group benchmarks python -m benchmarks.tier2.smoke
```

Outputs (JSONL records + figures) are written under `./reports/benchmarks/` (gitignored);
only curated result tables/figures are ever promoted into `docs/`. The exception is
third-party reference data: the entrants' (the compared third-party tools') and the exact
solvers' results are tracked under
`tier1/data/`, `tier2/data/` and `tier3/data/` (alongside `mdplib/data/`), because the
published comparison pages keep those values fixed across re-measurements of max-div —
republishing must not depend on files that exist only on the machine of the original
run.

The full campaign runs detached, so no session limit can end it:

```bash
nohup uv run --group benchmarks --python 3.14 python -u -m benchmarks.campaign > campaign.log 2>&1 &
```

Every driver skips cells (one problem at one size) already on file, so an interrupted run
resumes by rerunning. To
re-measure max-div alone (after solver changes), run the max-div-only drivers and then the
reports — the third-party side comes from the tracked reference data:

```bash
uv run --group benchmarks python -m benchmarks.tier1.rerun
uv run --group benchmarks python -m benchmarks.tier2.rerun
uv run --group benchmarks python -m benchmarks.tier3.full   # skips the entrant records already on file
uv run --group benchmarks python -m benchmarks.tier1.report
uv run --group benchmarks python -m benchmarks.tier2.report
uv run --group benchmarks python -m benchmarks.tier3.report
```

Refreshing the tracked reference data itself is a deliberate act: run the full drivers and
copy the fresh `third_party_*` / exact-solver outputs from `reports/benchmarks/` over the
`data/` folders.

## Measurement protocol

The three tiers share one protocol (`common/protocol.py`), aligned with the solver-scaling
protocol where the two overlap:

- max-div runs two **budget series** on the 1-2-5 grid, both ending at T_max: one worker from
  the smallest budget, and `N_WORKERS` from the first budget larger than the time spawning the
  workers takes.
  Every record stores the *measured* end-to-end wall-clock, never the nominal budget.
- Single-shot entrants run once per seed; their measured runtime, conversion included, is
  recorded the same way.
- `SEEDS` per cell; the tables quote medians at `QUOTED_BUDGETS_SEC`.
- Selection quality is always evaluated by `common/quality.py` under max-div's own
  diversity metrics, for every tool alike.
