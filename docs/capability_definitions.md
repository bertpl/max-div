# Capability Definitions

Every cell in the [comparison tables](comparison.md), the README capability table and the
per-solver [profiles](solvers/index.md) traces back to a per-tool record, and every column those
tables show is pinned here to one exact criterion. This page defines what each mark and figure
means; the records supply the evidence per tool.

--8<-- "generated/capability_definitions.md"

## How the size ceilings are measured

All three ceilings come from one measurement protocol, summarized here so every published figure
is reproducible from this page alone.

**Setup.**

- **Problem:** one unconstrained reference problem family —
  [U1](benchmarks/solver/problem_u1.md), fixed d = 2, with k = n/10, the same k-to-n sizing the
  solver benchmarks use.
- **Candidate sizes:** the 1-2-5 grid in n (100, 200, 500, 1000, ...); a published ceiling is
  always a grid value.
- **Machine:** a 16″ MacBook Pro with a 16-core M3 Max CPU, under a 32 GB peak-memory cap
  enforced by a memory watchdog.
- **Timing:** end-to-end from raw vectors, so a tool's distance and setup work is part of its
  cost.
- **Scoring:** the published U1 objective.
- **Monotonicity:** walks up the grid stop at the first fail; a tool that fails a test at one
  size is not retried at larger ones.

**Budgets.** The budgets in the time and quality definitions above bound every run — a run still
going at its budget is killed and counts as a fail.

**The four measurement stages**, each pruning the next:

1. **Memory calibration** — per tool, runs at smaller sizes in its most memory-efficient
   configuration record peak memory; a model fitted to those measurements gives the memory
   ceiling, and sizes it puts above 32 GB are treated as failed in every later stage without
   running.
2. **Time-ceiling walks** — per tool, ascend the grid in its fastest valid configuration at the
   one-minute budget until the first fail.
3. **Anchor runs** — the fifteen-minute reference runs, on a schedule and seeds fixed before any
   quality run is examined, so the anchor cannot be tuned to the results it judges.
4. **Quality runs** — per tool, seeded runs at the one-minute budget in its standard
   configuration, at sizes up to its time ceiling.

**The quality anchor.** For each size, the anchor is the best solution any tool in the suite
produces within fifteen minutes — the dedicated anchor runs and the one-minute quality runs both
count. Its stated properties:

- The anchor is not pinned to any tool: whichever solution is best sets it. The dedicated
  fifteen-minute runs include max-div's own parallel runs, so the reference is often produced by
  the tool under comparison — stated here openly.
- At sizes where no fifteen-minute run betters the one-minute field, the best one-minute tool is
  itself the anchor, so it passes by construction.
- A re-measurement that improves the anchor lowers every tool's quality ceiling with it.

**Seeds.** Stochastic tools run five seeds per size, deterministic ones three; every pass or fail
is judged on the median seed. Anchor runs use one pre-registered seed each.
