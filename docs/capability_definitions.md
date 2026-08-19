# Capability Definitions

Every cell in the [comparison tables](comparison.md), the README capability table and the
per-solver [profiles](solvers/index.md) is pinned here to one exact criterion. This page defines
what each mark and figure means; the reasoning behind each tool's own marks is on its profile
page.

--8<-- "generated/capability_definitions.md"

## How the size ceilings are measured

All three ceilings come from one measurement procedure, described here so the published figures
are reproducible from this page alone.

**The test problem.** Every measurement uses [U1](benchmarks/solver/problem_u1.md), one of the
built-in benchmark problems: n points in 2 dimensions, from which a tool must select the
k = n/10 most diverse ones. Candidate problem sizes lie on a logarithmic grid with three values
per decade — n = 100, 200, 500, 1000, ... — and a published ceiling is always one of these
values.

**Ground rules.**

- Every run executes on the same machine — a MacBook Pro with a 16-core M3 Max CPU — with peak
  memory capped at 32 GB.
- A run is timed end-to-end from raw input vectors: distance computation and any other setup a
  tool needs count toward its time.
- A run that exceeds its time budget or the memory cap is stopped and counts as a failure.
- Measurements proceed from small to large sizes; once a tool fails at some size, larger sizes
  are not attempted.
- Solution quality is always scored as the smallest distance between any two selected points —
  the **max-min objective**, the one objective every tool in this comparison can pursue. A tool
  that optimizes a different objective (the objective columns above show which) is still scored
  on max-min.

**The four measurements.**

1. **Time ceiling.** Starting from n = 100, the tool runs once per size with a one-minute
   budget, in its fastest valid configuration. The time ceiling is the largest size at which it
   returns a valid selection. Peak memory is recorded on every one of these runs.
2. **Memory ceiling.** A per-tool memory model is fitted to the peaks recorded at the largest
   sizes the tool completed, checked against the tool's documented data structures, and
   extrapolated: the memory ceiling is the largest candidate size whose predicted peak stays
   within 32 GB. No run is executed at the ceiling itself — at full memory most tools are far
   too slow to run at all. A tool whose most memory-efficient configuration differs from its
   fastest one gets a few extra small runs in that configuration.
3. **Best-known solutions.** The quality ceiling (next) compares each tool's solution to the
   best solution *any* tool can find at that size when given plenty of time. To establish that
   reference, every tool whose result improves with extra time gets one fifteen-minute run per
   size, with seeds fixed in advance — before any one-minute result has been looked at — so the
   reference cannot be tuned to the results it judges.
4. **Quality ceiling.** Each tool runs at each size up to its time ceiling with the one-minute
   budget, in its standard configuration — five seeded runs for stochastic tools, three for
   deterministic ones. A size counts as passed when the median solution closes at least 90% of
   the quality gap between a plain random selection (the median of 32 uniform-random draws) and
   the best-known solution at that size. The quality ceiling is the largest size that passes; a
   tool that passes at no size shows a dash with a footnote.

**Rules for the best-known solution.**

- It is whatever solution scores highest at that size, whether it was found in a fifteen-minute
  run or in a one-minute quality run. It is not pinned to any particular tool; max-div takes
  part like every other tool.
- If at some size no fifteen-minute run beats the best one-minute result, the best one-minute
  tool is being compared against itself and passes there by construction.
- Re-measuring later with better tools can raise the best-known solution, which lowers every
  tool's quality ceiling with it.
