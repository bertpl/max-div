# Head-to-Head — vs. Python Heuristics — Tables

The tables of the [Python-heuristics comparison](tier2.md):

- quality is minimum separation on U1;
- a `max-div` cell is the median over seeds at the quoted budget;
- an entrant cell is the mean over seeds, with the mean measured time in parentheses.

## I. Summary per size

The overtake budget is the smallest budget of the series at which `max-div`'s median reaches the best one-shot result; a dash means the median stays below the best one-shot result at every budget up to 60 s.

--8<-- "docs/benchmarks/third_party/head_to_head/results/tier2_summary.md"

## II. Entrants per size

A dash marks a size outside the tool's scaling time limit.

--8<-- "docs/benchmarks/third_party/head_to_head/results/tier2_entrants.md"
