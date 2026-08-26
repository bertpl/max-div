# Solver Scaling — Quality

The **largest n closing 50% / 90% of the quality gap** is the largest problem size up to which a configuration's median solution quality reaches that gap-closure fraction's verdict threshold at every measured size, per the [measurement protocol](protocol.md) (section IV.D).

The scoring objective (minimum separation under L2) was chosen because it is the one most benchmarked tools can pursue. Tools designed specifically for this objective are measured on the objective they were built for; tools supporting a wider range of distance and diversity metrics (see the [comparison table](../comparison.md)'s metric axes) are compared on this shared objective only.

--8<-- "generated/scaling_quality.md"

## Detailed Results

Each cell holds the percentage of the random-to-best-known quality gap a configuration's median closes at that size, rounded down to 0.1% — **bold** reaches the 90% fraction, *italic* only the 50% one. An empty cell is a size the configuration was not judged at — beyond its time limit, or no completed run.

Percentages rising with n say more about the reference than about the solver: the best-known solutions come from a fixed extended budget, so they are less converged as n grows, and at the largest sizes they come from the same fast tools being judged.

A high percentage there means the solver matches the best *known* solution, not that it approaches the optimum — the reason the [protocol](protocol.md) (section IV.D.4) ends a passing range at the first failing size.

--8<-- "generated/scaling_quality_gaps.md"

## Best-Known Solutions

The **best-known solution** per problem size is the reference the quality verdicts are judged against; beside the best-known value sit `Q_random` and the two thresholds the pair yields. The values below are the best over every measured run — the extended runs and the reference-budget quality runs of the [measurement protocol](protocol.md) (section IV.D).

The measured-time column shows each winning run's end-to-end time. A run that finishes somewhat past the extended budget still counts — see the overrun rule in the [protocol](protocol.md), section IV.D.2.

--8<-- "generated/scaling_best_known.md"
