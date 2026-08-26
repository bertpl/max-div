# Solver Scaling — Quality

The **largest n closing 50% / 90% of the quality gap** is the largest problem size up to which a configuration's median solution quality reaches that gap-closure fraction's verdict threshold at every measured size, per the [measurement protocol](protocol.md) (section IV.D).

The scoring objective (minimum separation under L2) was chosen because it is the one most benchmarked tools can pursue. Tools designed specifically for this objective are measured on home ground; tools supporting a wider range of distance and diversity metrics (see the [comparison table](../comparison.md)'s metric axes) are compared on this shared objective only.

--8<-- "generated/scaling_quality.md"

## Gap Closure per Size

Each cell holds the fraction of the random-to-best-known quality gap a configuration's median closes at that size — **bold** reaches the 90% fraction, *italic* only the 50% one. An empty cell is a size beyond the configuration's time limit.

--8<-- "generated/scaling_quality_gaps.md"

## Best-Known Solutions

The **best-known solution** per problem size is the reference the quality verdicts are judged against. The values below are the best over every measured run — the extended runs and the reference-budget quality runs of the [measurement protocol](protocol.md) (section IV.D).

The measured-time column shows each winning run's end-to-end time. A run that finishes somewhat past the extended budget still counts — see the overrun rule in the [protocol](protocol.md), section IV.D.2.

--8<-- "generated/scaling_best_known.md"
