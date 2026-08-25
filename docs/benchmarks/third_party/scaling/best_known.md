# Solver Scaling — Best-Known Solutions

The **best-known solution** per problem size is the reference the quality verdicts are judged against. The values below come from the extended runs of the [measurement protocol](protocol.md) (section IV.D): every configuration, run with the extended budget `T_extended = 15 × T_max`, sizes increasing until a run is killed or crashes.

*Measured with max-div v0.14.2, on the reference machine the [protocol](protocol.md) names.*

The measured-time column shows each winning run's end-to-end time. A run that finishes somewhat past the extended budget still counts — the protocol's stopping rule keeps every completed solution, since a late solution can only raise the reference for every solver alike.

--8<-- "generated/scaling_best_known.md"
