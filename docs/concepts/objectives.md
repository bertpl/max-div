# Objectives & the Diversity-Problem Landscape

max-div's diversity metrics correspond to well-studied objectives from the operations-research
literature on *dispersion problems* -- plus one that appears to be genuinely novel. This page names
each objective's literature family, so that results and terminology can be related to published
work (and to other tools, which typically implement exactly one of these families).

## A note on the name "Maximum Diversity Problem"

In the OR literature, the unqualified term **Maximum Diversity Problem (MDP)** conventionally
refers to one specific objective: maximize the *sum of pairwise distances* among the selected
points (also called *MaxSum diversity* or *remote-clique*). max-div covers that objective with
`MEAN_PAIRWISE_DISTANCE` -- but its separation-based metrics optimize a different family:
functions of each selected point's *nearest-neighbor distance within the selection*. If you come
from the MDP literature, read "maximum diversity" here in its broad sense (the whole family of
diversity-maximizing subset-selection problems), not as a pointer to the MaxSum objective alone.

## Where each metric sits

| max-div metric | Literature name(s) | Family |
|---|---|---|
| `MIN_SEPARATION` | **p-dispersion**, Max-Min diversity, *remote-edge* | NN-separation |
| `MEAN_SEPARATION` | **Max-SumMin** dispersion, *remote-pseudoforest*, "p-defense-sum" | NN-separation |
| `GEOMEAN_SEPARATION` (and its `APPROX_` variant) | no established name -- see below | NN-separation |
| `MEAN_PAIRWISE_DISTANCE` | **MaxSum diversity / classical MDP**, *remote-clique* | all-pairs |

### `MIN_SEPARATION` -- p-dispersion (Max-Min)

The classical **p-dispersion problem**: maximize the minimum distance between any two selected
points. One of the oldest and best-studied dispersion objectives (Erkut, 1990), with a rich exact
and heuristic literature. The greedy farthest-point construction carries a 2-approximation
guarantee (Ravi, Rosenkrantz & Tayi, 1994) -- the best possible unless P = NP.

### `MEAN_SEPARATION` -- Max-SumMin dispersion

Maximize the *sum (equivalently, mean) of each selected point's nearest-neighbor distance* within
the selection. Known as **Max-SumMin** dispersion or *remote-pseudoforest*, and as "p-defense-sum"
in the facility-location strand (Lei & Church, 2015). Strongly NP-hard, and markedly less charted
than p-dispersion: the first constant-factor approximation dates only from 2016 (Bhaskara et al.,
NIPS 2016), with the composable-coreset gap closed in 2023 (Mahabadi & Narayanan, APPROX 2023).

Beware a naming trap when searching the literature: **Max-MinSum** dispersion (also written
"MaxMinSum", *remote-star*) is a *different* objective -- the minimum over selected points of the
*sum* of distances to all other selected points -- despite the confusingly similar name.

### `GEOMEAN_SEPARATION` -- Nash social welfare over separations

Maximize the *geometric mean* of the selected points' nearest-neighbor separations. We have found
no established name -- nor any implementation -- for this objective in the dispersion literature;
it appears to be **novel as an implemented objective**, and it is max-div's default.

It can be understood two ways:

- **Max-SumMin on log-distances**: the geometric mean is the arithmetic mean after a log
  transform, so structurally it inherits the Max-SumMin family.
- **Nash social welfare** over separations: maximizing a product of per-point "utilities" is the
  Nash social-welfare rule from fair-division theory -- the canonical compromise between the
  egalitarian rule (`MIN_SEPARATION`: only the worst-off point matters) and the utilitarian rule
  (`MEAN_SEPARATION`: only the total matters, tolerating poorly-off points). Proportional
  fairness, applied to spread: every point's separation matters, and a near-duplicate (separation
  near zero) collapses the whole score, while no single huge separation can buy that back.

### `MEAN_PAIRWISE_DISTANCE` -- classical MaxSum diversity

Maximize the mean (equivalently, sum) of distances over all selected *pairs* -- the objective the
OR literature calls **the** Maximum Diversity Problem (*remote-clique*). This is the largest body
of published benchmark results (e.g. the MDPLIB instance library), and the objective to choose for
comparability with that literature. Its behavior differs from the separation family in one
important way: it maximizes *total* spread and does not penalize near-duplicates per se -- see
[Diversity & Distance](diversity.md) for the practical guidance.

## References

- Erkut, E. (1990). The discrete p-dispersion problem. *European Journal of Operational
  Research*, 46(1), 48-60.
- Ravi, S. S., Rosenkrantz, D. J., & Tayi, G. K. (1994). Heuristic and special case algorithms
  for dispersion problems. *Operations Research*, 42(2), 299-310.
- Chandra, B., & Halldórsson, M. M. (2001). Approximation algorithms for dispersion problems.
  *Journal of Algorithms*, 38(2), 438-465. (Source of the *remote-edge* / *remote-pseudoforest* /
  *remote-clique* / *remote-star* taxonomy.)
- Lei, T. L., & Church, R. L. (2015). On the unified dispersion problem: Efficient formulations
  and exact algorithms. *European Journal of Operational Research*, 241(3), 622-630.
- Bhaskara, A., Ghadiri, M., Mirrokni, V., & Svensson, O. (2016). Linear relaxations for finding
  diverse elements in metric spaces. *NIPS 2016*.
- Mahabadi, S., & Narayanan, S. (2023). Improved diversity maximization algorithms for matching
  and pseudoforest. *APPROX 2023*.
- Martí, R., Martínez-Gavara, A., Pérez-Peló, S., & Sánchez-Oro, J. (2022). A review on discrete
  diversity and dispersion maximization from an OR perspective. *European Journal of Operational
  Research*, 299(3), 795-813. (MDPLIB 2.0.)
