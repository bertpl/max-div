# Proving Feasibility

## The Question

Given [constraints](constraints.md), can *any* selection of `k` items satisfy all of them at once?

This is worth asking separately from solving, because the solver never answers it. Handed
constraints that cannot all be met, it returns the least infeasible selection it can find and a
constraints score below 1.

That looks exactly like a run which merely needed more time. Knowing no selection could have done
better turns a tempting "raise the budget" into a decision to change the constraints instead.

The question is NP-complete in general: with overlapping
[constraint groups](glossary.md#overlapping-constraints) it encodes problems no known method solves
quickly in every case. So the honest answer is three-valued, and the two definite answers are
*proofs* rather than strong hints.

## Pricing a Violation

Write $c_i(x)$ for how many selected items fall in constraint $i$'s group, with bounds
$[\mathit{lo}_i, \mathit{hi}_i]$ and [weight](glossary.md#constraint-weight) $w_i$. The total
weighted violation of a selection $x$ is

$$v(x) = \sum_i w_i \Big( \max(0,\; \mathit{lo}_i - c_i(x)) + \max(0,\; c_i(x) - \mathit{hi}_i) \Big)$$

and the constraints are satisfiable exactly when some $x$ with $k$ items has $v(x) = 0$.

Attach a price $\lambda^-_i \ge 0$ to each unit of shortfall and $\lambda^+_i \ge 0$ to each unit of
excess. Two facts turn those prices into a proof:

- **Priced cost never exceeds real violation**, provided each price is capped at the weight it
  stands in for. That cap is the whole reason prices are clamped to $[0, w_i]$ rather than left to
  grow.
- **Priced cost collapses to a per-item sum.** Regrouped by item, each one picks up a score
  $s_j = \sum_{i \,:\, j \in S_i} (\lambda^-_i - \lambda^+_i)$, and the priced cost becomes a
  constant minus the scores of the items chosen.

Only that last term depends on the selection, and it is subtracted -- so the priced cost is
smallest for the selection taking the `k` highest-scoring items. The worst case over all selections
is a sort, not a search, which is what makes the bound computable at all:

$$g(\lambda) = \sum_i \big( \lambda^-_i \mathit{lo}_i - \lambda^+_i \mathit{hi}_i \big) \;-\; \sum_{j \,\in\, \text{top-}k(s)} s_j$$

Every selection of `k` items violates the constraints by at least $g(\lambda)$, whatever prices
were used. Different prices give different bounds, so the search raises the price of constraints
below their minimum, lets over-full ones decay, and pushes $g$ as high as it will go.

## Reading the Certificate

If any prices drive $g(\lambda)$ above zero, every possible selection violates something, and the
constraints are **provably** unsatisfiable. The prices are reported alongside the verdict, and
re-checking them takes no trust in the search that found them: recompute the scores, sum the `k`
largest, evaluate $g$.

A bug in the price search can only produce a *worse* bound, never a wrong one, since the inequality
above holds for any prices in $[0, w_i]$. The one step that would break it is evaluating $g$ at
anything other than the exact top `k`: a near-miss selection makes the subtracted sum too small and
$g$ too large, manufacturing a proof that is not true. Every source of randomness is kept away from
that step for exactly this reason.

A positive $g$ is also a number, not just a sign -- a floor on the violation of any selection, and
so a cap on the constraints score the solver could ever report.

## Finding a Selection Is a Separate Job

The bound is one-sided. A positive $g$ proves infeasibility, but $g \le 0$ proves nothing at all --
it only means these prices failed to rule the problem out. A satisfying selection therefore has to
be produced, not deduced.

The prices help here too, in a different role. A high-scoring item is one belonging to constraints
below their minimum and not to constraints over their maximum, which makes the top-`k` scoring
selection a good first guess. It is rarely satisfying outright, so it is repaired: repeatedly take
the worst-violated constraint and swap one of its items for another, where the swap is estimated to
lower the total violation. Several rounds run, from different prices and with noise added.

Any selection reaching zero violation is a witness, and a witness needs no certificate -- it can be
checked directly against every constraint. Randomization is safe here precisely because nothing in
this phase claims a bound.

## Three Answers, Two of Them Proofs

| Verdict | What it means | What backs it |
|---------|---------------|---------------|
| **Feasible** | A selection satisfying every constraint exists. | The selection itself, checkable against the constraints. |
| **Infeasible** | No selection can satisfy every constraint. | Prices with $g > 0$, plus the violation floor they certify. |
| **Unknown** | Neither was established. | Nothing. |

**Unknown is not a weak "probably infeasible".** It carries no information whatsoever, and a caller
must behave exactly as if the check had never run. Two structural cases land there:

- problems satisfiable only in a fractional sense, where prices provably cannot separate them;
- satisfiable problems whose witnesses need several coordinated swaps at once, which the repair
  step does not attempt.

Exhausted search limits produce the rest.

## Using It

Ask about a problem directly:

```python
report = problem.check_feasibility()
print(report)
```

`thorough=True` changes only what happens after infeasibility is proven: the default stops there,
while `thorough=True` keeps searching for a tighter violation floor.

The same check can start a solve, rather than only describe the problem:

```python
from max_div.solver import InitializationStrategy, MaxDivSolverBuilder, seconds

solver = (
    MaxDivSolverBuilder(problem)
    .with_preset(seconds(5))
    .set_initialization_strategy(InitializationStrategy.most_feasible())  # after with_preset
    .build()
)
```

The solver then begins from a satisfying selection where one can be constructed, spending its whole
budget on diversity instead of first searching for feasibility; where infeasibility is proven, it
begins from the least infeasible selection found. Where the verdict is unknown -- and on problems
with no constraints -- it delegates to the strategy given as `fallback`, `random_one_shot` by
default.

## Further Reading

- Boyd & Vandenberghe, *Convex Optimization* (Cambridge University Press, 2004), chapter 5 -- the
  duality theory behind the bound, including why a relaxed problem's optimum bounds the original's.
- Fisher, "The Lagrangian Relaxation Method for Solving Integer Programming Problems",
  *Management Science* 27(1), 1981 -- the classic survey of the technique applied to integer
  programs, and of the subgradient search used to improve the prices.
- Geoffrion, "Lagrangean Relaxation for Integer Programming", *Mathematical Programming Study* 2,
  1974 -- establishes how strong the bound can get, and hence which problems it can never rule out.
