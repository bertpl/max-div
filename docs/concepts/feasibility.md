# Proving Feasibility

## The Question

Given [constraints](constraints.md), can *any* selection of `k` items satisfy all of them at once?

This is worth asking separately from solving, because the solver never answers it. Handed
constraints that cannot all be met, it returns the least infeasible selection it can find and a
constraints score below 1 -- which looks exactly like a run that merely needed more time. Knowing
that no selection could have done better turns a tempting "raise the budget" into a decision to
change the constraints instead.

The question is NP-complete in general: with overlapping [constraint groups](glossary.md#overlapping-constraints)
it encodes problems no known method solves quickly in every case. So the honest answer is
three-valued, and the two definite answers are *proofs* rather than strong hints.

## Pricing a Violation

Write $c_i(x)$ for how many selected items fall in constraint $i$'s group, with bounds
$[\mathit{lo}_i, \mathit{hi}_i]$ and [weight](glossary.md#constraint-weight) $w_i$. The total
weighted violation of a selection $x$ is

$$v(x) = \sum_i w_i \Big( \max(0,\; \mathit{lo}_i - c_i(x)) + \max(0,\; c_i(x) - \mathit{hi}_i) \Big)$$

and the constraints are satisfiable exactly when some $x$ with $k$ items has $v(x) = 0$.

The idea behind the proof machinery is to stop treating violation as a penalty and start treating
it as a **price**. Attach a price $\lambda^-_i \ge 0$ to each unit of shortfall and
$\lambda^+_i \ge 0$ to each unit of excess. As long as no price exceeds the weight it stands in
for -- that is, $0 \le \lambda^\pm_i \le w_i$ -- paying the prices is never more expensive than the
real violation:

$$v(x) \;\ge\; \sum_i \Big( \lambda^-_i \big(\mathit{lo}_i - c_i(x)\big) + \lambda^+_i \big(c_i(x) - \mathit{hi}_i\big) \Big)$$

The inequality holds term by term. Where a constraint is short, the real cost is
$w_i \cdot (\mathit{lo}_i - c_i)$ and the priced cost is at most that. Where it is not short, the
real cost is 0 while the priced term is negative, so the bound only loosens. The cap at $w_i$ is
what makes the whole construction work, and it is why the prices are clamped rather than left to
grow.

## Why the Bound Can Be Computed

A lower bound is only useful if it holds for *every* selection, and there are astronomically many.
What rescues this is that the right-hand side above collapses into something per-item.

Group the terms by item instead of by constraint. Each item $j$ picks up a score summing the
prices of every constraint it belongs to:

$$s_j = \sum_{i \,:\, j \in S_i} \big( \lambda^-_i - \lambda^+_i \big)$$

and the priced cost of a selection becomes a constant minus the scores of the items chosen:

$$\sum_i \big( \lambda^-_i \mathit{lo}_i - \lambda^+_i \mathit{hi}_i \big) \;-\; \sum_{j \in x} s_j$$

Only the last term depends on which items are selected, and it is subtracted -- so the selection
that makes the priced cost *smallest* is simply the one taking the `k` highest-scoring items. The
worst case over all selections is a sort, not a search. That value is the bound:

$$g(\lambda) = \sum_i \big( \lambda^-_i \mathit{lo}_i - \lambda^+_i \mathit{hi}_i \big) \;-\; \sum_{j \,\in\, \text{top-}k(s)} s_j$$

Every selection of `k` items violates the constraints by at least $g(\lambda)$, whatever prices
were used. Different prices give different bounds, so the machinery searches for prices that push
$g$ as high as it will go, nudging up the price of starved constraints and letting over-satisfied
ones decay.

## Reading the Certificate

If any prices drive $g(\lambda)$ above zero, then every possible selection violates something, and
the constraints are **provably** unsatisfiable. The prices are the proof: they are reported
alongside the verdict, and re-checking them needs no trust in the search that found them --
recompute the scores, sum the `k` largest, and evaluate $g$.

That is why the answer is a certificate rather than a claim. A bug in the price search can only
produce a *worse* bound, never a wrong one, because the inequality above holds for any prices
inside their boxes. The one thing that would break it is evaluating $g$ at anything other than the
exact top `k` -- a near-miss selection makes the subtracted sum too small and $g$ too large, which
would manufacture a proof of infeasibility that is not true. The implementation keeps every source
of randomness away from that step for exactly this reason.

A positive $g$ carries a second, quantitative payload: it is a floor on the violation of *any*
selection, so it also caps the constraints score the solver could ever report. An infeasible
problem thus comes with a best achievable score, not just a verdict.

## Finding a Selection Is a Separate Job

The bound is one-sided. A positive $g$ proves infeasibility, but $g \le 0$ proves nothing at all --
it only means these prices failed to rule the problem out. So a satisfying selection has to be
produced, not deduced.

The prices help here too, in a different role. Once they have matured, a high-scoring item is one
that starved constraints want and over-full constraints do not, which makes the top-`k` scoring
selection a good first guess. It is rarely feasible outright, so it is repaired: repeatedly take
the worst-violated constraint and swap one item for another where doing so lowers the total
violation. Several such attempts are made from slightly perturbed scores.

Any selection this produces with zero violation is a witness, and a witness needs no certificate --
it can be checked directly against every constraint. Randomization is safe in this phase precisely
because nothing here claims a bound.

## Three Answers, Two of Them Proofs

| Verdict | What it means | What backs it |
|---------|---------------|---------------|
| **Feasible** | A selection satisfying every constraint exists. | The selection itself, checkable against the constraints. |
| **Infeasible** | No selection can satisfy every constraint. | Prices with $g > 0$, plus the violation floor they certify. |
| **Unknown** | Neither was established. | Nothing. |

**Unknown is not a weak "probably infeasible".** It carries no information whatsoever, and a
caller must behave exactly as if the check had never run. Two situations land there by
construction: problems that are satisfiable only in a fractional sense, where prices provably
cannot separate them, and satisfiable problems whose witnesses need several coordinated swaps at
once, which the repair step does not attempt.

## Using It

Ask about a problem directly:

```python
report = problem.check_feasibility()
print(report)
```

`thorough=True` searches longer before answering: it tightens the violation floor on an infeasible
problem and settles some cases the default leaves unknown.

The same machinery can start a solve, rather than just describe it:

```python
from max_div.solver import InitializationStrategy

solver = (
    MaxDivSolverBuilder(problem)
    .set_initialization_strategy(InitializationStrategy.most_feasible())
    .with_preset(seconds(5))
    .build()
)
```

The solver then begins from a satisfying selection where one can be constructed, spending its whole
budget on diversity instead of first searching for feasibility. Where the verdict is unknown, it
initializes exactly as it otherwise would.

## Further Reading

- Boyd & Vandenberghe, *Convex Optimization* (Cambridge University Press, 2004), chapter 5 -- the
  duality theory this rests on, including why a relaxed problem's optimum bounds the original's.
- Fisher, "The Lagrangian Relaxation Method for Solving Integer Programming Problems",
  *Management Science* 27(1), 1981 -- the classic survey of the technique applied to integer
  programs, and of the subgradient search used to improve the prices.
- Geoffrion, "Lagrangean Relaxation for Integer Programming", *Mathematical Programming Study* 2,
  1974 -- establishes how strong the bound can get, and hence which problems are destined to come
  back unknown.
