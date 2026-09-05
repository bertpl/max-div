# Third-Party Solver Profiles

One profile per freely available tool that selects diverse subsets, in a uniform shape: what it is
and how it approaches the problem, the problem it actually targets stated in a shared notation with
its guarantee type, and a feature table generated from the same data that drives the capability
table in the README.

Read these alongside the [comparison](../comparison.md), which puts every tool's capabilities on one
grid; a profile is where a mark on that grid turns into an explanation.

**max-div itself is not profiled here.** It appears as a row in the
[comparison table](../comparison.md), where it belongs — a section surveying third-party tools
should not also carry a profile of the package they are being compared against. What max-div does
is the subject of the rest of this documentation.

## How to read these

The tools fall into three groups that answer different questions, and comparing across the groups
is usually a category error, not a fair fight.

**Exact solvers** prove optimality. They do not select diverse subsets; they solve models you
write, and the diversity problem has to become a model first. Read them as the reference for what
the best achievable answer *is* on small instances — not as competitors.

**One-shot pickers** run a single construction pass and return an answer in milliseconds. None of
them improves when given more time, and only code-FDM supports per-group selection constraints, over
disjoint groups. Where their
objective is the one you want and their scale suits you, they are the right tool and max-div is
overkill.

**Clustering and sampling tools** return a diverse-looking subset by a different contract: a
clustering picks cluster centers, which sit where the data is dense, and a sampler draws from a
distribution that favors spread-out subsets without maximizing anything. An empty objective row
means that different contract, not a weakness. They are measured as references under the dispersion
metrics.

## Guarantee types

Each profile states one, defined in the [glossary](../../../concepts/glossary.md):

| Type | Means |
|---|---|
| proven optimum | terminates with a certificate that no better selection exists |
| *k*-approximation | provably within a factor *k* of optimal |
| heuristic | no bound; judged empirically |
| sampler | draws from a distribution; no claim about any individual draw |

## Profiles

<div class="grid cards" markdown>

- **Exact solvers** — [OR-Tools CP-SAT](ortools-cpsat.md) · [SCIP](scip.md) · [HiGHS](highs.md)
- **One-shot pickers** — [RDKit MaxMinPicker](rdkit.md) · [fpsample](fpsample.md) · [skmatter](skmatter.md) · [apricot-select](apricot-select.md) · [qc-selector](qc-selector.md)
- **Fair selection** — [code-FDM](code-fdm.md)
- **Clustering & sampling** — [kmedoids](kmedoids.md) · [DPPy](dppy.md)

</div>

## A note on the facts

Every profile carries the date its facts were last checked against upstream, and every capability
mark that is not self-evident carries a footnote saying what it rests on. Versions, licenses and
activity were verified against PyPI and the projects' repositories on the dates shown.

Capability claims age differently from version numbers: a tool can gain a feature without anything
here noticing. Treat the verification date as what it is — when a person last looked.
