"""The candidate-size grid and the campaign's shared budget and memory constants.

These are the one source every stage shares. The published Capability Definitions page states
the same values as prose, and scripts/capability_data.py validates published size cells against
the same grid shape.
"""

REFERENCE_BUDGET_SEC = 60.0  # T_max: the budget every time/quality run is judged against

# Budget for the discarded warm-up run each sweep gives a configuration without recorded runs:
# enough for any configuration to produce a tiny-size answer, small enough that budget-honoring
# configurations return quickly.
WARMUP_BUDGET_SEC = 5.0
DEFAULT_SEED = 42  # the protocol's fixed seed; only the quality runs enumerate other seeds
MEMORY_CAP_BYTES = 32 * 2**30  # M_max: the machine-level cap on every run, and the largest-n-within-memory bar
GRID_MIN = 20  # the smallest size the benchmark problems build at

# A self-limiting solver (max-div under an end-to-end budget) is handed this much less than
# REFERENCE_BUDGET_SEC. Its real end-to-end time overshoots the budget it is given — by up to one
# optimization batch — so aiming under T_max keeps the measured time within the budget it is judged
# against. One-shot tools take no budget and ignore this.
SELF_LIMIT_MARGIN_SEC = 1.0

# Added to a run's hard-kill deadline on top of its budget. The child's untimed setup (interpreter
# start, imports, problem construction) runs in the same process, and killing during setup would
# misreport a slow import as a solver failure. This kill only bounds a stuck run: a run that
# finishes over T_max is failed on its measured time, not by this kill.
SETUP_GRACE_SEC = 90.0


def size_grid(bound: int) -> list[int]:
    """Return the candidate problem sizes from `GRID_MIN` up to and including `bound`.

    The grid is the 1-2-5 sequence (20, 50, 100, 200, 500, 1000, ...) — three sizes per decade,
    none smaller than `GRID_MIN`. Every published value lies on it.

    Raises:
        ValueError: If `bound` is below `GRID_MIN`.
    """
    if bound < GRID_MIN:
        raise ValueError(f"bound {bound} lies below the smallest grid size {GRID_MIN}")
    sizes: list[int] = []
    decade = 10
    while decade <= bound:
        sizes += [m * decade for m in (1, 2, 5) if GRID_MIN <= m * decade <= bound]
        decade *= 10
    return sizes


def operational_bound(d: int = 2) -> int:
    """Return the largest n whose raw float32 vectors array alone fits the memory cap.

    Beyond that n the problem's own input cannot exist under the cap, so no stage generates a
    larger instance.
    """
    return MEMORY_CAP_BYTES // (d * 4)
