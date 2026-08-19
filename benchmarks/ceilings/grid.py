"""The candidate-size grid and the campaign's budget and memory constants."""

# These constants are the one source every stage shares. The docs' Capability Definitions page
# states the same values as prose; scripts/capability_data.py validates published ceiling
# cells against the same grid shape.
REFERENCE_BUDGET_SEC = 60.0  # this budget bounds every time- and quality-stage run
BEST_KNOWN_BUDGET_SEC = 900.0  # this budget bounds every best-known-solution run
MEMORY_CAP_BYTES = 32 * 2**30  # caps every run's peak memory, and defines the memory ceiling
GRID_FLOOR = 100

# The grace is added to a run's hard-kill deadline on top of its budget: the child's untimed setup
# (interpreter start, imports, problem construction) happens inside the same process, and
# killing during setup would misreport a slow import as a tool failure.
SETUP_GRACE_SEC = 90.0


def size_grid(bound: int) -> list[int]:
    """Return the candidate problem sizes up to and including ``bound``.

    The grid is logarithmic with three values per decade (100, 200, 500, 1000, ...);
    every published ceiling is one of these values.
    """
    if bound < GRID_FLOOR:
        raise ValueError(f"bound {bound} lies below the grid floor {GRID_FLOOR}")
    sizes, decade = [], GRID_FLOOR
    while decade <= bound:
        sizes += [m * decade for m in (1, 2, 5) if m * decade <= bound]
        decade *= 10
    return sizes


def operational_bound(d: int = 2) -> int:
    """Return the largest n whose raw float32 vectors array itself fits the memory cap.

    Beyond this size the problem's input cannot exist under the campaign's own memory
    cap, so no stage ever generates an instance past it.
    """
    return MEMORY_CAP_BYTES // (d * 4)
