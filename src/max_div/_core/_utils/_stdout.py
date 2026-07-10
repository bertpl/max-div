import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def stdout_to_file(enabled: bool = True, filename: str | Path | None = None) -> Iterator[None]:
    """Context manager to redirect stdout to file, if enabled."""
    # --- argument validation -----------------------------
    if enabled and not filename:
        raise ValueError("`filename` must be provided when 'enabled' is True.")

    # --- context mgr -------------------------------------
    old_stdout = sys.stdout
    f = None
    if enabled:
        f = Path(filename).open("w")  # noqa: SIM115 — closed in the finally block of this context manager
        sys.stdout = f

    try:
        yield
    finally:
        sys.stdout = old_stdout
        if f is not None:
            f.close()
