import sys
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def stdout_to_file(enabled: bool = True, filename: str | Path | None = None):
    """Context manager to redirect stdout to file, if enabled."""

    # --- argument validation -----------------------------
    if enabled and not filename:
        raise ValueError("`filename` must be provided when 'enabled' is True.")

    # --- context mgr -------------------------------------
    old_stdout = sys.stdout
    f = None
    if enabled:
        f = open(filename, "w")
        sys.stdout = f

    try:
        yield
    finally:
        sys.stdout = old_stdout
        if f is not None:
            f.close()
