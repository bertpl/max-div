from ._cmd_benchmark import benchmark


# =================================================================================================
#  benchmark solver
# =================================================================================================
@benchmark.group(name="solver")
def solver() -> None:
    """Benchmarking of individual solver strategies & solver presets, based on built-in benchmark problems."""
