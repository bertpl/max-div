from ._cmd_benchmark import benchmark


# =================================================================================================
#  benchmark solver
# =================================================================================================
@benchmark.group(name="solver")
def solver():
    """
    Benchmarking functionality for individual solver strategies & solver presets, based on built-in benchmark problems.
    """
    pass
