from max_div._core.metrics import DiversityMetric
from max_div._core.problem import VectorMaxDivProblem

from ._registry import BenchmarkProblem, BenchmarkProblemRegistry


class BenchmarkProblemFactory:
    """Factory class for conveniently constructing MaxDivProblem instances for benchmarking purposes.

    This class makes all registered (and discovered) BenchmarkProblem subclasses available (see show_all)
      and allows creating corresponding MaxDivProblem instances by name & problem size (see construct_problem).
    """

    @classmethod
    def construct_problem(cls, name: str, n: int, diversity_metric: DiversityMetric) -> VectorMaxDivProblem:
        """Create and return an instance of MaxDivProblem for the benchmark problem with the given name.

        Args:
            name: Registered benchmark problem name, e.g. `"U1"` or `"C3"`.
            n: Problem size (number of vectors to choose from); all other dimensions are derived
                from it.  Must be >= 20.
            diversity_metric: Diversity metric to be maximized.
        """
        return cls._get_problem_class(name).create_problem_instance(n, diversity_metric)

    @classmethod
    def get_all_benchmark_problems(cls) -> dict[str, type[BenchmarkProblem]]:
        """Return a dict mapping benchmark problem names to their classes."""
        return BenchmarkProblemRegistry.get_registered_classes()

    @classmethod
    def get_all_benchmark_names(cls) -> list[str]:
        """Return a sorted list of all registered benchmark problem names."""
        return sorted(cls.get_all_benchmark_problems().keys())

    @classmethod
    def get_problem_dimensions(cls, name: str, n: int) -> tuple[int, int, int, int, int]:
        """Get problem dimensions as (d, n, k, m, n_con_indices)-tuple for the benchmark problem with the given name."""
        return cls._get_problem_class(name).get_problem_dimensions(n)

    @classmethod
    def show_all(cls) -> None:
        """Show all registered benchmark problems and their descriptions."""
        registered = cls.get_all_benchmark_problems()
        for name in sorted(registered.keys()):
            print(f"{name.ljust(20)}: {registered[name].description()}")

    @classmethod
    def _get_problem_class(cls, name: str) -> type[BenchmarkProblem]:
        """Return the registered problem class for the given name, or raise a ValueError naming the alternatives."""
        registered = BenchmarkProblemRegistry.get_registered_classes()
        problem_cls = registered.get(name)
        if problem_cls is None:
            raise ValueError(
                f"Benchmark problem '{name}' is not registered."
                f" Available benchmark problems: {sorted(registered.keys())}"
            )
        return problem_cls
