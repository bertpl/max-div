
from max_div._core._utils import ljust_str_list
from max_div._core.problem import MaxDivProblem

from ._registry import BenchmarkProblem, BenchmarkProblemRegistry


class BenchmarkProblemFactory:
    """
    Factory class for conveniently constructing MaxDivProblem instances for benchmarking purposes.

    This class makes all registered (and discovered) BenchmarkProblem subclasses available (see show_all)
      and allows creating corresponding MaxDivProblem instances by name & parameter values (see create_problem).
    """

    @classmethod
    def construct_problem(cls, name: str, **params) -> MaxDivProblem:
        """
        Create and return an instance of MaxDivProblem for the benchmark problem with the given name,
        using the provided parameters as needed.
        """

        # find BenchmarkProblem subclass
        registered = BenchmarkProblemRegistry.get_registered_classes()
        problem_cls = registered.get(name)

        # report issue or return problem instance
        if problem_cls is None:
            raise ValueError(
                f"Benchmark problem '{name}' is not registered."
                f" Available benchmark problems: {sorted(registered.keys())}"
            )
        return problem_cls.create_problem_instance(**params)

    @classmethod
    def get_all_benchmark_problems(cls) -> dict[str, type[BenchmarkProblem]]:
        """Return a dict mapping benchmark problem names to their classes."""
        return BenchmarkProblemRegistry.get_registered_classes()

    @classmethod
    def get_all_benchmark_names(cls) -> list[str]:
        """Return a sorted list of all registered benchmark problem names."""
        return sorted(cls.get_all_benchmark_problems().keys())

    @classmethod
    def get_problem_dimensions(cls, name: str, **params) -> tuple[int, int, int, int, int]:
        """
        Get problem dimensions as (d, n, k, m, n_con_indices)-tuple for the benchmark problem with the given name,
        using the provided parameters as needed.
        """
        return cls.get_all_benchmark_problems().get(name).get_problem_dimensions(**params)

    @classmethod
    def show_all(cls):
        """Show all registered benchmark problems and their parameters"""

        # --- get all registered classes ---
        registered = cls.get_all_benchmark_problems()

        # --- display ---
        for name in sorted(registered.keys()):
            problem_cls = registered[name]

            # show name & description
            print(f"{name.ljust(20)}: {problem_cls.description()}")

            # show params & descriptions
            params = problem_cls.supported_params()
            param_names = sorted(params.keys())
            param_names_ljust = ljust_str_list(param_names)
            for param_name, param_name_ljust in zip(param_names, param_names_ljust):
                param_desc = params[param_name]
                print(f"    - {param_name_ljust}: {param_desc}")

            # blank line between problems
            print()
