from typing import Type

from max_div.internal.formatting import ljust_str_list
from max_div.solver._problem import MaxDivProblem

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
        else:
            return problem_cls.create_problem_instance(**params)

    @classmethod
    def get_all_benchmark_problems(cls) -> dict[str, Type[BenchmarkProblem]]:
        return BenchmarkProblemRegistry.get_registered_classes()

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
