from abc import ABC, abstractmethod
from typing import Any, ClassVar

from max_div._core.metrics import DiversityMetric
from max_div._core.problem import VectorMaxDivProblem

# Below this, k = ceil(n/10) < 2 and a diversity problem is degenerate.
MIN_N = 20


# =================================================================================================
#  BenchmarkProblem base class
# =================================================================================================
class BenchmarkProblem(ABC):
    # -------------------------------------------------------------------------
    #  Registration hook
    # -------------------------------------------------------------------------
    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401 -- forwarded to type.__init_subclass__
        """This method ensures each child class is registered in the BenchmarkProblemRegistry upon import."""
        super().__init_subclass__(**kwargs)
        BenchmarkProblemRegistry.register(cls)

    # -------------------------------------------------------------------------
    #  Meta-data
    # -------------------------------------------------------------------------
    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Return name of this benchmark problem."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def description(cls) -> str:
        """Return single-line description of this benchmark problem."""
        raise NotImplementedError

    @classmethod
    def get_problem_dimensions(cls, n: int) -> tuple[int, int, int, int, int]:
        """Return problem dimensions as (d, n, k, m, n_con_indices)-tuple for the given problem size n.

        Dimensions can be indicative (especially n_con_indices) if they are stochastic.  Main goal of
        this method is to get an idea of dimensions without needing to create the full problem instance.
        """
        cls.validate_n(n)
        return cls._get_problem_dimensions(n)

    @classmethod
    @abstractmethod
    def _get_problem_dimensions(cls, n: int) -> tuple[int, int, int, int, int]:
        raise NotImplementedError

    # -------------------------------------------------------------------------
    #  Problem creation
    # -------------------------------------------------------------------------
    @classmethod
    def create_problem_instance(cls, n: int, diversity_metric: DiversityMetric) -> VectorMaxDivProblem:
        """Create and return a MaxDivProblem instance of this benchmark problem with the given size n.

        Args:
            n: Problem size (number of vectors to choose from); all other dimensions are derived
                from it.  Must be >= 20 — below that the selection size k drops under 2 and the
                problem is degenerate.
            diversity_metric: Diversity metric to be maximized.
        """
        cls.validate_n(n)
        return cls._create_problem_instance(n, diversity_metric)

    @classmethod
    def validate_n(cls, n: int) -> None:
        """Raise ValueError if n is below the smallest non-degenerate problem size."""
        if n < MIN_N:
            raise ValueError(
                f"Benchmark problem '{cls.name()}' requires n >= {MIN_N}, got n={n}."
                f" Below that, the selection size k drops under 2 and the problem is degenerate."
            )

    @classmethod
    @abstractmethod
    def _create_problem_instance(cls, n: int, diversity_metric: DiversityMetric) -> VectorMaxDivProblem:
        raise NotImplementedError


# =================================================================================================
#  Registry
# =================================================================================================
class BenchmarkProblemRegistry:
    """Minimal class to register all defined BenchmarkProblem subclasses; used by the factory class."""

    _registry: ClassVar[dict[str, type[BenchmarkProblem]]] = {}  # name -> class

    @classmethod
    def register(cls, problem_class: type[BenchmarkProblem]) -> None:
        cls._registry[problem_class.name()] = problem_class

    @classmethod
    def get_registered_classes(cls) -> dict[str, type[BenchmarkProblem]]:
        return cls._registry.copy()
