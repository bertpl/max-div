from abc import ABC, abstractmethod
from typing import Any, Type

from max_div.solver._problem import MaxDivProblem


# =================================================================================================
#  BenchmarkProblem base class
# =================================================================================================
class BenchmarkProblem(ABC):
    # -------------------------------------------------------------------------
    #  Registration hook
    # -------------------------------------------------------------------------
    def __init_subclass__(cls, **kwargs):
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
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def description(cls) -> str:
        """Return single-line description of this benchmark problem."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def supported_params(cls) -> dict[str, str]:
        """
        Return a dictionary of supported parameters for this benchmark problem,
        as (param_name, param_description) key-value pairs in a dict.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def get_example_parameters(cls) -> dict[str, Any]:
        """
        Return a dictionary of example acceptable parameter values for this benchmark problem,
        as (param_name, example_value) key-value pairs in a dict.
        """
        raise NotImplementedError()

    # -------------------------------------------------------------------------
    #  Problem creation
    # -------------------------------------------------------------------------
    @classmethod
    def create_problem_instance(cls, **kwargs) -> MaxDivProblem:
        """
        Create and return an instance of MaxDivProblem for this benchmark problem,
        using the provided parameters as needed.
        """

        # --- validate ----------------
        supported_params = cls.supported_params().keys()
        for key in kwargs.keys():
            if key not in supported_params:
                raise ValueError(
                    f"Parameter '{key}' is not supported by benchmark problem '{cls.name()}'."
                    f" Supported parameters: {list(supported_params)}"
                )

        # --- create ------------------
        return cls._create_problem_instance(**kwargs)

    @classmethod
    @abstractmethod
    def _create_problem_instance(cls, **kwargs) -> MaxDivProblem:
        raise NotImplementedError()


# =================================================================================================
#  Registry
# =================================================================================================
class BenchmarkProblemRegistry:
    """Minimal class to register all defined BenchmarkProblem subclasses; used by the factory class."""

    _registry: dict[str, Type[BenchmarkProblem]] = dict()  # name -> class

    @classmethod
    def register(cls, problem_class: Type[BenchmarkProblem]):
        cls._registry[problem_class.name()] = problem_class

    @classmethod
    def get_registered_classes(cls) -> dict[str, Type[BenchmarkProblem]]:
        return cls._registry.copy()
