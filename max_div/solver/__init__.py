"""
General usage of dimensions:
   n: number of initial vectors to choose from ('universe')
   d: dimensionality of the vectors
   k: number of vectors to be selected
   m: number of (group) constraints imposed on the problem
"""

from ._constraints import Constraint
from ._distance import DistanceMetric
from ._diversity import DiversityMetric
from ._problem import MaxDivProblem
from ._solution import MaxDivSolution
from ._solver import MaxDivSolver
from ._solver_builder import MaxDivSolverBuilder
