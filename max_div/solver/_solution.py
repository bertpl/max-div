from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class MaxDivSolution:
    i_selected: NDArray[np.int32]
