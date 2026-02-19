import math

import numpy as np

# --- 64-bit float - as Python floats ---
EPS = float(np.finfo(np.float64).eps)  # Machine epsilon for 64-bit float (float64)
HALF_EPS = float(math.sqrt(EPS))
ALMOST_ONE = float(1.0 - 2 * EPS)

# --- 32-bit float - as np.float32s ---
EPS_F32 = np.float32(np.finfo(np.float32).eps)  # Machine epsilon for 32-bit float (float32)
HALF_EPS_F32 = np.float32(math.sqrt(EPS_F32))
ALMOST_ONE_F32 = np.float32(1.0 - 2 * EPS_F32)
