"""Type aliases for NumPy arrays used across the package."""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.floating]
BoolArray: TypeAlias = NDArray[np.bool_]
IndexArray: TypeAlias = NDArray[np.integer]
