from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.spatial.transform import Rotation

from ..types import FloatArray
from ..utils import to_readonly_array
from .format import QuaternionFormat


@dataclass(frozen=True, eq=False)
class Quaternion:
    """
    Container class for unit quaternions.

    Parameters
    ----------
    value : FloatArray
        Quaternion components with shape (4,), ordered by ``format``.
    format : QuaternionFormat
        Order of the components.

    Raises
    ------
    ValueError
        If ``value`` does not have shape (4,) or is not normalized within an absolute
        tolerance of 1e-6.
    """

    value: FloatArray
    format: QuaternionFormat

    def __post_init__(self) -> None:
        """Validate the quaternion and store a read-only copy of it."""
        if self.value.shape != (4,):
            raise ValueError(
                f"Invalid quaternion: must be a shape (4,) array, got shape {self.value.shape}."
            )
        norm = float(np.linalg.norm(self.value))
        if not np.isclose(norm, 1.0, atol=1e-6):
            raise ValueError(
                f"Invalid quaternion: must be normalized, got norm {norm}."
            )
        object.__setattr__(self, "value", to_readonly_array(self.value))

    @property
    def wxyz(self) -> FloatArray:
        """Return the components in scalar-first order."""
        match self.format:
            case QuaternionFormat.WXYZ:
                return self.value
            case QuaternionFormat.XYZW:
                return np.roll(self.value, 1)

    @property
    def xyzw(self) -> FloatArray:
        """Return the components in scalar-last order."""
        match self.format:
            case QuaternionFormat.XYZW:
                return self.value
            case QuaternionFormat.WXYZ:
                return np.roll(self.value, -1)

    @property
    def is_scalar_first(self) -> bool:
        """Return whether ``value`` stores the scalar component first."""
        return self.format.is_scalar_first

    @cached_property
    def rotation_matrix(self) -> FloatArray:
        """Return the 3×3 rotation matrix in the column-vector convention ``v_new = R @ v``."""
        return Rotation.from_quat(
            self.value, scalar_first=self.is_scalar_first
        ).as_matrix()
