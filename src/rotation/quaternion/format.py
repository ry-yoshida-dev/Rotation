from enum import Enum

import numpy as np

from ..types import FloatArray


class QuaternionFormat(Enum):
    """
    Order of the quaternion components in an array.

    Attributes
    ----------
    WXYZ
        Scalar first, ``[w, x, y, z]``, the order quaternions store internally.
    XYZW
        Scalar last, ``[x, y, z, w]``, the order SciPy uses by default.
    """

    WXYZ = "wxyz"
    XYZW = "xyzw"

    @property
    def as_upper(self) -> str:
        """Return the format name in uppercase."""
        return self.value.upper()

    @property
    def as_lower(self) -> str:
        """Return the format name in lowercase."""
        return self.value.lower()

    @property
    def is_scalar_first(self) -> bool:
        """Return whether the scalar component comes first."""
        return self == QuaternionFormat.WXYZ

    def to_wxyz(self, components: FloatArray) -> FloatArray:
        """
        Reorder components from this format into scalar-first order.

        Parameters
        ----------
        components : FloatArray
            Quaternions with shape (..., 4) in this format.
        """
        match self:
            case QuaternionFormat.WXYZ:
                return components
            case QuaternionFormat.XYZW:
                return np.roll(components, 1, axis=-1)

    def from_wxyz(self, components: FloatArray) -> FloatArray:
        """
        Reorder components from scalar-first order into this format.

        Parameters
        ----------
        components : FloatArray
            Quaternions with shape (..., 4) in scalar-first order.
        """
        match self:
            case QuaternionFormat.WXYZ:
                return components
            case QuaternionFormat.XYZW:
                return np.roll(components, -1, axis=-1)
