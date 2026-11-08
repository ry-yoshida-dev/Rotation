from enum import Enum


class QuaternionFormat(Enum):
    """
    Order of the quaternion components.

    Attributes
    ----------
    WXYZ
        Scalar first, ``[w, x, y, z]``.
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
