import numpy as np
from dataclasses import dataclass
from functools import cached_property
from scipy.spatial.transform import Rotation # type: ignore

from .format import QuaternionFormat


@dataclass(frozen=True)
class Quaternion:
    """
    Container class for quaternion representation of rotation.

    Parameters
    ----------
    value: np.ndarray
        Quaternion values with shape (4,).
    format: QuaternionFormat
        The format of the quaternion (WXYZ or XYZW). Default is WXYZ.
    """
    value: np.ndarray
    format: QuaternionFormat

    def __post_init__(self) -> None:
        """Validate that the quaternion is valid."""
        self._is_valid_quaternion()

    def _is_valid_quaternion(self) -> None:
        """
        Check if the quaternion is valid.
        
        Raises
        ------
        ValueError:
            If the quaternion is not a shape (4,) array.
            If the quaternion is not normalized.
        """
        if self.value.shape != (4,):
            raise ValueError(f"Invalid quaternion: must be a shape (4,) array, got shape {self.value.shape}.")
        
        # Check if quaternion is normalized (approximately)
        norm = np.linalg.norm(self.value)
        if not np.isclose(norm, 1.0, atol=1e-6):
            raise ValueError(f"Invalid quaternion: must be normalized (norm={norm})")

    @property
    def wxyz(self) -> np.ndarray:
        """
        Get quaternion in WXYZ format.
        
        Returns
        -------
        np.ndarray:
            The quaternion in WXYZ format.
        """
        match self.format:
            case QuaternionFormat.WXYZ:
                return self.value
            case QuaternionFormat.XYZW:
                return np.roll(self.value, 1)

    @property
    def xyzw(self) -> np.ndarray:
        """
        Get quaternion in XYZW format (scipy format).
        
        Returns
        -------
        np.ndarray:
            The quaternion in XYZW format.
        """
        match self.format:
            case QuaternionFormat.XYZW:
                return self.value
            case QuaternionFormat.WXYZ:
                return np.roll(self.value, -1)

    @property
    def is_scalar_first(self) -> bool:
        """
        Check if the quaternion is in scalar first format.
        
        Returns
        -------
        bool:
            True if the quaternion is in WXYZ format, False if in XYZW format.
        """
        return self.format.is_scalar_first

    @cached_property
    def rotation_matrix(self) -> np.ndarray:
        """
        Convert quaternion to rotation matrix.

        Returns
        -------
        np.ndarray:
            The 3×3 rotation matrix (float64), row-vector convention
            ``v_new = R @ v``.
        """
        scipy_rotation = Rotation.from_quat(
            self.value,
            scalar_first=self.is_scalar_first,
        )
        return np.asarray(scipy_rotation.as_matrix(), dtype=np.float64)
