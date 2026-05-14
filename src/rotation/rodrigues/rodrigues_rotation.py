import numpy as np
from dataclasses import dataclass
from functools import cached_property

from .skew_symmetric_matrix import SkewSymmetricMatrix


@dataclass(frozen=True)
class RodriguesRotationParameter:
    """
    Container class for Rodrigues rotation parameter.

    Attributes
    ----------
    value: np.ndarray
        The Rodrigues rotation parameter.
    """
    value: np.ndarray

    def __post_init__(self) -> None:
        """
        Validate that the Rodrigues rotation parameter is valid.
        
        Raises
        ------
        ValueError:
            If the Rodrigues rotation vector is not a shape (3,) array.
        """
        if self.value.shape != (3,):
            raise ValueError(
                f"Rodrigues rotation vector must be a shape (3,) array, got shape {self.value.shape}."
            )

    @property
    def x(self) -> float:
        """
        Return the x component of the Rodrigues rotation parameter.

        Returns
        -------
        float:
            The x component of the Rodrigues rotation parameter.
        """
        return self.value[0]

    @property
    def y(self) -> float:
        """
        Return the y component of the Rodrigues rotation parameter.

        Returns
        -------
        float:
            The y component of the Rodrigues rotation parameter.
        """
        return self.value[1]

    @property
    def z(self) -> float:
        """
        Return the z component of the Rodrigues rotation parameter.

        Returns
        -------
        float:
            The z component of the Rodrigues rotation parameter.
        """
        return self.value[2]

    def transform(
        self, 
        vector: np.ndarray,
        ) -> np.ndarray:
        """
        Apply Rodrigues rotation formula to transform a vector.
        
        Parameters
        ----------
        vector: np.ndarray
            Vector to be rotated with shape (3,).
            
        Returns
        -------
        np.ndarray
            Transformed vector with shape (3,).

        Raises
        ------
        ValueError:
            If the input vector is not a shape (3,) array.
        """
        if vector.shape != (3,):
            raise ValueError(
                f"Input vector must be a shape (3,) array, got shape {vector.shape}."
            )

        return self.rotation_matrix @ vector

    @cached_property
    def rotation_matrix(self) -> np.ndarray:
        """
        Convert Rodrigues rotation parameter to rotation matrix.

        Returns
        -------
        np.ndarray:
            The 3×3 rotation matrix (float64), row-vector convention
            ``v_new = R @ v``.
        """
        theta = np.linalg.norm(self.value)

        if theta < 1e-6:
            return np.eye(3, dtype=np.float64)

        normalized_rodrigues = RodriguesRotationParameter(value=self.value / theta)

        K = SkewSymmetricMatrix.from_k_parameter(
            k_x=normalized_rodrigues.x,
            k_y=normalized_rodrigues.y,
            k_z=normalized_rodrigues.z,
        )

        I = np.eye(3, dtype=np.float64)
        skew_term = np.sin(theta) * K.value
        symmetric_term = (1 - np.cos(theta)) * K.squared
        R = I + skew_term + symmetric_term

        return np.asarray(R, dtype=np.float64)
