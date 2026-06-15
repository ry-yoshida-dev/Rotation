from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from ..types import FloatArray

@dataclass(frozen=True)
class SkewSymmetricMatrix:
    """
    Container class for skew symmetric matrix.
    
    Attributes
    ----------
    value: FloatArray
        The skew symmetric matrix with shape (3, 3).
    """
    value: FloatArray

    def __post_init__(self) -> None:
        """
        Validate that the skew symmetric matrix is valid.
        
        Raises
        ------
        ValueError:
            If the skew symmetric matrix is not a shape (3, 3) array.
        """
        if self.value.shape != (3, 3):
            raise ValueError(
                f"Skew symmetric matrix must be a shape (3, 3) array, got shape {self.value.shape}."
            )

    @property
    def squared(self) -> FloatArray:
        """
        Return the squared of the skew symmetric matrix.

        Returns
        -------
        FloatArray:
            The squared of the skew symmetric matrix with shape (3, 3).
        """
        return self.value @ self.value

    @classmethod
    def from_k_parameter(
        cls, 
        k_x: float, 
        k_y: float, 
        k_z: float
        ) -> SkewSymmetricMatrix:
        """
        Create a skew symmetric matrix from a k parameter.

        Parameters
        ----------
        k_x: float
            The x component of the k parameter.
        k_y: float
            The y component of the k parameter.
        k_z: float
            The z component of the k parameter.

        Returns
        -------
        SkewSymmetricMatrix:
            The skew symmetric matrix.

        """
        return cls(value=np.array([[0, -k_z, k_y],
                                   [k_z, 0, -k_x],
                                   [-k_y, k_x, 0]]))
                                   