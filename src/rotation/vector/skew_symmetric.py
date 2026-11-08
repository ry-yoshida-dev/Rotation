import numpy as np

from ..types import FloatArray


class SkewSymmetricMatrix:
    """
    Conversions between 3-vectors and 3×3 skew-symmetric matrices.

    The skew-symmetric matrix ``[ω]×`` of ``ω`` satisfies ``[ω]× @ v = ω × v``. It is
    the Lie algebra element so(3) of the rotation vector ``ω``: the hat map builds it,
    and the vee map recovers ``ω``. Every method takes a stack with shape (..., 3) or
    (..., 3, 3) and works on all of it at once.
    """

    @staticmethod
    def from_vectors(vectors: FloatArray) -> FloatArray:
        """
        Return the skew-symmetric matrix of each vector, the hat map.

        Parameters
        ----------
        vectors : FloatArray
            Vectors with shape (..., 3).

        Returns
        -------
        FloatArray
            Matrices with shape (..., 3, 3).
        """
        x = vectors[..., 0]
        y = vectors[..., 1]
        z = vectors[..., 2]
        zero = np.zeros_like(x)
        rows = [
            np.stack([zero, -z, y], axis=-1),
            np.stack([z, zero, -x], axis=-1),
            np.stack([-y, x, zero], axis=-1),
        ]
        return np.stack(rows, axis=-2)

    @staticmethod
    def to_vectors(matrices: FloatArray, *, atol: float = 1e-6) -> FloatArray:
        """
        Return the vector of each skew-symmetric matrix, the vee map.

        Parameters
        ----------
        matrices : FloatArray
            Skew-symmetric matrices with shape (..., 3, 3).
        atol : float, default 1e-6
            Absolute tolerance of the skew-symmetry check.

        Returns
        -------
        FloatArray
            Vectors with shape (..., 3).

        Raises
        ------
        ValueError
            If a matrix is not skew-symmetric.
        """
        if not np.allclose(matrices, -np.swapaxes(matrices, -1, -2), atol=atol):
            raise ValueError("Invalid matrix: must be skew-symmetric.")
        return np.stack(
            [matrices[..., 2, 1], matrices[..., 0, 2], matrices[..., 1, 0]], axis=-1
        )
