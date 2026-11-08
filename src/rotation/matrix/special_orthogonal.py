from typing import cast

import numpy as np

from ..types import FloatArray


class SpecialOrthogonalMatrix:
    """
    Checks and projections of 3×3 matrices onto SO(3).

    Every method takes one matrix with shape (3, 3) or a stack with shape (..., 3, 3)
    and works on all of them at once.
    """

    @staticmethod
    def validate(matrices: FloatArray, *, atol: float = 1e-6) -> None:
        """
        Check that every matrix is orthogonal and has determinant +1.

        Parameters
        ----------
        matrices : FloatArray
            Matrices with shape (..., 3, 3).
        atol : float, default 1e-6
            Absolute tolerance of the orthogonality and determinant checks.

        Raises
        ------
        ValueError
            If a matrix is not orthogonal or does not have determinant +1.
        """
        identity = np.eye(3)
        gram = np.swapaxes(matrices, -1, -2) @ matrices
        if not np.allclose(gram, identity, atol=atol):
            raise ValueError("Invalid rotation matrix: must be orthogonal.")
        determinants = SpecialOrthogonalMatrix.determinants(matrices)
        if not np.allclose(determinants, 1.0, atol=atol):
            raise ValueError(
                f"Invalid rotation matrix: must have determinant +1, got {determinants}."
            )

    @staticmethod
    def project_by_svd(matrices: FloatArray) -> FloatArray:
        """
        Return the rotation matrix closest to each matrix in the Frobenius norm.

        Parameters
        ----------
        matrices : FloatArray
            Approximate rotation matrices with shape (..., 3, 3).
        """
        decomposition = np.linalg.svd(matrices)
        left: FloatArray = decomposition.U
        right: FloatArray = decomposition.Vh
        # Flip the last singular direction where needed to turn a reflection into a rotation.
        signs = np.sign(SpecialOrthogonalMatrix.determinants(left @ right))
        left[..., :, -1] *= signs[..., np.newaxis]
        return left @ right

    @staticmethod
    def project_by_qr(matrices: FloatArray) -> FloatArray:
        """
        Return the Q factor of the QR decomposition of each matrix, as a rotation.

        The result keeps the direction of the first column. It is cheaper than the
        SVD projection but not the closest rotation in general.

        Parameters
        ----------
        matrices : FloatArray
            Approximate rotation matrices with shape (..., 3, 3).
        """
        decomposition = np.linalg.qr(matrices)
        orthogonal: FloatArray = decomposition.Q
        upper: FloatArray = decomposition.R
        # Make the diagonal of R positive so that Q's columns follow the input's columns.
        diagonal_signs = np.sign(np.diagonal(upper, axis1=-2, axis2=-1))
        diagonal_signs[diagonal_signs == 0] = 1.0
        orthogonal = orthogonal * diagonal_signs[..., np.newaxis, :]
        signs = np.sign(SpecialOrthogonalMatrix.determinants(orthogonal))
        orthogonal[..., :, -1] *= signs[..., np.newaxis]
        return orthogonal

    @staticmethod
    def determinants(matrices: FloatArray) -> FloatArray:
        """
        Return the determinant of each matrix.

        Parameters
        ----------
        matrices : FloatArray
            Matrices with shape (..., 3, 3).

        Returns
        -------
        FloatArray
            Determinants with shape (...).
        """
        # NumPy types the result as Any unless the array shape is known statically.
        return cast(FloatArray, np.linalg.det(matrices))
