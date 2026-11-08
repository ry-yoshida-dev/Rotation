"""Tests for subclasses that add fields to the rotation containers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rotation import Quaternion, QuaternionFormat, RotationMatrix, RotationVector


@dataclass(frozen=True, eq=False)
class TaggedMatrix(RotationMatrix):
    tag: str


@dataclass(frozen=True, eq=False)
class TaggedVector(RotationVector):
    tag: str


@dataclass(frozen=True, eq=False)
class TaggedQuaternion(Quaternion):
    tag: str


def test_matrix_operations_keep_added_field() -> None:
    """
    Verify derived matrices keep the subclass and its extra field.

    Output: TaggedMatrix with the same tag from inverse, power, and composition.
    """
    matrix = TaggedMatrix(value=np.eye(3), tag="frame")
    for result in (matrix.inverse, matrix**0.5, matrix @ RotationVector.identity()):
        assert isinstance(result, TaggedMatrix)
        assert result.tag == "frame"


def test_vector_operations_keep_added_field() -> None:
    """
    Verify derived rotation vectors keep the subclass and its extra field.

    Output: TaggedVector with the same tag.
    """
    vector = TaggedVector(value=np.array([0.0, 0.0, 1.0]), tag="frame")
    result = vector.inverse
    assert isinstance(result, TaggedVector)
    assert result.tag == "frame"


def test_quaternion_operations_keep_format_and_added_field() -> None:
    """
    Verify derived quaternions keep the format and the extra field.

    Output: TaggedQuaternion with the same tag and format, except with_format.
    """
    quaternion = TaggedQuaternion(
        value=np.array([0.0, 0.0, 0.0, -1.0]), format=QuaternionFormat.XYZW, tag="frame"
    )
    for result in (quaternion.inverse, quaternion.canonical(), quaternion @ quaternion):
        assert isinstance(result, TaggedQuaternion)
        assert result.tag == "frame"
        assert result.format is QuaternionFormat.XYZW
    reordered = quaternion.with_format(QuaternionFormat.WXYZ)
    assert isinstance(reordered, TaggedQuaternion)
    assert reordered.tag == "frame"
    np.testing.assert_allclose(reordered.value, [-1.0, 0.0, 0.0, 0.0])
