# Rotation

## Overview

`rotation` is a Python package for 3D rotations. It provides immutable, validated rotation matrices, rotation vectors (axis–angle), and quaternions in WXYZ or XYZW layout, with conversions between them. See [src/rotation/README.md](src/rotation/README.md) for package details.

## Installation

From the project root:

```bash
pip install -e .
```

## Usage

```python
import numpy as np

from rotation import Quaternion, QuaternionFormat, RotationMatrix, RotationVector

vector = RotationVector.from_axis_angle(axis=np.array([0.0, 0.0, 1.0]), angle=np.pi / 2)
matrix = RotationMatrix(value=vector.rotation_matrix)
print(matrix.x_axis)  # [0.0, 1.0, 0.0]

composed = matrix @ matrix
print(composed.rotation_vector)  # [0.0, 0.0, 3.14159265...]

quaternion = Quaternion(value=np.array([1.0, 0.0, 0.0, 0.0]), format=QuaternionFormat.WXYZ)
print(quaternion.xyzw)  # [0.0, 0.0, 0.0, 1.0]
```
