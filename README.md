# Rotation

## Overview

`rotation` is a Python package for 3D rotations as mathematical objects in SO(3). Every representation converts into every other, and they share one set of operations: composition with `@`, inverse, powers, applying to vectors, and the angle between rotations. Coordinate handedness and Euler angles are out of scope.

| Representation | Class | Stored value |
| --- | --- | --- |
| Rotation matrix | `RotationMatrix` | (3, 3) matrix in SO(3) |
| Rotation vector (axis–angle) | `RotationVector` | (3,) vector, axis scaled by the angle |
| Unit quaternion | `Quaternion` | (4,) components `[w, x, y, z]` |

## Installation

From the project root:

```bash
pip install -e .
```

## Usage

```python
import numpy as np

from rotation import Quaternion, QuaternionFormat, RotationMatrix, RotationVector

quarter_turn = RotationVector.from_axis_angle(axis=[0.0, 0.0, 1.0], angle=np.pi / 2)
matrix = RotationMatrix.from_rotation(quarter_turn)
print(matrix.apply([1.0, 0.0, 0.0]))  # [0.0, 1.0, 0.0]

half_turn = matrix @ quarter_turn  # composes across representations
print(half_turn.angle)  # 3.14159...
print((matrix**0.5).angle_to(matrix))  # 0.78539...

quaternion = Quaternion.from_rotation(matrix)
print(quaternion.components(QuaternionFormat.XYZW))  # [0.0, 0.0, 0.707..., 0.707...]
print(quaternion.inverse().is_close(matrix.inverse()))  # True
```
