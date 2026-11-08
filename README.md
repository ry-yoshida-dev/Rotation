# Rotation

## Overview

`rotation` is a Python package for 3D rotations as mathematical objects in SO(3). Every representation converts into every other, and they share one set of operations: composition with `@`, inverse, powers, applying to vectors, and the angle between rotations. Each representation also has a batch class that processes N rotations at once with vectorized operations. Coordinate handedness and Euler angles are out of scope.

| Representation | Single | Batch | Stored value of one rotation |
| --- | --- | --- | --- |
| Rotation matrix | `RotationMatrix` | `RotationMatrices` | (3, 3) matrix in SO(3) |
| Rotation vector (axis–angle) | `RotationVector` | `RotationVectors` | (3,) vector, axis scaled by the angle |
| Unit quaternion | `Quaternion` | `Quaternions` | (4,) components `[w, x, y, z]` |

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

Batches broadcast a single rotation and pair two batches element by element:

```python
from rotation import Quaternions, RotationVectors

turns = RotationVectors.from_axes_angles(axes=np.eye(3), angles=[0.1, 0.2, 0.3])
quaternions = Quaternions.from_rotation(turns)
print((quaternions @ matrix).angles_to(turns))  # [1.5707..., 1.5707..., 1.5707...]
print(quaternions.apply(np.eye(3)).shape)  # (3, 3), one vector per rotation
print(quaternions[0])  # Quaternion(...)
```
