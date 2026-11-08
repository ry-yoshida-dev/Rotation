# vector

## Overview

Rotations stored as rotation vectors (axis–angle): the direction is the rotation axis and the length is the angle in radians.

## Components

| Component | Description |
|-----------|-------------|
| [single.py](./single.py) | `RotationVector`, one rotation vector. |
| [batch.py](./batch.py) | `RotationVectors`, N rotation vectors. |
| [skew_symmetric.py](./skew_symmetric.py) | Hat and vee maps between vectors and skew-symmetric matrices. |

## Examples

```python
import numpy as np

from rotation.vector import RotationVector

vector = RotationVector.from_axis_angle(axis=[1.0, 0.0, 0.0], angle=np.pi / 4)
print(vector.angle)  # 0.785398...
print(vector.axis)  # [1.0, 0.0, 0.0]
print(vector.skew_matrix @ [0.0, 1.0, 0.0])  # ω × (0, 1, 0) = [0.0, 0.0, 0.785...]
```
