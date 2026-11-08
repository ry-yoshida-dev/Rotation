# vector

## Overview

Rotations stored as rotation vectors (axis–angle): the direction is the rotation axis and the length is the angle in radians.

## Components

| Component | Description |
|-----------|-------------|
| [single.py](./single.py) | `RotationVector`, one rotation vector. |
| [batch.py](./batch.py) | `RotationVectors`, N rotation vectors. |

## Examples

```python
import numpy as np

from rotation.vector import RotationVector

vector = RotationVector.from_axis_angle(axis=[1.0, 0.0, 0.0], angle=np.pi / 4)
print(vector.angle)  # 0.785398...
print(vector.axis)  # [1.0, 0.0, 0.0]
```
