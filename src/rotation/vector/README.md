# vector

## Overview

Rotation vectors (axis–angle): the direction is the rotation axis and the length is the angle in radians.

## Components

| Component | Description |
|-----------|-------------|
| [container.py](./container.py) | `RotationVector`, the rotation vector container. |
| [protocol.py](./protocol.py) | Structural type that lets the mixins return the concrete subclass. |
| [mixin](./mixin/README.md) | Factories shared by rotation vector containers. |

## Examples

```python
import numpy as np

from rotation.vector import RotationVector

vector = RotationVector.from_axis_angle(axis=np.array([1.0, 0.0, 0.0]), angle=np.pi / 4)
print(vector.angle)  # 0.785398...
print(RotationVector.from_matrix(vector.rotation_matrix).value)
```
