# vector

## Overview

Rotation vectors (axis–angle): direction of the vector is the rotation axis, length is the angle in radians. `RotationVector` is a small frozen container; classmethods on the mixin build instances from matrices or axis–angle pairs, and `rotation_matrix` returns a 3×3 `numpy.ndarray` (via OpenCV `Rodrigues`).

## Components

| Component | Description |
|-----------|-------------|
| [container.py](./container.py) | `RotationVector` — `value`, `angle`, `rotation_matrix` |
| [protocol.py](./protocol.py) | `RotationVectorLike`, `RotationVectorType`, `rotation_vector_ctor` |
| [mixin/](./mixin/README.md) | `RotationVectorFactoryMixin` — `from_matrix`, `from_axis_angle`, `zero_vector` |

See the parent package [../README.md](../README.md) for context.

## Example

```python
import numpy as np

from rotation.vector import RotationVector

# From axis and angle (axis is normalized for you)
v = RotationVector.from_axis_angle(
    axis=np.array([0.0, 0.0, 1.0]),
    angle=np.pi / 3,
)

# From a validated rotation matrix (SO(3))
R = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
v_from_R = RotationVector.from_matrix(R)

# Magnitude = angle (rad); matrix as ndarray
theta = v.angle
R_opencv = v.rotation_matrix
```
