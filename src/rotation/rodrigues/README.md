# rodrigues

## Overview

Rodrigues rotation parameters (3D vectors encoding axis direction and rotation magnitude) and the associated skew-symmetric matrices.  
`RodriguesRotationParameter` converts to a 3×3 rotation matrix (`numpy.ndarray`) via Rodrigues’ formula; points can be rotated with `transform`.

## Components

| Component | Description |
|-----------|-------------|
| [rodrigues_rotation.py](./rodrigues_rotation.py) | `RodriguesRotationParameter` — validation, `rotation_matrix`, `transform` for vectors |
| [skew_symmetric_matrix.py](./skew_symmetric_matrix.py) | `SkewSymmetricMatrix` — construction and squared term used in Rodrigues’ formula |

See the parent package [../README.md](../README.md) for context.

## Example

```python
import numpy as np

from rotation.rodrigues import RodriguesRotationParameter, SkewSymmetricMatrix

# 90° about +Z: rotation vector = (0, 0, π/2)
r = RodriguesRotationParameter(value=np.array([0.0, 0.0, np.pi / 2], dtype=np.float64))

R = r.rotation_matrix  # (3, 3) ndarray
p = np.array([1.0, 0.0, 0.0])
p_rot = r.transform(p)

# Skew matrix for a unit axis (used internally in Rodrigues’ formula)
K = SkewSymmetricMatrix.from_k_parameter(k_x=0.0, k_y=0.0, k_z=1.0)
S = K.squared
```
