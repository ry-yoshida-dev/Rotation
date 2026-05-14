# quaternion

## Overview

Quaternion representation of rotations.  
Uses `scipy.spatial.transform.Rotation` to convert to a 3×3 rotation matrix (`numpy.ndarray`). `QuaternionFormat` selects scalar component order (WXYZ vs XYZW).

## Components

| Component | Description |
|-----------|-------------|
| [container.py](./container.py) | `Quaternion` — normalized 4-vector, format accessors, `rotation_matrix` |
| [format.py](./format.py) | `QuaternionFormat` — `WXYZ` / `XYZW` enum |

See the parent package [../README.md](../README.md) for context.

## Example

```python
import numpy as np

from rotation.quaternion import Quaternion, QuaternionFormat

# Scalar-first unit quaternion (no rotation)
q_wxyz = Quaternion(
    value=np.array([1.0, 0.0, 0.0, 0.0]),
    format=QuaternionFormat.WXYZ,
)

# Same rotation, stored as XYZW (scipy-style)
q_xyzw = Quaternion(
    value=np.array([0.0, 0.0, 0.0, 1.0]),
    format=QuaternionFormat.XYZW,
)

# Canonical views and 3×3 rotation matrix (ndarray)
wxyz = q_wxyz.wxyz
xyzw = q_xyzw.xyzw
R = q_wxyz.rotation_matrix
```

`value` must be normalized; invalid norms raise `ValueError` in `__post_init__`.
