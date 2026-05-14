# Rotation

## Overview

`Rotation` is a Python package for working with 3D rotations.  
It provides validated rotation matrices, rotation vectors (axis–angle), quaternions (WXYZ / XYZW), Rodrigues rotation parameters, and skew-symmetric matrices.

Module layout, component index, and a package-level example: [src/rotation/README.md](src/rotation/README.md).

## Installation

Install dependencies only:

```bash
pip install -r requirements.txt
```

From the repository root, add `src` to your Python path so the package can be imported:

```bash
export PYTHONPATH=src
```

## Example

```python
import numpy as np

from rotation import (
    Quaternion,
    QuaternionFormat,
    RodriguesRotationParameter,
    RotationMatrix,
    RotationVector,
    SkewSymmetricMatrix,
)

# Identity rotation matrix
R0 = RotationMatrix.unit_matrix()

# Rodrigues vector (axis * angle); convert to ndarray
r = RodriguesRotationParameter(value=np.array([0.0, 0.0, np.pi / 4]))
R1 = r.rotation_matrix

# Compose rotations (R1 is a raw 3×3 ndarray)
R = RotationMatrix(value=R0.value @ R1)

# Rotation vector (same axis–angle idea as Rodrigues parameter here)
v = RotationVector.from_axis_angle(axis=np.array([0.0, 0.0, 1.0]), angle=np.pi / 6)
Rv = v.rotation_matrix

# Quaternion (normalized), WXYZ layout
q = Quaternion(
    value=np.array([1.0, 0.0, 0.0, 0.0]),
    format=QuaternionFormat.WXYZ,
)
R_from_q = q.rotation_matrix

# Skew-symmetric matrix from components
K = SkewSymmetricMatrix.from_k_parameter(k_x=0.0, k_y=0.0, k_z=1.0)
```
