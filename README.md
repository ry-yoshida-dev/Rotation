# Rotation

## Overview

`Rotation` is a Python package for working with 3D rotations.  
It provides validated rotation matrices, rotation vectors (axis–angle), and quaternions (WXYZ / XYZW).

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
    RotationMatrix,
    RotationVector,
)

# Identity rotation matrix
R0 = RotationMatrix.unit_matrix()

# Rotation vector (axis * angle, e.g. OpenCV rvec); convert to ndarray
v = RotationVector(value=np.array([0.0, 0.0, np.pi / 4]))
R1 = v.rotation_matrix

# Compose rotations (R1 is a raw 3×3 ndarray)
R = RotationMatrix(value=R0.value @ R1)

# Rotation vector from an explicit axis and angle
w = RotationVector.from_axis_angle(axis=np.array([0.0, 0.0, 1.0]), angle=np.pi / 6)
Rw = w.rotation_matrix

# Quaternion (normalized), WXYZ layout
q = Quaternion(
    value=np.array([1.0, 0.0, 0.0, 0.0]),
    format=QuaternionFormat.WXYZ,
)
R_from_q = q.rotation_matrix
```
