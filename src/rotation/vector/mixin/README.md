# mixin

## Overview

`RotationVectorFactoryMixin` supplies classmethods (`from_matrix`, `from_axis_angle`, `zero_vector`) used by `RotationVector`. Factories call `rotation_vector_ctor(cls)` so subclasses keep correct static typing.

## Components

| Component | Description |
|-----------|-------------|
| [factory.py](./factory.py) | Matrix → vector conversion with optional SO(3) validation, axis–angle constructor, identity as zero vector |

See the vector package [../README.md](../README.md) for context.

## Example

```python
import numpy as np

from rotation.vector import RotationVector

R = np.eye(3, dtype=np.float64)
v0 = RotationVector.from_matrix(R)

v1 = RotationVector.from_axis_angle(
    axis=np.array([1.0, 0.0, 0.0]),
    angle=np.pi / 4,
)

v2 = RotationVector.zero_vector()
```
