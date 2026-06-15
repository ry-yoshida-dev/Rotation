# protocols

## Overview

Structural typing protocols and factory narrowing helpers for rotation-matrix containers and their class constructors. Used by mixins to preserve static types when calling `cls(*, value=...)` or `type(self)(*, value=...)`.

## Components

| Component | Description |
|-----------|-------------|
| [like.py](./like.py) | `RotationMatrixLike` — structural protocol requiring `value: FloatArray` |
| [matrix_type.py](./matrix_type.py) | `RotationMatrixType` — callable class protocol for `(*, value: ndarray) -> _R` |
| [ctor.py](./ctor.py) | `rotation_matrix_ctor` — narrows `type[_R]` / `type(self)` to `RotationMatrixType[_R]` |

See the matrix package [../README.md](../README.md) for context.
