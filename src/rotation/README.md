# rotation

## Overview

Immutable rotations in SO(3). Each representation stores a validated, read-only float64 `value` and shares the operations of its abstract class. Equality is identity-based, so compare rotations with `is_close`.

## Components

| Component | Description |
|-----------|-------------|
| [representation.py](./representation.py) | `RotationRepresentation`, the operations every representation shares: array forms of each representation, inverse, powers, and applying to vectors. |
| [single.py](./single.py) | `SingleRotation`, one rotation: conversion, identity, random sampling, alignment of vectors, composition, angles, and comparison. |
| [batch.py](./batch.py) | `BatchRotation`, N rotations: the same operations vectorized, plus the mean, indexing, and iteration. |
| [matrix](./matrix/README.md) | Rotation matrices and projections of approximate matrices onto SO(3). |
| [vector](./vector/README.md) | Rotation vectors (axis–angle) and their skew-symmetric matrices. |
| [quaternion](./quaternion/README.md) | Unit quaternions and their component order. |
| [types.py](./types.py) | Shared NumPy array type alias for floating-point values. |
| [utils.py](./utils.py) | Read-only copies stored by every container. |
