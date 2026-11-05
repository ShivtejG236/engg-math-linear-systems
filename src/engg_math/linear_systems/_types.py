"""Shared type aliases."""

from __future__ import annotations

from typing import Any

import numpy.typing as npt

# dtype follows the configured precision (float32/float64, complex), so it is not fixed statically.
Array = npt.NDArray[Any]
