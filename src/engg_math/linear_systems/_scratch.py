"""Pre-allocated buffers for allocation-free hot loops."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ._types import Array


class Scratchpad:
    """Zero-initialised, column-major buffer reused across operations."""

    def __init__(self, shape: tuple[int, int], dtype: np.typing.DTypeLike) -> None:
        """Allocate once; ``array`` is the reusable storage."""
        self._buffer = np.zeros(shape, dtype=dtype, order="F")

    @property
    def array(self) -> Array:
        """The underlying buffer. Results written here alias it."""
        return self._buffer

    def reset(self) -> None:
        """Zero the buffer."""
        self._buffer.fill(0)


class Arena:
    """Cache of scratchpads keyed by shape and dtype."""

    def __init__(self) -> None:
        """Create an empty arena."""
        self._pads: dict[tuple[tuple[int, int], np.dtype[np.generic]], Scratchpad] = {}

    def get_pad(self, shape: tuple[int, int], dtype: np.typing.DTypeLike) -> Scratchpad:
        """Return the pad for ``(shape, dtype)``, allocating on first use."""
        key = (shape, np.dtype(dtype))
        if key not in self._pads:
            self._pads[key] = Scratchpad(shape, dtype)
        return self._pads[key]
