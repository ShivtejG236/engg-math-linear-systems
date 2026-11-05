"""Hermitian matrix stored as one triangle."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from engg_math.core import ShapeError

from ._dense import DenseMatrix

if TYPE_CHECKING:
    import numpy.typing as npt
    from rich.console import RenderableType


class Hermitian(DenseMatrix):
    """Square matrix keeping only the ``uplo`` triangle of the input; the rest is zeroed."""

    def __init__(
        self, data: npt.ArrayLike, uplo: str = "U", dtype: npt.DTypeLike | None = None
    ) -> None:
        """Pack the ``"U"`` (upper) or ``"L"`` (lower) triangle of ``data``."""
        array = np.asarray(data)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:  # noqa: PLR2004
            raise ShapeError(f"Hermitian needs a square 2-D array, got shape {array.shape}")
        side = uplo.upper()
        if side not in {"U", "L"}:
            raise ValueError(f"uplo must be 'U' or 'L', got {uplo!r}")
        super().__init__(np.triu(array) if side == "U" else np.tril(array), dtype)
        self.uplo = side

    def expand(self) -> DenseMatrix:
        """Full matrix. The diagonal is taken as real, as Hermitian symmetry requires."""
        packed = self._array
        diagonal = np.diag(packed).real
        strict = packed - np.diag(np.diag(packed))
        return DenseMatrix(strict + strict.conj().T + np.diag(diagonal), dtype=self.dtype)

    def __repr__(self) -> str:
        """Unambiguous summary with the packed data."""
        return (
            f"Hermitian(shape={self.shape}, uplo='{self.uplo}', dtype={self.dtype})\n{self._array}"
        )

    def __rich__(self) -> RenderableType:
        """Rich renderable; elements zeroed by packing are dimmed."""
        from ._render import hermitian_panel  # noqa: PLC0415 - keeps rich optional

        return hermitian_panel(self)
