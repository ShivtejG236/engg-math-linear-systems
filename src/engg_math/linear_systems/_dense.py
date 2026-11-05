"""Dense, column-major, labelled and unit-aware matrix."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from engg_math.core import ShapeError, get_settings
from engg_math.units import DimensionalTracker

from ._axes import check_axis_units, make_labels, make_tracker
from ._errors import StrictModeError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    import numpy.typing as npt
    import pint
    from rich.console import RenderableType

    from ._scratch import Scratchpad
    from ._types import Array


class DenseMatrix:
    """2-D array in Fortran order with row/column labels and per-axis units.

    ``A[i, j]`` has unit ``row_units[i] / col_units[j]``. 1-D input becomes a column vector.
    """

    def __init__(  # noqa: PLR0913
        self,
        data: npt.ArrayLike,
        dtype: npt.DTypeLike | None = None,
        *,
        row_labels: Iterable[str] | None = None,
        col_labels: Iterable[str] | None = None,
        row_units: str | pint.Unit | Sequence[str | pint.Unit] | DimensionalTracker | None = None,
        col_units: str | pint.Unit | Sequence[str | pint.Unit] | DimensionalTracker | None = None,
        copy: bool = True,
    ) -> None:
        """Wrap ``data``.

        ``dtype`` defaults to the configured float (or complex) dtype. With ``copy=False`` the
        array is used as is when already Fortran-contiguous with the right dtype.
        """
        raw = np.asarray(data)
        if dtype is None:
            settings = get_settings()
            dtype = settings.complex_dtype if np.iscomplexobj(raw) else settings.float_dtype
        array = (
            np.array(raw, dtype=dtype, order="F") if copy else np.asfortranarray(raw, dtype=dtype)
        )
        if array.ndim == 1:
            array = array.reshape(-1, 1, order="F")
        elif array.ndim != 2:  # noqa: PLR2004
            raise ShapeError(f"expected 1-D or 2-D data, got {array.ndim}-D")
        self._array: Array = array
        rows, cols = array.shape
        self.row_labels = make_labels(row_labels, rows, "row")
        self.col_labels = make_labels(col_labels, cols, "column")
        self._col_index = {label: i for i, label in enumerate(self.col_labels)}
        self.row_units = make_tracker(row_units, rows)
        self.col_units = make_tracker(col_units, cols)

    # ---- properties

    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, columns)``."""
        rows, cols = self._array.shape
        return rows, cols

    @property
    def dtype(self) -> np.dtype[np.generic]:
        """Element dtype."""
        return self._array.dtype

    @property
    def data(self) -> Array:
        """The Fortran-ordered array (not a copy)."""
        return self._array

    # ---- access

    def col(self, label: str) -> Array:
        """Column ``label`` as a 1-D view."""
        try:
            return self._array[:, self._col_index[label]]
        except KeyError:
            raise KeyError(
                f"column {label!r} not found; available: {list(self.col_labels)}"
            ) from None

    def dot(self, col_a: str, col_b: str) -> float | complex:
        """Inner product of two named columns (no conjugation)."""
        return np.dot(self.col(col_a), self.col(col_b)).item()  # type: ignore[no-any-return]

    # ---- arithmetic

    def _like(self, array: Array, rows_from: DenseMatrix, cols_from: DenseMatrix) -> DenseMatrix:
        return DenseMatrix(
            array,
            dtype=array.dtype,
            row_labels=rows_from.row_labels,
            col_labels=cols_from.col_labels,
            row_units=rows_from.row_units,
            col_units=cols_from.col_units,
            copy=False,
        )

    @staticmethod
    def _target(
        out: Scratchpad | None, shape: tuple[int, int], dtype: np.dtype[np.generic]
    ) -> Array:
        if out is None:
            return np.empty(shape, dtype=dtype, order="F")
        if out.array.shape != shape:
            raise ShapeError(f"scratchpad has shape {out.array.shape}, result needs {shape}")
        return out.array

    def _elementwise(
        self,
        other: DenseMatrix,
        ufunc: Callable[..., object],
        out: Scratchpad | None,
        what: str,
    ) -> DenseMatrix:
        if not isinstance(other, DenseMatrix):
            raise TypeError(f"{what} needs a DenseMatrix, got {type(other).__name__}")
        if self.shape != other.shape:
            raise ShapeError(f"{what}: shapes {self.shape} and {other.shape} differ")
        check_axis_units(self.row_units, other.row_units, wildcard=False, what=f"{what} rows")
        check_axis_units(self.col_units, other.col_units, wildcard=False, what=f"{what} columns")
        target = self._target(out, self.shape, np.result_type(self.dtype, other.dtype))
        ufunc(self._array, other._array, out=target)
        return self._like(target, self, self)

    def add(self, other: DenseMatrix, out: Scratchpad | None = None) -> DenseMatrix:
        """Add ``other``; with ``out`` the result is written there and aliases it."""
        return self._elementwise(other, np.add, out, "add")

    def subtract(self, other: DenseMatrix, out: Scratchpad | None = None) -> DenseMatrix:
        """``self - other``; result is written to ``out`` when given."""
        return self._elementwise(other, np.subtract, out, "subtract")

    def matmul(self, other: DenseMatrix, out: Scratchpad | None = None) -> DenseMatrix:
        """Multiply by ``other`` after a dimensional check (rows of self, columns of other)."""
        if not isinstance(other, DenseMatrix):
            raise TypeError(f"matmul needs a DenseMatrix, got {type(other).__name__}")
        if self.shape[1] != other.shape[0]:
            raise ShapeError(f"matmul: shapes {self.shape} and {other.shape} are not aligned")
        DimensionalTracker.validate_matmul(self.col_units, other.row_units)
        shape = (self.shape[0], other.shape[1])
        target = self._target(out, shape, np.result_type(self.dtype, other.dtype))
        np.matmul(self._array, other._array, out=target)
        return self._like(target, self, other)

    def invert(self) -> None:
        """Raise always: factorise and solve instead."""
        raise NotImplementedError(
            "explicit inversion is forbidden; factorise (lu/cholesky/qr/svd) and solve"
        )

    # ---- operators (allocate; refused in strict mode)

    @staticmethod
    def _require_lenient(op: str) -> None:
        if get_settings().strict_mode:
            raise StrictModeError(
                f"operator {op} allocates and is forbidden in strict mode; "
                "use add/subtract/matmul with out=Scratchpad"
            )

    def __matmul__(self, other: DenseMatrix) -> DenseMatrix:
        """``self @ other``."""
        self._require_lenient("@")
        return self.matmul(other)

    def __add__(self, other: DenseMatrix) -> DenseMatrix:
        """``self + other``."""
        self._require_lenient("+")
        return self.add(other)

    def __sub__(self, other: DenseMatrix) -> DenseMatrix:
        """``self - other``."""
        self._require_lenient("-")
        return self.subtract(other)

    # ---- presentation

    def __repr__(self) -> str:
        """Unambiguous summary with the data."""
        return f"DenseMatrix(shape={self.shape}, dtype={self.dtype}, order='F')\n{self._array}"

    def __rich__(self) -> RenderableType:
        """Rich renderable (needs ``engg-math-viz``; falls back to text)."""
        from ._render import dense_panel  # noqa: PLC0415 - keeps rich optional

        return dense_panel(self)
