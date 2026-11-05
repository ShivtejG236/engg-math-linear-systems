"""Mutable staging area that freezes into a ``DenseMatrix``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from engg_math.core import ShapeError, get_settings
from engg_math.units import DimensionalTracker

from ._axes import make_tracker
from ._dense import DenseMatrix
from ._errors import LabelError, StrictModeError

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy.typing as npt
    import pint
    from rich.console import RenderableType

    from ._types import Array


class EngineeringDataFrame:
    """Build a labelled, unit-tagged matrix column by column or row by row.

    Allocates freely, so it is unavailable in strict mode. Call ``to_dense`` to produce the
    immutable-by-convention ``DenseMatrix`` used by solvers.
    """

    def __init__(self, row_labels: Iterable[str] | None = None) -> None:
        """Start empty; ``row_labels`` fixes the row count and names up front."""
        if get_settings().strict_mode:
            raise StrictModeError("EngineeringDataFrame is unavailable in strict mode")
        self._row_labels: list[str] = list(row_labels) if row_labels is not None else []
        self._col_labels: list[str] = []
        self._col_units: list[pint.Unit] = []
        self._columns: list[Array] = []

    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, columns)``."""
        return len(self._row_labels), len(self._columns)

    @property
    def row_labels(self) -> tuple[str, ...]:
        """Row labels."""
        return tuple(self._row_labels)

    @property
    def col_labels(self) -> tuple[str, ...]:
        """Column labels."""
        return tuple(self._col_labels)

    @property
    def col_units(self) -> tuple[pint.Unit, ...]:
        """Column units."""
        return tuple(self._col_units)

    def add_col(
        self, label: str, values: npt.ArrayLike, unit: str | None = None
    ) -> EngineeringDataFrame:
        """Append a unique-named column of ``len(row_labels)`` values with an optional unit."""
        if label in self._col_labels:
            raise LabelError(f"column {label!r} already exists")
        array = np.array(values, dtype=get_settings().float_dtype)
        if array.ndim != 1:
            raise ShapeError(f"column values must be 1-D, got {array.ndim}-D")
        if not self._row_labels:
            self._row_labels = [str(i) for i in range(array.size)]
        elif array.size != len(self._row_labels):
            raise ShapeError(f"column has {array.size} values for {len(self._row_labels)} rows")
        tracker = make_tracker(unit, 1) if unit else make_tracker(None, 1)
        self._columns.append(array)
        self._col_labels.append(label)
        self._col_units.append(tracker[0])
        return self

    def add_row(self, label: str, values: npt.ArrayLike) -> EngineeringDataFrame:
        """Append a unique-named row with one value per existing column."""
        if not self._columns:
            raise ShapeError("add a column before adding rows")
        array = np.asarray(values, dtype=get_settings().float_dtype)
        if array.shape != (len(self._columns),):
            raise ShapeError(f"row has shape {array.shape} for {len(self._columns)} columns")
        if label in self._row_labels:
            raise LabelError(f"row {label!r} already exists")
        self._columns = [np.append(col, v) for col, v in zip(self._columns, array, strict=True)]
        self._row_labels.append(label)
        return self

    def col(self, label: str) -> Array:
        """Column ``label``."""
        try:
            return self._columns[self._col_labels.index(label)]
        except ValueError:
            raise KeyError(f"column {label!r} not found; available: {self._col_labels}") from None

    def dot(self, col_a: str, col_b: str) -> float:
        """Inner product of two named columns."""
        return float(np.dot(self.col(col_a), self.col(col_b)))

    def to_dense(self) -> DenseMatrix:
        """Freeze into a column-major ``DenseMatrix``."""
        if not self._columns:
            raise ShapeError("cannot freeze an empty dataframe")
        return DenseMatrix(
            np.column_stack(self._columns),
            row_labels=self._row_labels,
            col_labels=self._col_labels,
            col_units=DimensionalTracker(self._col_units, length=len(self._columns)),
        )

    def __repr__(self) -> str:
        """Unambiguous summary."""
        return f"EngineeringDataFrame(shape={self.shape}, col_labels={self._col_labels})"

    def __rich__(self) -> RenderableType:
        """Rich renderable (needs ``engg-math-viz``; falls back to text)."""
        from ._render import dataframe_panel  # noqa: PLC0415 - keeps rich optional

        return dataframe_panel(self)
