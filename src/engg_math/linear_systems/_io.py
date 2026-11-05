"""Readers producing column-major ``DenseMatrix`` objects."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING, Literal

import numpy as np

from engg_math.core import ShapeError, get_settings

from ._dense import DenseMatrix

if TYPE_CHECKING:
    import os
    from collections.abc import Iterable, Sequence


def read_csv(path: str | os.PathLike[str]) -> DenseMatrix:
    """Read a CSV whose header row names the columns and whose first column names the rows."""
    with open(path, newline="", encoding="utf-8") as handle:  # noqa: PTH123
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or len(header) < 2:  # noqa: PLR2004
            raise ShapeError(f"{path}: need a header row with a label column and >= 1 data column")
        row_labels: list[str] = []
        rows: list[list[float]] = []
        for number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(header):
                raise ShapeError(f"{path}:{number}: {len(row)} fields, header has {len(header)}")
            try:
                rows.append([float(cell) for cell in row[1:]])
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: non-numeric value ({exc})") from exc
            row_labels.append(row[0])
    array = np.array(rows, dtype=get_settings().float_dtype, order="F").reshape(len(rows), -1)
    return DenseMatrix(array, row_labels=row_labels, col_labels=header[1:], copy=False)


def read_json(path: str | os.PathLike[str]) -> DenseMatrix:
    """Read an object with ``data`` and optional ``row_labels``, ``col_labels``, ``*_units``."""
    with open(path, encoding="utf-8") as handle:  # noqa: PTH123
        try:
            content = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(content, dict) or "data" not in content:
        raise ValueError(f"{path}: expected an object with a 'data' key")
    return DenseMatrix(
        content["data"],
        row_labels=content.get("row_labels"),
        col_labels=content.get("col_labels"),
        row_units=content.get("row_units"),
        col_units=content.get("col_units"),
    )


def read_binary(  # noqa: PLR0913
    path: str | os.PathLike[str],
    shape: tuple[int, int],
    *,
    order: Literal["C", "F"] = "C",
    row_labels: Iterable[str] | None = None,
    col_labels: Iterable[str] | None = None,
    row_units: Sequence[str] | str | None = None,
    col_units: Sequence[str] | str | None = None,
) -> DenseMatrix:
    """Read raw native-endian values of the configured float dtype.

    ``order`` is the layout on disk: ``"C"`` row-major (default), ``"F"`` column-major.
    """
    flat = np.fromfile(path, dtype=get_settings().float_dtype)
    if flat.size != shape[0] * shape[1]:
        raise ShapeError(f"{path}: {flat.size} values do not match shape {shape}")
    return DenseMatrix(
        flat.reshape(shape, order=order),
        row_labels=row_labels,
        col_labels=col_labels,
        row_units=row_units,
        col_units=col_units,
    )
