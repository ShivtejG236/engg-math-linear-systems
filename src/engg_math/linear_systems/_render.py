"""Rich renderables; ``rich`` and ``engg-math-viz`` are optional (text fallback)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rich.console import RenderableType

    from ._dataframe import EngineeringDataFrame
    from ._dense import DenseMatrix
    from ._hermitian import Hermitian
    from ._sparse import SparseMatrix

MAX_ROWS = 24
MAX_COLS = 10
_SPARSE_PREVIEW_CHARS = 600


def _dims(shape: tuple[int, int]) -> str:
    return f"{shape[0]}x{shape[1]}"


def _unit_text(unit: Any) -> str:  # noqa: ANN401 - pint.Unit
    return f"{unit:~P}" or "1"


def _fallback(obj: object) -> RenderableType:
    from rich.text import Text  # noqa: PLC0415

    return Text(repr(obj))


def _fmt(value: Any) -> str:  # noqa: ANN401 - numpy scalar of any dtype
    if np.iscomplexobj(value):
        return f"{value:.4g}"
    if np.issubdtype(type(value), np.floating):
        return f"{value:10.4g}"
    return str(value)


def _grid(
    array: Any,  # noqa: ANN401 - ndarray of any dtype
    row_labels: tuple[str, ...],
    col_labels: tuple[str, ...],
    *,
    dim: Any = None,  # noqa: ANN401 - callable (row, col) -> bool
) -> Any:  # noqa: ANN401 - rich.table.Table, imported lazily
    from rich.table import Table  # noqa: PLC0415

    table = Table(show_header=True, box=None, pad_edge=False, padding=(0, 1))
    table.add_column("")
    shown_cols = min(array.shape[1], MAX_COLS)
    for label in col_labels[:shown_cols]:
        table.add_column(label, justify="right")
    if array.shape[1] > shown_cols:
        table.add_column("…")
    for i in range(min(array.shape[0], MAX_ROWS)):
        cells = []
        for j in range(shown_cols):
            text = _fmt(array[i, j])
            cells.append(f"[dim]{text}[/dim]" if dim is not None and dim(i, j) else text)
        if array.shape[1] > shown_cols:
            cells.append("…")
        table.add_row(f"[bold]{row_labels[i]}[/bold]", *cells)
    if array.shape[0] > MAX_ROWS:
        table.add_row("…")
    return table


def _framed(body: Any, title: str, subtitle: str, kind: str) -> RenderableType:  # noqa: ANN401
    from rich.panel import Panel  # noqa: PLC0415

    from engg_math.viz import STYLES  # noqa: PLC0415

    return Panel(
        body,
        title=f"[bold]{title}[/bold]",
        subtitle=f"[dim]{subtitle}[/dim]",
        border_style=STYLES[kind],  # type: ignore[index]
        expand=False,
    )


def _units_line(matrix: DenseMatrix) -> str:
    rows = ", ".join(_unit_text(u) for u in matrix.row_units.units[:MAX_ROWS])
    cols = ", ".join(_unit_text(u) for u in matrix.col_units.units[:MAX_COLS])
    return f"[dim]row units[/dim] {rows}\n[dim]col units[/dim] {cols}"


def dense_panel(matrix: DenseMatrix) -> RenderableType:
    """Labelled table with an axis-unit footer."""
    try:
        from rich.console import Group  # noqa: PLC0415
        from rich.text import Text  # noqa: PLC0415

        body = Group(
            _grid(matrix.data, matrix.row_labels, matrix.col_labels),
            Text.from_markup(_units_line(matrix)),
        )
        return _framed(body, "DenseMatrix", f"{_dims(matrix.shape)} · {matrix.dtype} · F", "info")
    except ImportError:
        return _fallback(matrix)


def hermitian_panel(matrix: Hermitian) -> RenderableType:
    """Packed triangle shown solid; zeroed triangle dimmed."""
    upper = matrix.uplo == "U"
    try:
        body = _grid(
            matrix.data,
            matrix.row_labels,
            matrix.col_labels,
            dim=lambda i, j: (i > j) if upper else (i < j),
        )
        return _framed(body, "Hermitian", f"{_dims(matrix.shape)} · uplo={matrix.uplo}", "math")
    except ImportError:
        return _fallback(matrix)


def sparse_panel(matrix: SparseMatrix) -> RenderableType:
    """Render a truncated scipy listing with shape, format and fill."""
    try:
        from rich.text import Text  # noqa: PLC0415

        text = str(matrix.data)
        if len(text) > _SPARSE_PREVIEW_CHARS:
            text = text[:_SPARSE_PREVIEW_CHARS] + "\n… (truncated)"
        subtitle = f"{_dims(matrix.shape)} · {matrix.format} · nnz={matrix.data.nnz}"
        return _framed(Text(text), "SparseMatrix", subtitle, "info")
    except ImportError:
        return _fallback(matrix)


def dataframe_panel(frame: EngineeringDataFrame) -> RenderableType:
    """Columns with a units row."""
    try:
        from rich.table import Table  # noqa: PLC0415

        table = Table(show_header=True, box=None, pad_edge=False, padding=(0, 1))
        table.add_column("")
        for label in frame.col_labels:
            table.add_column(label, justify="right")
        for i, row_label in enumerate(frame.row_labels[:MAX_ROWS]):
            table.add_row(
                f"[bold]{row_label}[/bold]", *(_fmt(frame.col(c)[i]) for c in frame.col_labels)
            )
        table.add_row("[dim]units[/dim]", *(f"[dim]{_unit_text(u)}[/dim]" for u in frame.col_units))
        return _framed(table, "EngineeringDataFrame", f"{_dims(frame.shape)} · staging", "math")
    except ImportError:
        return _fallback(frame)
