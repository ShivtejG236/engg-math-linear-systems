"""Typer application: non-interactive commands plus an interactive session."""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from engg_math.core import EnggMathError, get_settings
from engg_math.linear_systems import (
    DenseMatrix,
    RobustSolver,
    condition_number,
    nullity,
    rank,
    read_csv,
    read_json,
)
from engg_math.viz import emit, kv_table, make_console, panel

if TYPE_CHECKING:
    from typing import Any

    from rich.console import Console

EXIT_FAILURE = 1

app = typer.Typer(
    name="engg-math-linear-systems",
    help="Inspect and solve linear systems with conditioning checks.",
    no_args_is_help=True,
    add_completion=False,
)

JsonOption = Annotated[bool, typer.Option("--json", help="Machine-readable output on stdout.")]
FileArgument = Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)]


def _load(path: Path) -> DenseMatrix:
    """Dispatch on the file suffix."""
    readers = {".csv": read_csv, ".json": read_json}
    reader = readers.get(path.suffix.lower())
    if reader is None:
        raise typer.BadParameter(f"{path.name}: unsupported suffix; use .csv or .json")
    return reader(path)


def _fail(message: str) -> typer.Exit:
    make_console(stderr=True).print(panel(message, "error", "error"))
    return typer.Exit(EXIT_FAILURE)


@app.command()
def inspect(file: FileArgument, as_json: JsonOption = False) -> None:
    """Report shape, rank, nullity and condition number of a matrix file."""
    try:
        matrix = _load(file)
        square = matrix.shape[0] == matrix.shape[1]
        data: dict[str, Any] = {
            "shape": list(matrix.shape),
            "rank": rank(matrix),
            "nullity": nullity(matrix),
            "condition_number": condition_number(matrix),
            "square": square,
        }
    except (EnggMathError, OSError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    emit(data, as_json=as_json, render=lambda c: c.print(kv_table(data, title=file.name)))


@app.command()
def solve(
    matrix: FileArgument,
    rhs: FileArgument,
    as_json: JsonOption = False,
) -> None:
    """Solve ``A x = b`` with the strategy chosen from the condition number."""
    try:
        a, b = _load(matrix), _load(rhs)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver = RobustSolver(a)
            x = solver.solve(b)
    except (EnggMathError, OSError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    data: dict[str, Any] = {
        "strategy": solver.strategy,
        "condition_number": solver.kappa,
        "unknowns": list(x.row_labels),
        "solution": x.data.T.tolist(),
        "warnings": [str(w.message) for w in caught],
    }

    def render(console: Console) -> None:
        console.print(kv_table({k: data[k] for k in ("strategy", "condition_number")}))
        console.print(x)
        for message in data["warnings"]:
            console.print(panel(message, "warning", "warning"))

    emit(data, as_json=as_json, render=render)


@app.command()
def settings(as_json: JsonOption = False) -> None:
    """Show the active numerical settings."""
    data = dataclasses.asdict(get_settings())
    emit(
        {k: str(v) for k, v in data.items()},
        as_json=as_json,
        render=lambda c: c.print(kv_table({k: str(v) for k, v in data.items()}, title="settings")),
    )
