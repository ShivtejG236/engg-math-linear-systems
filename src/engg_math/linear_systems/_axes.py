"""Label and unit handling shared by matrix types."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pint.errors import PintError

from engg_math.core import ShapeError
from engg_math.units import DimensionalError, DimensionalTracker, check_compatible

from ._errors import LabelError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import pint


def make_labels(labels: Iterable[str] | None, length: int, axis: str) -> tuple[str, ...]:
    """Validate ``labels`` against ``length``; default to ``"0".."n-1"``."""
    if labels is None:
        return tuple(str(i) for i in range(length))
    result = tuple(labels)
    if len(result) != length:
        raise ShapeError(f"{len(result)} {axis} labels for {length} {axis}s")
    if len(set(result)) != length:
        raise LabelError(f"duplicate {axis} labels")
    return result


def make_tracker(
    units: str | pint.Unit | Sequence[str | pint.Unit] | DimensionalTracker | None, length: int
) -> DimensionalTracker:
    """Build a tracker of ``length``; unknown unit names raise ``DimensionalError``."""
    if isinstance(units, DimensionalTracker):
        units = units.units
    try:
        return DimensionalTracker(units, length=length)
    except PintError as exc:
        raise DimensionalError(f"unknown unit in {units!r}: {exc}") from exc


def check_axis_units(
    a: DimensionalTracker, b: DimensionalTracker, *, wildcard: bool, what: str
) -> None:
    """Require equal dimensionality along an axis.

    With ``wildcard`` a dimensionless entry on either side matches anything (the same rule
    as ``DimensionalTracker.validate_matmul``); without it, addition-style strictness applies.
    """
    if len(a) != len(b):
        raise ShapeError(f"{what}: axis lengths differ ({len(a)} vs {len(b)})")
    for i, (ua, ub) in enumerate(zip(a.units, b.units, strict=True)):
        if wildcard and (ua.dimensionless or ub.dimensionless):
            continue
        try:
            check_compatible(ua, ub)
        except DimensionalError as exc:
            raise DimensionalError(f"{what}, index {i}: [{ua}] vs [{ub}]") from exc
