"""Solvers over factorisations."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np

from engg_math.core import (
    ConditioningWarning,
    IllConditionedError,
    ShapeError,
    check_condition,
    get_settings,
)

from ._axes import check_axis_units
from ._dense import DenseMatrix
from ._errors import NotPositiveDefiniteError
from ._factorizations import (
    FactoredRepresentation,
    SVDFactorization,
    factorize_cholesky,
    factorize_lu,
    factorize_svd,
    is_hermitian,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from ._types import Array

Strategy = Literal["cholesky", "lu", "svd"]


def _rhs(
    b: DenseMatrix | npt.ArrayLike, factorization: FactoredRepresentation
) -> tuple[Array, tuple[str, ...] | None]:
    """Validate ``b`` against the factored matrix; return its array and column labels."""
    info = factorization.info
    if isinstance(b, DenseMatrix):
        array, col_labels = b.data, b.col_labels
        check_axis_units(info.row_units, b.row_units, wildcard=True, what="right-hand side rows")
    else:
        array, col_labels = np.asarray(b), None
    if array.ndim not in {1, 2} or array.shape[0] != info.shape[0]:
        raise ShapeError(f"right-hand side has shape {array.shape} for {info.shape[0]} equations")
    return array, col_labels


def _solution(
    factorization: FactoredRepresentation, x: Array, col_labels: tuple[str, ...] | None
) -> DenseMatrix:
    """Label ``x``: rows are the unknowns (A's columns), columns follow ``b``."""
    info = factorization.info
    return DenseMatrix(
        x, row_labels=info.col_labels, col_labels=col_labels, row_units=info.col_units, copy=False
    )


class DirectSolver:
    """Apply a factorisation; unknowns inherit ``A``'s column labels and units."""

    def __init__(self, factorization: FactoredRepresentation) -> None:
        """Bind to ``factorization``."""
        self.factorization = factorization

    def solve(self, b: DenseMatrix | npt.ArrayLike) -> DenseMatrix:
        """Solve ``A x = b``."""
        array, col_labels = _rhs(b, self.factorization)
        return _solution(self.factorization, self.factorization.solve(array), col_labels)


class PseudoinverseSolver:
    """Minimum-norm least squares by truncated SVD; never forms ``A^+``."""

    def __init__(self, svd: SVDFactorization, rcond: float | None = None) -> None:
        """``rcond`` is relative to the largest singular value (default ``eps * max(m, n)``)."""
        self.svd = svd
        self.rcond = rcond

    def solve(self, b: DenseMatrix | npt.ArrayLike) -> DenseMatrix:
        """Solve ``min ||A x - b||`` with the smallest ``||x||``."""
        array, col_labels = _rhs(b, self.svd)
        return _solution(self.svd, self.svd.solve(array, rcond=self.rcond), col_labels)


class RobustSolver:
    """Pick a factorisation from conditioning and structure.

    Square, Hermitian, well conditioned: Cholesky; other square: LU; rectangular or
    ``kappa >= kappa_error``: truncated SVD. Between ``kappa_warning`` and ``kappa_error`` a
    ``ConditioningWarning`` is issued. In strict mode ill-conditioning raises instead of
    falling back. Thresholds come from ``engg_math.core`` settings.
    """

    def __init__(self, a: DenseMatrix) -> None:
        """Condition ``a`` and factorise."""
        if not np.all(np.isfinite(a.data)):
            raise ValueError("matrix contains NaN or infinity")
        self.a = a
        self.kappa = float(np.linalg.cond(a.data, p=2))
        self.strategy: Strategy
        self.factorization: FactoredRepresentation
        square = a.shape[0] == a.shape[1]
        if square and self._well_conditioned():
            if is_hermitian(a.data):
                try:
                    self.factorization = factorize_cholesky(a)
                except NotPositiveDefiniteError:
                    pass  # Hermitian but indefinite: LU below
                else:
                    self.strategy = "cholesky"
                    return
            self.factorization, self.strategy = factorize_lu(a), "lu"
        else:
            self.factorization, self.strategy = factorize_svd(a), "svd"

    def _well_conditioned(self) -> bool:
        try:
            check_condition(self.kappa, "RobustSolver")
        except IllConditionedError:
            if get_settings().strict_mode:
                raise
            warnings.warn(
                f"RobustSolver: kappa={self.kappa:.3e} is beyond kappa_error; using truncated SVD",
                ConditioningWarning,
                stacklevel=3,
            )
            return False
        return True

    def solve(self, b: DenseMatrix | npt.ArrayLike) -> DenseMatrix:
        """Solve with the chosen factorisation."""
        array, col_labels = _rhs(b, self.factorization)
        return _solution(self.factorization, self.factorization.solve(array), col_labels)
