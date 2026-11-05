"""LU, Cholesky, QR and SVD factorisations, used to solve rather than to invert."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg as la

from engg_math.core import ShapeError

from ._axes import make_labels, make_tracker
from ._dense import DenseMatrix
from ._errors import NotPositiveDefiniteError, SingularMatrixError

if TYPE_CHECKING:
    from engg_math.units import DimensionalTracker

    from ._types import Array

# Relative tolerance for treating a matrix as Hermitian before Cholesky (reads one triangle only).
HERMITIAN_RTOL = 1e-12


@dataclass(frozen=True)
class AxisInfo:
    """What a solution needs from the factored matrix: shape, row units, column labels and units."""

    shape: tuple[int, int]
    row_units: DimensionalTracker
    col_labels: tuple[str, ...]
    col_units: DimensionalTracker

    @classmethod
    def of(cls, a: DenseMatrix) -> AxisInfo:
        """Capture from ``a``."""
        return cls(a.shape, a.row_units, a.col_labels, a.col_units)

    @classmethod
    def default(cls, shape: tuple[int, int]) -> AxisInfo:
        """Dimensionless, numbered axes."""
        return cls(
            shape,
            make_tracker(None, shape[0]),
            make_labels(None, shape[1], "column"),
            make_tracker(None, shape[1]),
        )


class FactoredRepresentation(ABC):
    """A factorisation of ``A`` that solves ``A x = b``."""

    info: AxisInfo

    @abstractmethod
    def solve(self, b: Array) -> Array:
        """Solve for ``x`` given right-hand side ``b`` (1-D or 2-D, ``A.shape[0]`` rows)."""


def _require_square(a: DenseMatrix, what: str) -> None:
    if a.shape[0] != a.shape[1]:
        raise ShapeError(f"{what} needs a square matrix, got {a.shape}")


class LUFactorization(FactoredRepresentation):
    """``P A = L U`` held in LAPACK's packed form (``lu``, ``piv``)."""

    def __init__(self, lu: Array, piv: Array, info: AxisInfo | None = None) -> None:
        """Wrap packed LU factors from ``scipy.linalg.lu_factor``."""
        self.lu = lu
        self.piv = piv
        self.info = info or AxisInfo.default((lu.shape[0], lu.shape[1]))

    @property
    def L(self) -> DenseMatrix:  # noqa: N802
        """Unit lower-triangular factor."""
        return DenseMatrix(np.tril(self.lu, -1) + np.eye(*self.lu.shape, dtype=self.lu.dtype))

    @property
    def U(self) -> DenseMatrix:  # noqa: N802
        """Upper-triangular factor."""
        return DenseMatrix(np.triu(self.lu))

    @property
    def P(self) -> DenseMatrix:  # noqa: N802
        """Permutation matrix with ``A = P L U``."""
        perm = np.arange(self.lu.shape[0])
        for i, p in enumerate(self.piv):
            perm[[i, p]] = perm[[p, i]]
        eye = np.eye(self.lu.shape[0], dtype=self.lu.dtype)
        return DenseMatrix(eye[:, perm])

    def solve(self, b: Array) -> Array:
        """Forward and back substitution on the permuted right-hand side."""
        x: Array = la.lu_solve((self.lu, self.piv), b)
        return x


class LLTFactorization(FactoredRepresentation):
    """``A = L L^H`` for Hermitian positive-definite ``A``; no pivoting."""

    def __init__(self, c: Array, info: AxisInfo | None = None) -> None:
        """Wrap the lower-triangular result of ``scipy.linalg.cho_factor(lower=True)``."""
        self.c = c
        self.info = info or AxisInfo.default((c.shape[0], c.shape[1]))

    @property
    def L(self) -> DenseMatrix:  # noqa: N802
        """Lower-triangular factor."""
        return DenseMatrix(np.tril(self.c))

    def solve(self, b: Array) -> Array:
        """Two triangular solves."""
        x: Array = la.cho_solve((self.c, True), b)
        return x


class QRFactorization(FactoredRepresentation):
    """Economy QR, ``A = Q R`` with ``Q`` of shape ``(m, n)``; least squares for ``m >= n``."""

    def __init__(self, q: Array, r: Array, info: AxisInfo | None = None) -> None:
        """Wrap economy-size factors."""
        self.q = q
        self.r = r
        self.info = info or AxisInfo.default((q.shape[0], r.shape[1]))

    @property
    def Q(self) -> DenseMatrix:  # noqa: N802
        """Orthonormal-column factor."""
        return DenseMatrix(self.q)

    @property
    def R(self) -> DenseMatrix:  # noqa: N802
        """Upper-triangular factor."""
        return DenseMatrix(self.r)

    def solve(self, b: Array) -> Array:
        """Solve ``R x = Q^H b`` by back substitution (least squares when ``m > n``)."""
        x: Array = la.solve_triangular(self.r, self.q.conj().T @ b, lower=False)
        return x


class SVDFactorization(FactoredRepresentation):
    """``A = U diag(s) V^H``. Solves are minimum-norm least squares with truncation."""

    def __init__(self, u: Array, s: Array, vh: Array, info: AxisInfo | None = None) -> None:
        """Wrap economy SVD factors; ``s`` is descending."""
        self.u = u
        self.s = s
        self.vh = vh
        self.info = info or AxisInfo.default((u.shape[0], vh.shape[1]))

    @property
    def U(self) -> DenseMatrix:  # noqa: N802
        """Left singular vectors."""
        return DenseMatrix(self.u)

    @property
    def Vh(self) -> DenseMatrix:  # noqa: N802
        """Conjugate-transposed right singular vectors."""
        return DenseMatrix(self.vh)

    @property
    def kappa(self) -> float:
        """2-norm condition number (``inf`` when rank deficient)."""
        smallest = float(self.s[-1])
        return float("inf") if smallest == 0.0 else float(self.s[0]) / smallest

    def solve(self, b: Array, rcond: float | None = None) -> Array:
        """Minimum-norm least-squares solution.

        Singular values below ``rcond * s_max`` are dropped; ``rcond`` defaults to
        ``eps * max(m, n)``, the LAPACK ``gelsd`` convention.
        """
        eps = float(np.finfo(self.s.dtype).eps)
        cutoff = (eps * max(self.info.shape) if rcond is None else rcond) * float(self.s[0])
        keep = self.s > cutoff
        s_inv = np.zeros_like(self.s)
        s_inv[keep] = 1.0 / self.s[keep]
        projected = self.u.conj().T @ b
        scaled = s_inv.reshape(-1, *([1] * (projected.ndim - 1))) * projected
        x: Array = self.vh.conj().T @ scaled
        return x


def factorize_lu(a: DenseMatrix, *, overwrite: bool = False) -> LUFactorization:
    """LU with partial pivoting of a square matrix.

    With ``overwrite=True`` the factors are written into ``a``'s memory (no copy) and ``a``'s
    contents are destroyed.
    """
    _require_square(a, "LU")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", la.LinAlgWarning)  # exact singularity is checked below
        lu, piv = la.lu_factor(a.data, overwrite_a=overwrite)
    if np.any(np.diag(lu) == 0):
        raise SingularMatrixError("matrix is exactly singular (zero pivot in LU)")
    return LUFactorization(lu, piv, AxisInfo.of(a))


def is_hermitian(a: Array) -> bool:
    """Whether ``a`` equals its conjugate transpose within ``HERMITIAN_RTOL``."""
    return bool(
        a.shape[0] == a.shape[1] and np.allclose(a, a.conj().T, rtol=HERMITIAN_RTOL, atol=0)
    )


def factorize_cholesky(a: DenseMatrix) -> LLTFactorization:
    """Cholesky of a Hermitian positive-definite matrix."""
    _require_square(a, "Cholesky")
    if not is_hermitian(a.data):
        raise NotPositiveDefiniteError("matrix is not Hermitian; Cholesky reads one triangle only")
    try:
        c, _ = la.cho_factor(a.data, lower=True)
    except la.LinAlgError as exc:
        raise NotPositiveDefiniteError(str(exc)) from exc
    return LLTFactorization(c, AxisInfo.of(a))


def factorize_qr(a: DenseMatrix) -> QRFactorization:
    """Economy QR of a matrix with at least as many rows as columns, full column rank."""
    if a.shape[0] < a.shape[1]:
        raise ShapeError(f"QR solve needs rows >= columns, got {a.shape}")
    q, r = la.qr(a.data, mode="economic")
    if np.any(np.diag(r) == 0):
        raise SingularMatrixError("matrix is rank deficient (zero on the diagonal of R)")
    return QRFactorization(q, r, AxisInfo.of(a))


def factorize_svd(a: DenseMatrix) -> SVDFactorization:
    """Economy SVD of any matrix."""
    u, s, vh = la.svd(a.data, full_matrices=False)
    return SVDFactorization(u, s, vh, AxisInfo.of(a))
