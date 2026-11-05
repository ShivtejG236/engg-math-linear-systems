"""Matrix properties and Householder reflections."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from engg_math.core import ShapeError

from ._dense import DenseMatrix

if TYPE_CHECKING:
    import numpy.typing as npt

    from ._types import Array


def _require_square(a: DenseMatrix, what: str) -> None:
    if a.shape[0] != a.shape[1]:
        raise ShapeError(f"{what} is defined for square matrices, got {a.shape}")


def condition_number(a: DenseMatrix, p: float | str | None = None) -> float:
    """Condition number in norm ``p`` (2-norm by default, from the SVD)."""
    return float(np.linalg.cond(a.data, p=p))  # type: ignore[arg-type]


def rank(a: DenseMatrix, tol: float | None = None) -> int:
    """Numerical rank from the SVD; ``tol`` defaults to ``s_max * max(m, n) * eps``."""
    return int(np.linalg.matrix_rank(a.data, tol=tol))


def nullity(a: DenseMatrix, tol: float | None = None) -> int:
    """Dimension of the null space: ``columns - rank``."""
    return a.shape[1] - rank(a, tol=tol)


def trace(a: DenseMatrix) -> float | complex:
    """Sum of the diagonal of a square matrix."""
    _require_square(a, "trace")
    return np.trace(a.data).item()  # type: ignore[no-any-return]


def log_determinant(a: DenseMatrix) -> tuple[float | complex, float]:
    """``(sign, log|det|)`` from an LU factorisation; avoids overflow of the raw determinant."""
    _require_square(a, "determinant")
    sign, logabs = np.linalg.slogdet(a.data)
    return sign.item(), float(logabs)


def determinant(a: DenseMatrix) -> float | complex:
    """Return ``sign * exp(log|det|)``; prefer ``log_determinant`` for large matrices."""
    sign, logabs = log_determinant(a)
    return sign * float(np.exp(logabs))


def householder_reflection(x: npt.ArrayLike) -> tuple[DenseMatrix, float]:
    """Return ``(v, beta)`` with ``v[0] = 1`` so ``H = I - beta v v^T`` maps ``x`` to ``c e1``."""
    vec = np.asarray(x, dtype=float).ravel()
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return DenseMatrix(vec.reshape(-1, 1)), 0.0
    sign = 1.0 if vec[0] >= 0 else -1.0
    head = vec[0] + sign * norm
    v = vec / head
    v[0] = 1.0
    tail = float(np.linalg.norm(vec[1:]))
    return DenseMatrix(v.reshape(-1, 1)), 2.0 * head**2 / (head**2 + tail**2)


def apply_householder(a: DenseMatrix, v: DenseMatrix, beta: float) -> DenseMatrix:
    """``H A`` for ``H = I - beta v v^T``, without forming ``H``."""
    if v.shape[0] != a.shape[0] or v.shape[1] != 1:
        raise ShapeError(f"v has shape {v.shape}, A has {a.shape[0]} rows")
    w: Array = beta * (v.data.T @ a.data)
    return DenseMatrix(a.data - v.data @ w, dtype=a.dtype)
