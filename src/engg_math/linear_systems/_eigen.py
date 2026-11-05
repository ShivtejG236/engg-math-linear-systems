"""Eigen-decomposition and diagonalisation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg as la

from engg_math.core import ShapeError, check_condition

from ._dense import DenseMatrix

if TYPE_CHECKING:
    from ._types import Array


@dataclass(frozen=True)
class EigenPair:
    """Eigenvalues and right eigenvectors of a square matrix."""

    values: Array
    vectors: DenseMatrix

    @classmethod
    def compute(cls, a: DenseMatrix) -> EigenPair:
        """Eigendecomposition ``A v = lambda v``."""
        if a.shape[0] != a.shape[1]:
            raise ShapeError(f"eigendecomposition needs a square matrix, got {a.shape}")
        w, v = la.eig(a.data)
        return cls(values=w, vectors=DenseMatrix(v))


@dataclass(frozen=True, eq=False)
class JordanCanonicalForm:
    """Diagonalisation ``A = P J P^-1`` of a diagonalisable matrix.

    ``P^-1`` is never formed; ``expm`` applies it through an LU factorisation of ``P``.
    Defective matrices (a true Jordan block) make ``P`` ill conditioned and are rejected.
    """

    P: DenseMatrix
    J: DenseMatrix
    _lu: tuple[Array, Array] = field(repr=False)
    _real: bool = field(repr=False)

    @classmethod
    def from_matrix(cls, a: DenseMatrix) -> JordanCanonicalForm:
        """Diagonalise ``a``; ``IllConditionedError`` if the eigenvectors are near-dependent."""
        pair = EigenPair.compute(a)
        p = pair.vectors
        check_condition(float(np.linalg.cond(p.data)), "eigenvector matrix (is A defective?)")
        lu, piv = la.lu_factor(p.data)
        return cls(
            P=p,
            J=DenseMatrix(np.diag(pair.values)),
            _lu=(lu, piv),
            _real=not np.iscomplexobj(a.data),
        )

    def expm(self, t: float = 1.0) -> DenseMatrix:
        """Matrix exponential ``exp(A t) = P exp(J t) P^-1``."""
        scaled = self.P.data * np.exp(np.diag(self.J.data) * t)
        # X = scaled P^-1  <=>  P^T X^T = scaled^T
        x = la.lu_solve(self._lu, scaled.T, trans=1).T
        return DenseMatrix(x.real if self._real else x)
