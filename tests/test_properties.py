"""Property tests: algebraic invariants of factorisations and solvers."""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

from engg_math.linear_systems import (
    DenseMatrix,
    DirectSolver,
    RobustSolver,
    factorize_cholesky,
    factorize_lu,
    factorize_qr,
    factorize_svd,
)

_ELEMENTS = st.floats(-10, 10, allow_nan=False, allow_infinity=False, width=64)
_SIZE = st.integers(2, 6)


def _square(n: int) -> st.SearchStrategy[np.ndarray]:
    return arrays(np.float64, (n, n), elements=_ELEMENTS)


@st.composite
def _well_conditioned(draw: st.DrawFn) -> np.ndarray:
    """Diagonally dominant matrix: nonsingular with a modest condition number."""
    n = draw(_SIZE)
    m = draw(_square(n))
    return m + np.diag(np.abs(m).sum(axis=1) + 1.0)


@st.composite
def _spd(draw: st.DrawFn) -> np.ndarray:
    n = draw(_SIZE)
    m = draw(_square(n))
    spd: np.ndarray = m @ m.T + np.eye(n)
    return spd


@settings(deadline=None, max_examples=50)
@given(_well_conditioned())
def test_lu_reconstructs_input(m: np.ndarray) -> None:
    """REQ-LS-001: P L U = A."""
    lu = factorize_lu(DenseMatrix(m))
    # rtol/atol scale with entries <= ~70 and n <= 6; backward error is O(n * eps * |A|)
    assert_allclose(lu.P.data @ lu.L.data @ lu.U.data, m, rtol=1e-10, atol=1e-10)


@settings(deadline=None, max_examples=50)
@given(_well_conditioned(), st.data())
def test_solvers_have_small_residual(m: np.ndarray, data: st.DataObject) -> None:
    """REQ-LS-002: A x = b to within the conditioning-bounded residual."""
    b = data.draw(arrays(np.float64, (m.shape[0],), elements=_ELEMENTS))
    for solver in (DirectSolver(factorize_lu(DenseMatrix(m))), RobustSolver(DenseMatrix(m))):
        x = solver.solve(b).data[:, 0]
        assert_allclose(m @ x, b, rtol=1e-9, atol=1e-9)  # diag. dominant -> kappa small


@settings(deadline=None, max_examples=50)
@given(_spd(), st.data())
def test_cholesky_matches_lu_on_spd(m: np.ndarray, data: st.DataObject) -> None:
    """REQ-LS-003: both direct paths agree on SPD systems."""
    b = data.draw(arrays(np.float64, (m.shape[0],), elements=_ELEMENTS))
    x_llt = DirectSolver(factorize_cholesky(DenseMatrix(m))).solve(b).data
    x_lu = DirectSolver(factorize_lu(DenseMatrix(m))).solve(b).data
    # SPD with +I shift: kappa bounded by ||M||^2, so agreement is well within 1e-6
    assert_allclose(x_llt, x_lu, rtol=1e-6, atol=1e-8)


@settings(deadline=None, max_examples=50)
@given(st.integers(3, 7), st.integers(2, 3), st.data())
def test_qr_is_orthonormal_and_reconstructs(rows: int, cols: int, data: st.DataObject) -> None:
    """REQ-LS-004: Q^T Q = I and Q R = A."""
    m = data.draw(arrays(np.float64, (rows, cols), elements=_ELEMENTS))
    assume(np.linalg.matrix_rank(m) == cols)  # QR solve path needs full column rank
    qr = factorize_qr(DenseMatrix(m))
    assert_allclose(qr.Q.data.T @ qr.Q.data, np.eye(cols), rtol=0, atol=1e-12)
    assert_allclose(qr.Q.data @ qr.R.data, m, rtol=1e-10, atol=1e-10)


@settings(deadline=None, max_examples=50)
@given(st.integers(2, 6), st.integers(2, 6), st.data())
def test_svd_reconstructs_and_is_sorted(rows: int, cols: int, data: st.DataObject) -> None:
    """REQ-LS-005: U S Vh = A with non-increasing singular values."""
    m = data.draw(arrays(np.float64, (rows, cols), elements=_ELEMENTS))
    svd = factorize_svd(DenseMatrix(m))
    assert_allclose((svd.U.data * svd.s) @ svd.Vh.data, m, rtol=1e-10, atol=1e-10)
    assert np.all(np.diff(svd.s) <= 0)
