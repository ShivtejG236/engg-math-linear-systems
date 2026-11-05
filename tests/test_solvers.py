"""Unit tests: factorisations, solvers and analysis."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from engg_math.core import ConditioningWarning, IllConditionedError, ShapeError, override
from engg_math.linear_systems import (
    DenseMatrix,
    DirectSolver,
    NotPositiveDefiniteError,
    PseudoinverseSolver,
    RobustSolver,
    SingularMatrixError,
    apply_householder,
    condition_number,
    determinant,
    factorize_cholesky,
    factorize_lu,
    factorize_qr,
    factorize_svd,
    householder_reflection,
    log_determinant,
    nullity,
    rank,
    trace,
)

A = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
B = np.array([1.0, 2.0, 3.0])
TOL = 1e-12  # well-conditioned 3x3, double precision


def _x(solver: DirectSolver | RobustSolver | PseudoinverseSolver, b: object = B) -> np.ndarray:
    return solver.solve(b).data[:, 0]  # type: ignore[arg-type]


def test_lu_reconstructs_matrix_and_solves() -> None:
    lu = factorize_lu(DenseMatrix(A))
    assert_allclose(lu.P.data @ lu.L.data @ lu.U.data, A, rtol=TOL, atol=TOL)
    assert_allclose(_x(DirectSolver(lu)), np.linalg.solve(A, B), rtol=TOL, atol=TOL)


def test_lu_with_row_pivoting() -> None:
    m = np.array([[0.0, 1.0], [2.0, 3.0]])
    lu = factorize_lu(DenseMatrix(m))
    assert_allclose(lu.P.data @ lu.L.data @ lu.U.data, m, rtol=TOL, atol=TOL)


def test_lu_overwrite_does_not_destroy_input() -> None:
    """Regression: in-place LU set the input array to None."""
    a = DenseMatrix(A.copy())
    factorize_lu(a, overwrite=True)
    assert a.data is not None


def test_lu_raises_on_singular() -> None:
    with pytest.raises(SingularMatrixError):
        factorize_lu(DenseMatrix(np.array([[1.0, 2.0], [2.0, 4.0]])))


def test_lu_requires_square() -> None:
    with pytest.raises(ShapeError):
        factorize_lu(DenseMatrix(np.ones((2, 3))))


def test_cholesky_solves_and_exposes_factor() -> None:
    llt = factorize_cholesky(DenseMatrix(A))
    assert_allclose(llt.L.data @ llt.L.data.T, A, rtol=TOL, atol=TOL)
    assert_allclose(_x(DirectSolver(llt)), np.linalg.solve(A, B), rtol=TOL, atol=TOL)


def test_cholesky_rejects_non_symmetric_and_indefinite() -> None:
    with pytest.raises(NotPositiveDefiniteError):
        factorize_cholesky(DenseMatrix(np.array([[1.0, 2.0], [0.0, 1.0]])))
    with pytest.raises(NotPositiveDefiniteError):
        factorize_cholesky(DenseMatrix(np.array([[1.0, 2.0], [2.0, 1.0]])))


def test_qr_least_squares() -> None:
    m = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 2.0, 2.0])
    qr = factorize_qr(DenseMatrix(m))
    assert_allclose(qr.Q.data @ qr.R.data, m, rtol=TOL, atol=TOL)
    assert_allclose(
        _x(DirectSolver(qr), b), np.linalg.lstsq(m, b, rcond=None)[0], rtol=TOL, atol=TOL
    )


def test_qr_requires_tall_matrix() -> None:
    with pytest.raises(ShapeError):
        factorize_qr(DenseMatrix(np.ones((2, 3))))


def test_svd_solve_handles_matrix_rhs() -> None:
    """Regression: SVD solve failed for 2-D right-hand sides."""
    svd = factorize_svd(DenseMatrix(A))
    rhs = np.column_stack([B, 2 * B])
    x = PseudoinverseSolver(svd).solve(rhs).data
    assert_allclose(A @ x, rhs, rtol=TOL, atol=TOL)
    assert svd.kappa == pytest.approx(np.linalg.cond(A))
    assert_allclose((svd.U.data * svd.s) @ svd.Vh.data, A, rtol=TOL, atol=TOL)


def test_pseudoinverse_gives_minimum_norm_for_rank_deficient() -> None:
    m = np.array([[1.0, 1.0], [1.0, 1.0]])
    x = _x(PseudoinverseSolver(factorize_svd(DenseMatrix(m))), np.array([2.0, 2.0]))
    assert_allclose(x, [1.0, 1.0], rtol=TOL, atol=TOL)


def test_solution_inherits_column_labels_and_units() -> None:
    """Regression: unknowns took b's labels instead of A's column labels."""
    a = DenseMatrix(A, row_labels=list("pqr"), col_labels=list("xyz"))
    b = DenseMatrix(B, row_labels=list("uvw"), col_labels=["rhs"])
    x = RobustSolver(a).solve(b)
    assert x.row_labels == ("x", "y", "z")
    assert x.col_labels == ("rhs",)


@pytest.mark.parametrize(
    ("matrix", "strategy"),
    [
        (A, "cholesky"),
        (np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 2.0]]), "lu"),
    ],
)
def test_robust_solver_picks_strategy_and_solves(matrix: np.ndarray, strategy: str) -> None:
    solver = RobustSolver(DenseMatrix(matrix))
    assert solver.strategy == strategy
    assert_allclose(_x(solver), np.linalg.solve(matrix, B), rtol=TOL, atol=TOL)


def test_robust_solver_nonsymmetric_matrix_is_not_cholesky() -> None:
    """Regression: Cholesky was applied to non-symmetric matrices, giving wrong answers."""
    m = np.array([[4.0, 3.0], [1.0, 3.0]])
    solver = RobustSolver(DenseMatrix(m))
    assert solver.strategy != "cholesky"
    assert_allclose(
        _x(solver, np.array([1.0, 2.0])), np.linalg.solve(m, [1.0, 2.0]), rtol=TOL, atol=TOL
    )


def test_robust_solver_falls_back_to_svd_with_warning() -> None:
    m = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-9]])
    with pytest.warns(ConditioningWarning):
        solver = RobustSolver(DenseMatrix(m))
    assert solver.strategy == "svd"
    assert solver.kappa > 1e7


def test_robust_solver_strict_mode_raises_on_ill_conditioned() -> None:
    m = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-9]])
    with override(strict_mode=True), pytest.raises(IllConditionedError):
        RobustSolver(DenseMatrix(m))


def test_robust_solver_rectangular_uses_svd() -> None:
    m = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    solver = RobustSolver(DenseMatrix(m))
    assert solver.strategy == "svd"
    assert solver.factorization is not None


def test_rhs_shape_mismatch_raises() -> None:
    with pytest.raises(ShapeError):
        RobustSolver(DenseMatrix(A)).solve(np.ones(2))


def test_analysis_functions() -> None:
    a = DenseMatrix(A)
    assert condition_number(a) == pytest.approx(np.linalg.cond(A))
    assert condition_number(a, 1) == pytest.approx(np.linalg.cond(A, 1))
    assert rank(a) == 3
    assert nullity(a) == 0
    assert trace(a) == pytest.approx(9.0)
    assert determinant(a) == pytest.approx(np.linalg.det(A))  # noqa: TID251 - reference value
    sign, logabs = log_determinant(a)
    assert sign * np.exp(logabs) == pytest.approx(np.linalg.det(A))  # noqa: TID251
    singular = DenseMatrix(np.array([[1.0, 2.0], [2.0, 4.0]]))
    assert rank(singular) == 1
    assert nullity(singular) == 1


def test_analysis_requires_square() -> None:
    with pytest.raises(ShapeError):
        trace(DenseMatrix(np.ones((2, 3))))


def test_householder_zeroes_subcolumn() -> None:
    x = np.array([3.0, 4.0, 0.0])
    v, beta = householder_reflection(x)
    reflected = apply_householder(DenseMatrix(x), v, beta).data[:, 0]
    assert_allclose(reflected[1:], 0.0, atol=1e-12)
    assert abs(reflected[0]) == pytest.approx(5.0)
