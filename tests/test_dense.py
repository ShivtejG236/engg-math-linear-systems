"""Unit tests: DenseMatrix, Hermitian, Scratchpad, labels and units."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from engg_math.core import ShapeError, override
from engg_math.linear_systems import (
    Arena,
    DenseMatrix,
    Hermitian,
    LabelError,
    Scratchpad,
    StrictModeError,
)
from engg_math.units import DimensionalError


def test_storage_is_column_major() -> None:
    m = DenseMatrix([[1.0, 2.0], [3.0, 4.0]])
    assert m.data.flags.f_contiguous


def test_vector_input_becomes_column() -> None:
    assert DenseMatrix([1.0, 2.0, 3.0]).shape == (3, 1)


def test_column_lookup_and_dot() -> None:
    m = DenseMatrix([[1.0, 2.0], [3.0, 4.0]], col_labels=["a", "b"])
    assert_allclose(m.col("b"), [2.0, 4.0], rtol=0, atol=0)
    assert m.dot("a", "b") == pytest.approx(14.0)


def test_duplicate_labels_raise() -> None:
    with pytest.raises(LabelError):
        DenseMatrix(np.eye(2), col_labels=["a", "a"])


def test_label_length_mismatch_raises() -> None:
    with pytest.raises(ShapeError):
        DenseMatrix(np.eye(2), row_labels=["a"])


def test_unknown_column_raises() -> None:
    with pytest.raises(KeyError):
        DenseMatrix(np.eye(2), col_labels=["a", "b"]).col("z")


def test_add_subtract_matmul_with_scratchpad() -> None:
    a = DenseMatrix([[1.0, 2.0], [3.0, 4.0]])
    b = DenseMatrix(np.eye(2))
    pad = Scratchpad((2, 2), np.float64)
    assert_allclose(a.add(b, out=pad).data, [[2, 2], [3, 5]], rtol=0, atol=0)
    assert_allclose(a.subtract(b).data, [[0, 2], [3, 3]], rtol=0, atol=0)
    assert_allclose(a.matmul(b).data, a.data, rtol=0, atol=0)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ShapeError):
        DenseMatrix(np.eye(2)).add(DenseMatrix(np.eye(3)))
    with pytest.raises(ShapeError):
        DenseMatrix(np.ones((2, 3))).matmul(DenseMatrix(np.ones((2, 3))))


def test_add_with_incompatible_units_raises() -> None:
    a = DenseMatrix(np.eye(2), row_units=["m", "m"], col_units=["m", "m"])
    b = DenseMatrix(np.eye(2), row_units=["s", "s"], col_units=["s", "s"])
    with pytest.raises(DimensionalError):
        a.add(b)


def test_invert_is_forbidden() -> None:
    with pytest.raises(NotImplementedError):
        DenseMatrix(np.eye(2)).invert()


def test_operators_work_and_are_blocked_in_strict_mode() -> None:
    a = DenseMatrix(np.eye(2))
    assert_allclose((a + a).data, 2 * np.eye(2), rtol=0, atol=0)
    assert_allclose((a - a).data, 0, rtol=0, atol=0)
    assert_allclose((a @ a).data, np.eye(2), rtol=0, atol=0)
    with override(strict_mode=True):
        for op in (a.__add__, a.__sub__, a.__matmul__):
            with pytest.raises(StrictModeError):
                op(a)


def test_repr_and_rich_render() -> None:
    a = DenseMatrix(np.eye(2), col_labels=["x", "y"])
    assert "DenseMatrix" in repr(a)
    assert a.__rich__() is not None


def test_scratchpad_reset_and_arena_reuse() -> None:
    pad = Scratchpad((2, 2), np.float64)
    pad.array[:] = 5
    pad.reset()
    assert not pad.array.any()
    arena = Arena()
    assert arena.get_pad((2, 2), np.float64) is arena.get_pad((2, 2), np.float64)


def test_hermitian_expand_keeps_real_diagonal_of_complex_matrix() -> None:
    """Regression: the diagonal was conjugated on expansion."""
    full = np.array([[2.0, 1 - 1j], [1 + 1j, 3.0]])
    h = Hermitian(full, uplo="U")
    assert_allclose(h.expand().data, full, rtol=0, atol=0)


def test_hermitian_lower_triangle() -> None:
    full = np.array([[2.0, 1.0], [1.0, 3.0]])
    assert_allclose(Hermitian(full, uplo="L").expand().data, full, rtol=0, atol=0)
    assert "Hermitian" in repr(Hermitian(full))
    assert Hermitian(full).__rich__() is not None
