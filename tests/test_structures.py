"""Unit tests: sparse, dataframe, eigen, I/O."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import scipy.linalg
from numpy.testing import assert_allclose

from engg_math.core import ShapeError, override
from engg_math.linear_systems import (
    DenseMatrix,
    EigenPair,
    EngineeringDataFrame,
    JordanCanonicalForm,
    LabelError,
    SparseMatrix,
    StrictModeError,
    TopologyError,
    read_binary,
    read_csv,
    read_json,
)
from engg_math.units import DimensionalError

if TYPE_CHECKING:
    from pathlib import Path

TOL = 1e-12  # double precision, small well-conditioned inputs


# --- sparse -----------------------------------------------------------------


def _sparse() -> SparseMatrix:
    return SparseMatrix(np.array([[4.0, 0.0, 1.0], [0.0, 3.0, 0.0], [1.0, 0.0, 2.0]]))


def test_sparse_basics() -> None:
    s = _sparse()
    assert s.shape == (3, 3)
    assert s.format == "csc"
    assert not s.frozen
    assert "nnz=5" in repr(s)
    assert s.__rich__() is not None
    assert SparseMatrix(np.eye(2), format="csr").format == "csr"
    assert s.data.nnz == 5
    assert s.dtype == np.float64


def test_sparse_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="format"):
        SparseMatrix(np.eye(2), format="coo")


def test_sparse_stream_values_requires_freeze_and_exact_count() -> None:
    s = _sparse()
    with pytest.raises(TopologyError):
        s.stream_values(np.ones(5))
    s.freeze_topology()
    assert s.frozen
    with pytest.raises(ShapeError):
        s.stream_values(np.ones(4))
    s.stream_values(np.arange(5.0))
    assert_allclose(s.data.data, np.arange(5.0), rtol=0, atol=0)
    with pytest.raises(TopologyError):
        s.optimize_ordering()


def test_sparse_ordering_returns_permutation() -> None:
    s = _sparse()
    perm = s.optimize_ordering()
    assert sorted(perm.tolist()) == [0, 1, 2]
    with pytest.raises(ShapeError):
        SparseMatrix(np.ones((2, 3))).optimize_ordering()


def test_sparse_matmul_dispatch_and_strict_mode() -> None:
    s = _sparse()
    dense = np.arange(3.0)
    assert isinstance(s @ s, SparseMatrix)
    assert isinstance(s @ DenseMatrix(dense), DenseMatrix)
    product = s @ dense
    assert isinstance(product, np.ndarray)
    assert_allclose(product, s.data.toarray() @ dense, rtol=TOL, atol=TOL)
    with pytest.raises(TypeError):
        s @ "x"
    with override(strict_mode=True), pytest.raises(StrictModeError):
        s @ s
    with pytest.raises(NotImplementedError):
        s.invert()


# --- dataframe --------------------------------------------------------------


def test_dataframe_builds_and_converts() -> None:
    df = EngineeringDataFrame(["r1", "r2"])
    df.add_col("len", [1.0, 2.0], "m").add_col("t", [3.0, 4.0], "s")
    assert df.shape == (2, 2)
    assert df.col_labels == ("len", "t")
    assert df.row_labels == ("r1", "r2")
    assert df.dot("len", "t") == pytest.approx(11.0)
    dense = df.to_dense()
    assert dense.col_labels == ("len", "t")
    assert df.__rich__() is not None
    assert "EngineeringDataFrame" in repr(df)


def test_dataframe_shares_unit_registry() -> None:
    """Regression: a new pint registry was created per column, so units never compared equal."""
    df = EngineeringDataFrame().add_col("a", [1.0], "m").add_col("b", [1.0], "m")
    assert df.col_units[0] == df.col_units[1]


def test_dataframe_add_row_and_errors() -> None:
    df = EngineeringDataFrame()
    with pytest.raises(ShapeError):
        df.add_row("x", [1.0])
    df.add_col("a", [1.0, 2.0])
    df.add_row("extra", [3.0])
    assert df.shape == (3, 1)
    with pytest.raises(LabelError):
        df.add_col("a", [1.0, 2.0, 3.0])
    with pytest.raises(LabelError):
        df.add_row("extra", [1.0])
    with pytest.raises(ShapeError):
        df.add_row("bad", [1.0, 2.0])
    with pytest.raises(ShapeError):
        df.add_col("b", [1.0])
    with pytest.raises(ShapeError):
        df.add_col("c", np.ones((3, 2)))
    with override(strict_mode=True), pytest.raises(StrictModeError):
        EngineeringDataFrame()


def test_dataframe_rejects_bad_unit() -> None:
    with pytest.raises(DimensionalError):
        EngineeringDataFrame().add_col("a", [1.0], "not_a_unit")


# --- eigen ------------------------------------------------------------------


def test_eigenpair_right_only_and_both() -> None:
    """Regression: ``right=False`` crashed."""
    m = DenseMatrix(np.array([[2.0, 0.0], [0.0, 3.0]]))
    pair = EigenPair.compute(m)
    assert_allclose(sorted(pair.values.real), [2.0, 3.0], rtol=TOL, atol=TOL)


def test_jordan_expm_matches_scipy() -> None:
    m = np.array([[1.0, 2.0], [0.0, 3.0]])
    jcf = JordanCanonicalForm.from_matrix(DenseMatrix(m))
    assert_allclose(jcf.expm(0.5).data, scipy.linalg.expm(0.5 * m), rtol=1e-10, atol=1e-12)
    assert_allclose(jcf.expm(0.0).data, np.eye(2), rtol=1e-10, atol=1e-12)


def test_jordan_expm_complex_eigenvalues_is_real() -> None:
    m = np.array([[0.0, 1.0], [-1.0, 0.0]])
    result = JordanCanonicalForm.from_matrix(DenseMatrix(m)).expm(1.0)
    assert_allclose(result.data, scipy.linalg.expm(m), rtol=1e-10, atol=1e-12)


# --- io ---------------------------------------------------------------------


def test_read_csv_and_json(tmp_path: Path) -> None:
    csv = tmp_path / "m.csv"
    csv.write_text("row,a,b\nr1,1,2\nr2,3,4\n", encoding="utf-8")
    m = read_csv(csv)
    assert_allclose(m.data, [[1, 2], [3, 4]], rtol=0, atol=0)
    assert m.row_labels == ("r1", "r2")
    assert m.col_labels == ("a", "b")
    js = tmp_path / "m.json"
    js.write_text(
        json.dumps({"data": [[1, 2], [3, 4]], "col_labels": ["a", "b"]}), encoding="utf-8"
    )
    assert_allclose(read_json(js).data, [[1, 2], [3, 4]], rtol=0, atol=0)


def test_read_io_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    with pytest.raises(OSError, match="missing"):
        read_csv(missing)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.json"):
        read_json(bad)
    ragged = tmp_path / "ragged.csv"
    ragged.write_text("row,a,b\nr1,1,2\nr2,3\n", encoding="utf-8")
    with pytest.raises(ShapeError, match="ragged"):
        read_csv(ragged)
    text = tmp_path / "text.csv"
    text.write_text("row,a\nr1,abc\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-numeric"):
        read_csv(text)
    header_only = tmp_path / "h.csv"
    header_only.write_text("row\n", encoding="utf-8")
    with pytest.raises(ShapeError):
        read_csv(header_only)
    no_data = tmp_path / "nd.json"
    no_data.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="'data'"):
        read_json(no_data)


def test_read_binary_orders(tmp_path: Path) -> None:
    data = np.arange(6.0).reshape(2, 3)
    c_file, f_file = tmp_path / "c.bin", tmp_path / "f.bin"
    data.tofile(c_file)
    data.ravel(order="F").tofile(f_file)
    assert_allclose(read_binary(c_file, (2, 3)).data, data, rtol=0, atol=0)
    assert_allclose(read_binary(f_file, (2, 3), order="F").data, data, rtol=0, atol=0)
    with pytest.raises(ShapeError):
        read_binary(c_file, (4, 4))
