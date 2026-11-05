"""Sparse matrix with a freezable sparsity pattern."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee

from engg_math.core import ShapeError, get_settings

from ._dense import DenseMatrix
from ._errors import StrictModeError, TopologyError

if TYPE_CHECKING:
    import numpy.typing as npt
    from rich.console import RenderableType

    from ._types import Array


class SparseMatrix:
    """CSC (default) or CSR matrix. Freeze the pattern to stream new values without allocating."""

    def __init__(self, data: Any, format: str = "csc", dtype: npt.DTypeLike | None = None) -> None:  # noqa: A002, ANN401
        """Build from a dense array or any scipy sparse matrix."""
        kind = format.lower()
        if kind == "csc":
            self._matrix: Any = sp.csc_matrix(data, dtype=dtype)  # type: ignore[arg-type]
        elif kind == "csr":
            self._matrix = sp.csr_matrix(data, dtype=dtype)
        else:
            raise ValueError(f"format must be 'csc' or 'csr', got {format!r}")
        self._topology_frozen = False

    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, columns)``."""
        rows, cols = self._matrix.shape
        return int(rows), int(cols)

    @property
    def dtype(self) -> np.dtype[np.generic]:
        """Element dtype."""
        return self._matrix.dtype  # type: ignore[no-any-return]

    @property
    def data(self) -> Any:  # noqa: ANN401 - scipy sparse matrix; scipy has no common base type
        """The underlying scipy matrix (not a copy)."""
        return self._matrix

    @property
    def format(self) -> str:
        """``"csc"`` or ``"csr"``."""
        return str(self._matrix.format)

    @property
    def frozen(self) -> bool:
        """Whether the sparsity pattern is locked."""
        return self._topology_frozen

    def optimize_ordering(self) -> Array:
        """Reorder with reverse Cuthill-McKee to cut bandwidth; return the permutation."""
        if self._topology_frozen:
            raise TopologyError("cannot reorder after the topology is frozen")
        if self.shape[0] != self.shape[1]:
            raise ShapeError(f"RCM ordering needs a square matrix, got {self.shape}")
        perm: Array = reverse_cuthill_mckee(self._matrix, symmetric_mode=False)
        self._matrix = self._matrix[perm, :][:, perm]
        return perm

    def freeze_topology(self) -> None:
        """Make ``indptr`` and ``indices`` read-only."""
        self._matrix.indptr.flags.writeable = False
        self._matrix.indices.flags.writeable = False
        self._topology_frozen = True

    def stream_values(self, new_data: npt.ArrayLike) -> None:
        """Overwrite stored values in place; needs a frozen pattern and exactly ``nnz`` values."""
        if not self._topology_frozen:
            raise TopologyError("freeze the topology before streaming values")
        values = np.asarray(new_data, dtype=self.dtype)
        if values.shape != (self._matrix.nnz,):
            raise ShapeError(f"expected {self._matrix.nnz} values, got shape {values.shape}")
        np.copyto(self._matrix.data, values)

    def invert(self) -> None:
        """Raise always: factorise and solve instead."""
        raise NotImplementedError("explicit inversion is forbidden; factorise and solve")

    def __matmul__(self, other: object) -> SparseMatrix | DenseMatrix | Array:
        """Sparse x sparse -> sparse; x DenseMatrix -> DenseMatrix; x ndarray -> ndarray."""
        if get_settings().strict_mode:
            raise StrictModeError("operator @ allocates and is forbidden in strict mode")
        if isinstance(other, SparseMatrix):
            return SparseMatrix(self._matrix @ other._matrix, format=self.format)
        if isinstance(other, DenseMatrix):
            return DenseMatrix(self._matrix @ other.data, col_labels=other.col_labels, copy=False)
        if isinstance(other, np.ndarray):
            result: Array = self._matrix @ other
            return result
        return NotImplemented

    def __repr__(self) -> str:
        """Unambiguous summary."""
        return (
            f"SparseMatrix(shape={self.shape}, format='{self.format}', "
            f"nnz={self._matrix.nnz}, frozen={self._topology_frozen})"
        )

    def __rich__(self) -> RenderableType:
        """Rich renderable (needs ``engg-math-viz``; falls back to text)."""
        from ._render import sparse_panel  # noqa: PLC0415 - keeps rich optional

        return sparse_panel(self)
