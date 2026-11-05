"""Column-major, labelled, unit-aware linear algebra with factorisation-based solvers."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ._analysis import (
    apply_householder,
    condition_number,
    determinant,
    householder_reflection,
    log_determinant,
    nullity,
    rank,
    trace,
)
from ._dataframe import EngineeringDataFrame
from ._dense import DenseMatrix
from ._eigen import EigenPair, JordanCanonicalForm
from ._errors import (
    LabelError,
    NotPositiveDefiniteError,
    SingularMatrixError,
    StrictModeError,
    TopologyError,
)
from ._factorizations import (
    FactoredRepresentation,
    LLTFactorization,
    LUFactorization,
    QRFactorization,
    SVDFactorization,
    factorize_cholesky,
    factorize_lu,
    factorize_qr,
    factorize_svd,
)
from ._hermitian import Hermitian
from ._io import read_binary, read_csv, read_json
from ._scratch import Arena, Scratchpad
from ._solvers import DirectSolver, PseudoinverseSolver, RobustSolver
from ._sparse import SparseMatrix

try:
    __version__ = version("engg-math-linear-systems")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = [
    "Arena",
    "DenseMatrix",
    "DirectSolver",
    "EigenPair",
    "EngineeringDataFrame",
    "FactoredRepresentation",
    "Hermitian",
    "JordanCanonicalForm",
    "LLTFactorization",
    "LUFactorization",
    "LabelError",
    "NotPositiveDefiniteError",
    "PseudoinverseSolver",
    "QRFactorization",
    "RobustSolver",
    "SVDFactorization",
    "Scratchpad",
    "SingularMatrixError",
    "SparseMatrix",
    "StrictModeError",
    "TopologyError",
    "__version__",
    "apply_householder",
    "condition_number",
    "determinant",
    "factorize_cholesky",
    "factorize_lu",
    "factorize_qr",
    "factorize_svd",
    "householder_reflection",
    "log_determinant",
    "nullity",
    "rank",
    "read_binary",
    "read_csv",
    "read_json",
    "trace",
]
