"""Package exceptions. All derive from the ecosystem base ``EnggMathError``."""

from __future__ import annotations

from engg_math.core import EnggMathError, NumericalError


class LabelError(EnggMathError, ValueError):
    """Row or column labels are duplicated or do not match the matrix."""


class StrictModeError(EnggMathError, RuntimeError):
    """Operation is forbidden while ``strict_mode`` is enabled."""


class TopologyError(EnggMathError, RuntimeError):
    """Sparse structure was changed in the wrong state (frozen or not yet frozen)."""


class SingularMatrixError(NumericalError):
    """Matrix is exactly singular (or rank deficient where full rank is required)."""


class NotPositiveDefiniteError(NumericalError):
    """Matrix is not Hermitian positive-definite, so Cholesky does not apply."""
