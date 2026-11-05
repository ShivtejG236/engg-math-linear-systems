# engg-math-linear-systems

Column-major, labelled, unit-aware dense and sparse matrices with factorisation-based solvers.
Part of the [Engineering Mathematics](https://github.com/engineering-mathematics) ecosystem (Tier 1, Foundations).

```bash
pip install engg-math-linear-systems          # library
pip install "engg-math-linear-systems[cli]"   # + terminal session and rich output
```

Python 3.11+. Imports live under the shared `engg_math` namespace:

```python
from engg_math.linear_systems import DenseMatrix, RobustSolver

a = DenseMatrix([[4.0, 2.0], [2.0, 3.0]], col_labels=["x", "y"], col_units=["m", "m"])
x = RobustSolver(a).solve([8.0, 7.0])
x.row_labels  # ('x', 'y'): unknowns carry A's column labels and units
```

## Guarantees

| Rule | Mechanism |
|---|---|
| Fortran (column-major) storage | `DenseMatrix` stores `order="F"`; I/O readers produce it directly |
| Solve, never invert | No `inv`/`pinv`/`det` in the code base (ruff `TID251`); `invert()` raises |
| Condition numbers are policy | `RobustSolver` applies `kappa_warning` / `kappa_error` from `engg_math.core` |
| Units are a safety feature | `matmul`, `add`, `subtract` and solves raise `DimensionalError` on mismatch |
| Labels are validated | Duplicate or mismatched labels raise `LabelError` / `ShapeError` |
| Strict mode | `ENGG_MATH_STRICT=1` forbids allocating operators and `EngineeringDataFrame` |
| Typed errors | All exceptions derive from `EnggMathError` |

## API

| Area | Names |
|---|---|
| Matrices | `DenseMatrix`, `Hermitian`, `SparseMatrix`, `EngineeringDataFrame` |
| Memory | `Scratchpad`, `Arena` (`out=` for allocation-free `add`/`subtract`/`matmul`) |
| Factorisations | `factorize_lu` / `_cholesky` / `_qr` / `_svd`; `overwrite=True` on LU is in place |
| Solvers | `DirectSolver`, `PseudoinverseSolver`, `RobustSolver` |
| Analysis | `condition_number`, `rank`, `nullity`, `trace`, `determinant`, `log_determinant`, Householder |
| Spectral | `EigenPair`, `JordanCanonicalForm` (diagonalisable matrices; `expm(t)` without `P^-1`) |
| I/O | `read_csv`, `read_json`, `read_binary` |

Configuration (precision, strict mode, thresholds) comes from `engg-math-core`; see its README.

## CLI

`engg-math-linear-systems` (extra `cli`) opens an interactive session; `--json` and `NO_COLOR` are honoured by the
non-interactive commands.

## Development

```bash
uv sync --all-extras
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run lint-imports
uv run pytest
```

`engg-math-core`, `-units` and `-viz` are resolved from git (`[tool.uv.sources]`) until they are on PyPI.

Licence: Apache-2.0.
