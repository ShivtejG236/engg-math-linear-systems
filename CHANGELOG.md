# Changelog

Format: [Keep a Changelog](https://keepachangelog.com), [SemVer](https://semver.org).

## [0.1.0] - unreleased

First release as `engg-math-linear-systems`, replacing the pre-ecosystem `linear_systems` package.

### Changed

- Import path is `engg_math.linear_systems`; src layout under the implicit `engg_math` namespace.
- Settings come from `engg-math-core` (`ENGG_MATH_*` variables); `settings.yaml` is gone.
- Units come from `engg-math-units`; `DimensionalTracker` and the module-level registry are removed.
- `viz`, `typer` and `questionary` are extras; the library depends on numpy, scipy, pint, core and units.
- Factorisations use LAPACK's packed forms (`lu_factor`, `cho_factor`); QR is economy-size.
- `PseudoinverseSolver(tol=...)` is now `rcond`, relative to the largest singular value.
- Labels are tuples. `read_binary` takes an explicit on-disk `order`.

### Fixed

- `RobustSolver` applied Cholesky to non-symmetric matrices by reading one triangle, giving wrong
  answers. Hermitian structure is now checked first.
- Solutions took labels from `b`; they now take the labels and units of `A`'s columns.
- `SVDFactorization.solve` failed for multi-column right-hand sides.
- `Hermitian.expand` conjugated the diagonal of complex matrices.
- `EngineeringDataFrame` created a new unit registry per column, making units incomparable.
- `EigenPair.compute(right=False)` crashed.

### Removed

- `np.linalg.inv` in `JordanCanonicalForm` (`P_inv`); use `expm`.
- `np.linalg.det` in `determinant`; now LU-based.
- `factorize_lu_in_place` (it discarded the input); use `factorize_lu(a, overwrite=True)`.
- `print` in `RobustSolver`; a `ConditioningWarning` is issued instead.
- Module-level `global_arena` and `Config` singleton.

### Added

- Dimensional checks on `add`/`subtract` and on solve right-hand sides.
- `subtract`, `log_determinant`, `JordanCanonicalForm.expm`, `LabelError`, `StrictModeError`,
  `TopologyError`, `SingularMatrixError`, `NotPositiveDefiniteError`.
