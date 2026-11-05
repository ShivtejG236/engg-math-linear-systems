"""CLI behaviour through typer's runner."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from engg_math.linear_systems.cli import app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


@pytest.fixture(autouse=True)
def _plain_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")


def _write(path: Path, rows: list[list[float]], cols: list[str]) -> Path:
    lines = ["row," + ",".join(cols)]
    lines += [f"r{i}," + ",".join(map(str, row)) for i, row in enumerate(rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_inspect_json(tmp_path: Path) -> None:
    f = _write(tmp_path / "a.csv", [[2.0, 0.0], [0.0, 1.0]], ["x", "y"])
    result = runner.invoke(app, ["inspect", str(f), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["shape"] == [2, 2]
    assert data["rank"] == 2
    assert data["condition_number"] == pytest.approx(2.0)


def test_inspect_human(tmp_path: Path) -> None:
    f = _write(tmp_path / "a.csv", [[2.0, 0.0], [0.0, 1.0]], ["x", "y"])
    result = runner.invoke(app, ["inspect", str(f)])
    assert result.exit_code == 0
    assert "rank" in result.stdout


def test_solve_json(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.csv", [[2.0, 0.0], [0.0, 4.0]], ["x", "y"])
    b = _write(tmp_path / "b.csv", [[2.0], [4.0]], ["rhs"])
    result = runner.invoke(app, ["solve", str(a), str(b), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["unknowns"] == ["x", "y"]
    assert data["solution"][0] == pytest.approx([1.0, 1.0], rel=1e-12)  # Cholesky rounding
    assert data["strategy"] in {"cholesky", "lu"}


def test_solve_human_shows_warning_for_ill_conditioned(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.csv", [[1.0, 1.0], [1.0, 1.0 + 1e-9]], ["x", "y"])
    b = _write(tmp_path / "b.csv", [[1.0], [1.0]], ["rhs"])
    result = runner.invoke(app, ["solve", str(a), str(b)])
    assert result.exit_code == 0
    assert "warning" in result.stdout.lower()


def test_solve_shape_mismatch_fails_with_message(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.csv", [[1.0, 0.0], [0.0, 1.0]], ["x", "y"])
    b = _write(tmp_path / "b.csv", [[1.0], [1.0], [1.0]], ["rhs"])
    result = runner.invoke(app, ["solve", str(a), str(b)])
    assert result.exit_code == 1
    assert "right-hand side" in result.output


def test_unsupported_suffix_is_usage_error(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x", encoding="utf-8")
    assert runner.invoke(app, ["inspect", str(f)]).exit_code == 2


def test_missing_file_is_usage_error(tmp_path: Path) -> None:
    assert runner.invoke(app, ["inspect", str(tmp_path / "none.csv")]).exit_code == 2


def test_malformed_file_exits_nonzero(tmp_path: Path) -> None:
    f = tmp_path / "a.json"
    f.write_text("{", encoding="utf-8")
    assert runner.invoke(app, ["inspect", str(f)]).exit_code == 1


def test_settings_json_and_human() -> None:
    data = json.loads(runner.invoke(app, ["settings", "--json"]).stdout)
    assert data["precision"] == "float64"
    assert "kappa_warning" in runner.invoke(app, ["settings"]).stdout


def test_no_args_shows_help() -> None:
    assert "inspect" in runner.invoke(app, []).output
