"""Verification selection, coverage scope, and process-tree lifetime contracts."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from tools.verify import (
    _check_diff_coverage,
    _coverage_args,
    _find_test_files,
    _integration_targets,
    _test_node_targets,
    run_cmd,
)


def test_changed_body_selects_only_its_test_and_preserves_class_node(tmp_path: Path) -> None:
    file = tmp_path / "test_sample.py"
    file.write_text(
        "def test_first():\n    assert True\n\n"
        "class TestBook:\n    def test_second(self):\n        assert True\n",
        encoding="utf-8",
    )
    assert _test_node_targets(str(file), {2}) == [f"{file}::test_first"]
    assert _test_node_targets(str(file), {6}) == [f"{file}::TestBook::test_second"]


@pytest.mark.parametrize("changed", [None, set(), {1}, {2}, {3}, {4}])
def test_shared_fixture_or_module_change_retains_whole_file(tmp_path: Path, changed: set[int] | None) -> None:
    file = tmp_path / "test_shared.py"
    file.write_text(
        "import pytest\n@pytest.fixture\ndef shared():\n    return 1\n"
        "def test_first(shared):\n    assert shared == 1\n",
        encoding="utf-8",
    )
    assert _test_node_targets(str(file), changed) == [str(file)]


def test_test_decorator_change_keeps_parameterized_test(tmp_path: Path) -> None:
    file = tmp_path / "test_params.py"
    file.write_text(
        "@pytest.mark.parametrize('x', [1, 2])\ndef test_value(x):\n    assert x > 0\n",
        encoding="utf-8",
    )
    assert _test_node_targets(str(file), {1}) == [f"{file}::test_value"]


def test_deleted_shared_helper_keeps_whole_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    file = Path("tests/integration/test_shared.py")
    file.parent.mkdir(parents=True)
    file.write_text("def shared():\n    return 2\ndef test_value():\n    assert shared() == 2\n", encoding="utf-8")
    monkeypatch.setattr(
        "tools.verify.run_cmd",
        lambda cmd: subprocess.CompletedProcess(cmd, 0, "@@ -2,1 +2,0 @@\n-    return 1\n", ""),
    )
    assert _integration_targets([str(file)]) == [str(file)]


def test_unit_file_scope_is_not_narrowed() -> None:
    assert _integration_targets(["tests/unit/test_value.py"]) == ["tests/unit/test_value.py"]


def test_verifier_source_maps_to_its_own_regression_suite() -> None:
    tests, unmapped = _find_test_files(["tools/verify.py"])
    assert tests == ["tests/unit/tools/test_verify.py"]
    assert unmapped == []


def test_command_preserves_output_and_status() -> None:
    result = run_cmd([sys.executable, "-c", "import sys; print('failure'); sys.exit(7)"])
    assert result.returncode == 7
    assert result.stdout.strip() == "failure"


def test_command_environment_override_is_child_local() -> None:
    key = "VERIFY_ENV_BOUNDARY"
    result = run_cmd(
        [sys.executable, "-c", f"import os; print(os.environ[{key!r}])"],
        env_overrides={key: "isolated"},
    )
    assert result.stdout.strip() == "isolated"
    assert key not in os.environ


def test_timeout_terminates_forked_child_and_releases_pipes(tmp_path: Path) -> None:
    pid_path = tmp_path / "child.pid"
    code = (
        "import os,time,pathlib; pid=os.fork(); "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(pid)) if pid else None; "
        "time.sleep(60)"
    )
    started = time.monotonic()
    result = run_cmd([sys.executable, "-c", code], timeout=1)
    assert result.returncode == 124
    assert time.monotonic() - started < 10
    child_pid = int(pid_path.read_text(encoding="utf-8"))
    try:
        assert not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
    finally:
        if psutil.pid_exists(child_pid) and psutil.Process(child_pid).status() != psutil.STATUS_ZOMBIE:
            os.kill(child_pid, signal.SIGKILL)


def test_coverage_sources_are_parent_directories() -> None:
    result = _coverage_args(
        [
            "src/mhs/execution/accumulator.py",
            "src/mhs/execution/integrity.py",
            "src/common/paths.py",
        ],
        "x/coverage.json",
    )
    assert result == [
        "--cov=src/common",
        "--cov=src/mhs/execution",
        "--cov-report=json:x/coverage.json",
    ]
    assert all("src." not in arg for arg in result)
    assert not any(arg.endswith(".py") for arg in result)


def test_no_sources_without_changed_src_files() -> None:
    assert _coverage_args([], "x/coverage.json") == []


def test_directory_source_measures_numpy_package_without_reloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[3])
    test_file = tmp_path / "test_numpy_only.py"
    test_file.write_text(
        "import numpy\ndef test_numpy():\n    assert numpy.arange(3).tolist() == [0, 1, 2]\n",
        encoding="utf-8",
    )
    cov_json = tmp_path / "coverage.json"
    argv = _coverage_args(["src/mhs/execution/accumulator.py"], str(cov_json))
    log_dir = tmp_path / "logs"
    backtests_dir = tmp_path / "backtests"
    result = run_cmd(
        [sys.executable, "-m", "pytest", str(test_file), "-p", "no:cacheprovider", *argv, "-q"],
        timeout=120,
        env_overrides={
            "COVERAGE_FILE": str(tmp_path / ".coverage"),
            "CRYPTO_PILOT_LOG_DIR": str(log_dir),
            "CRYPTO_PILOT_BACKTESTS_DIR": str(backtests_dir),
        },
    )
    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, combined
    assert "cannot load module more than once" not in combined

    report = json.loads(cov_json.read_text(encoding="utf-8"))
    assert "src/mhs/execution/accumulator.py" in report.get("files", {})
    assert not log_dir.exists()


@pytest.mark.parametrize("report_bytes", [None, b"{", b"\xff"])
def test_missing_or_unreadable_report_fails_closed(tmp_path: Path, report_bytes: bytes | None) -> None:
    cov_json = tmp_path / "coverage.json"
    if report_bytes is not None:
        cov_json.write_bytes(report_bytes)
    diags, pct = _check_diff_coverage(["src/a.py"], str(cov_json), [])
    assert pct is None
    assert len(diags) == 1
    assert diags[0]["file"] == ""
    assert diags[0]["line"] == 0
    assert diags[0]["error"] == f"coverage report missing or unreadable: {cov_json}"


def test_mapped_file_absent_from_report_fails_closed(tmp_path: Path) -> None:
    cov_json = tmp_path / "coverage.json"
    cov_json.write_text(json.dumps({"files": {}}), encoding="utf-8")
    diags, _ = _check_diff_coverage(["src/a.py"], str(cov_json), [])
    assert len(diags) == 1
    assert diags[0]["file"] == "src/a.py"
    assert diags[0]["error"] == "no coverage data for src/a.py"


def test_unmapped_file_is_not_reported_as_missing_data(tmp_path: Path) -> None:
    cov_json = tmp_path / "coverage.json"
    cov_json.write_text(json.dumps({"files": {}}), encoding="utf-8")
    diags, pct = _check_diff_coverage(["src/a.py"], str(cov_json), ["src/a.py"])
    assert diags == []
    assert pct is None


def test_report_keys_match_repo_relative_posix_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cov_json = tmp_path / "coverage.json"
    cov_json.write_text(
        json.dumps(
            {
                "files": {
                    "src/mhs/execution/accumulator.py": {
                        "executed_lines": [9],
                        "missing_lines": [10],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("tools.verify._git_diff_added_lines", lambda file: {9, 10})
    diags, pct = _check_diff_coverage(["src/mhs/execution/accumulator.py"], str(cov_json), [])
    assert pct == 50
    assert len(diags) == 1
    assert diags[0]["file"] == "src/mhs/execution/accumulator.py"
    assert "[10]" in diags[0]["error"]
