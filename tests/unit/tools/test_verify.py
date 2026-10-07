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
    PERIODIC_SLOW_GATE,
    _HEAVY_TIMEOUT_SECONDS,
    _check_diff_coverage,
    _collect_deferred_heavy,
    _coverage_args,
    _deferral_lines,
    _find_test_files,
    _integration_targets,
    _is_integration_target,
    _marker_expression,
    _slow_deferral_lines,
    _slow_marked_files,
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


@pytest.mark.parametrize(
    ("run_slow", "include_heavy", "expected"),
    [
        (False, False, "not slow and not e2e_heavy"),
        (False, True, "not slow"),
        (True, False, "slow"),
        (True, True, "slow"),
    ],
)
def test_marker_expression_defers_heavy_unless_included(
    run_slow: bool, include_heavy: bool, expected: str
) -> None:
    assert _marker_expression(run_slow=run_slow, include_heavy=include_heavy) == expected


def test_integration_target_detection() -> None:
    assert _is_integration_target("tests/integration/mhs/test_x.py::T::t") is True
    assert _is_integration_target("/abs/repo/tests/integration/test_y.py") is True
    assert _is_integration_target("tests/unit/test_z.py") is False


def test_no_collection_without_integration_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(cmd: list[str], timeout: int = 120, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"run_cmd must not be called for {cmd}")

    monkeypatch.setattr("tools.verify.run_cmd", _fail)
    assert _collect_deferred_heavy(["tests/unit/test_z.py"]) == []


def test_collected_heavy_ids_are_parsed_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake(cmd: list[str], timeout: int = 120, **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        stdout = (
            "tests/integration/mhs/test_h.py::Test::test_a[param id]\n"
            "tests/integration/mhs/test_h.py::Test::test_b\n"
            "tests/integration/mhs/test_h.py::Test::test_a[param id]\n"
            "\n"
            "2/40 tests collected (38 deselected) in 3.1s\n"
        )
        return subprocess.CompletedProcess(cmd, 0, stdout, "")

    monkeypatch.setattr("tools.verify.run_cmd", _fake)
    ids = _collect_deferred_heavy(["tests/integration/mhs/test_h.py"])
    assert ids == [
        "tests/integration/mhs/test_h.py::Test::test_a[param id]",
        "tests/integration/mhs/test_h.py::Test::test_b",
    ]
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "--collect-only" in cmd
    m_indices = [i for i, x in enumerate(cmd) if x == "-m"]
    assert cmd[m_indices[-1] + 1] == "not slow and e2e_heavy"
    assert "-q" not in cmd


def test_nothing_heavy_selected_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake(cmd: list[str], timeout: int = 120, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 5, "", "")

    monkeypatch.setattr("tools.verify.run_cmd", _fake)
    assert _collect_deferred_heavy(["tests/integration/mhs/test_h.py"]) == []


@pytest.mark.parametrize("exit_code", [2, 124])
def test_collection_failure_fails_closed(monkeypatch: pytest.MonkeyPatch, exit_code: int) -> None:
    def _fake(cmd: list[str], timeout: int = 120, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, exit_code, "first output line\nsecond\n", "err\n")

    monkeypatch.setattr("tools.verify.run_cmd", _fake)
    with pytest.raises(RuntimeError, match="first output line"):
        _collect_deferred_heavy(["tests/integration/mhs/test_h.py"])


def test_real_collection_lists_only_the_heavy_node(tmp_path: Path) -> None:
    probe = tmp_path / "tests" / "integration" / "test_tier_probe.py"
    probe.parent.mkdir(parents=True)
    probe.write_text(
        "import pytest\n"
        "def test_plain():\n    assert True\n"
        "@pytest.mark.e2e_heavy\n"
        "def test_heavy():\n    assert True\n"
        "@pytest.mark.slow\n"
        "def test_slow_only():\n    assert True\n",
        encoding="utf-8",
    )
    ids = _collect_deferred_heavy([str(probe)])
    assert len(ids) == 1
    assert ids[0].endswith("test_tier_probe.py::test_heavy")


def _run_main_with_fake(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    argv: list[str],
    collect_ids: list[str],
    pytest_exit: int,
    *,
    env_workers: str | None = None,
) -> tuple[dict[str, object], list[list[str]]]:
    import tools.verify as verify_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LEAN_CHECK_WORKERS", raising=False)
    if env_workers is not None:
        monkeypatch.setenv("LEAN_CHECK_WORKERS", env_workers)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(verify_mod, "_available_memory_gb", lambda: 8.0)
    monkeypatch.setattr(verify_mod.os, "cpu_count", lambda: 4)
    calls: list[list[str]] = []
    timeouts: dict[str, object] = {}

    def _fake(cmd: list[str], timeout: int = 120, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        if "--collect-only" in cmd:
            timeouts["collect"] = timeout
            stdout = "".join(f"{nodeid}\n" for nodeid in collect_ids)
            return subprocess.CompletedProcess(cmd, 0 if collect_ids else 5, stdout, "")
        timeouts["pytest"] = timeout
        if pytest_exit == 0:
            return subprocess.CompletedProcess(cmd, 0, "1 passed in 0.1s\n", "")
        if pytest_exit == 5:
            return subprocess.CompletedProcess(cmd, 5, "no tests ran\n", "")
        return subprocess.CompletedProcess(cmd, pytest_exit, "FAIL some test\nAssertionError: boom\n", "")

    monkeypatch.setattr(verify_mod, "run_cmd", _fake)
    return timeouts, calls


def test_default_run_defers_and_lists_heavy_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json as _json

    ids = [
        "tests/integration/mhs/test_h.py::Test::test_one",
        "tests/integration/mhs/test_h.py::Test::test_two",
    ]
    timeouts, calls = _run_main_with_fake(
        monkeypatch, tmp_path,
        ["verify", "--files", "tests/integration/mhs/test_h.py", "--skip-lint", "--skip-mypy", "--no-cov"],
        ids, 5,
    )
    import tools.verify as verify_mod

    verify_mod.main()
    out, err = capsys.readouterr()
    from tools.verify import PERIODIC_HEAVY_GATE

    assert PERIODIC_HEAVY_GATE in out
    for nodeid in ids:
        assert f"DEFERRED | {nodeid}" in out
    pytest_cmds = [c for c in calls if "--collect-only" not in c]
    assert len(pytest_cmds) == 1
    m_indices = [i for i, x in enumerate(pytest_cmds[0]) if x == "-m"]
    assert pytest_cmds[0][m_indices[-1] + 1] == "not slow and not e2e_heavy"
    payload = _json.loads(err.strip().splitlines()[-1])
    assert payload["status"] == "PASS"
    assert len([d for d in payload["diagnostics"] if "deferred e2e_heavy:" in d["error"]]) == 2


@pytest.mark.parametrize("pytest_exit", [1, 124])
def test_failure_still_lists_deferred_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], pytest_exit: int
) -> None:
    ids = ["tests/integration/mhs/test_h.py::Test::test_one"]
    _timeouts, _calls = _run_main_with_fake(
        monkeypatch, tmp_path,
        ["verify", "--files", "tests/integration/mhs/test_h.py", "--skip-lint", "--skip-mypy", "--no-cov"],
        ids, pytest_exit,
    )
    import tools.verify as verify_mod

    with pytest.raises(SystemExit) as exc:
        verify_mod.main()
    assert exc.value.code == 1
    out, _ = capsys.readouterr()
    assert f"DEFERRED | {ids[0]}" in out
    assert out.index(f"DEFERRED | {ids[0]}") < out.index("FAIL |")


def test_run_heavy_includes_tier_without_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.verify as verify_mod

    timeouts, calls = _run_main_with_fake(
        monkeypatch, tmp_path,
        ["verify", "--files", "tests/integration/mhs/test_h.py", "--skip-lint", "--skip-mypy", "--no-cov", "--run-heavy"],
        [], 0,
    )
    verify_mod.main()
    assert not any("--collect-only" in c for c in calls)
    pytest_cmds = [c for c in calls if "--collect-only" not in c]
    m_indices = [i for i, x in enumerate(pytest_cmds[0]) if x == "-m"]
    assert pytest_cmds[0][m_indices[-1] + 1] == "not slow"
    assert timeouts["pytest"] == _HEAVY_TIMEOUT_SECONDS


def test_run_heavy_respects_explicit_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.verify as verify_mod

    timeouts, _ = _run_main_with_fake(
        monkeypatch, tmp_path,
        ["verify", "--files", "tests/integration/mhs/test_h.py", "--skip-lint", "--skip-mypy", "--no-cov",
         "--run-heavy", "--timeout", "30"],
        [], 0,
    )
    verify_mod.main()
    assert timeouts["pytest"] == 30


def test_parallel_runs_keep_groups_together(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.verify as verify_mod

    _, calls = _run_main_with_fake(
        monkeypatch, tmp_path,
        ["verify", "--files", "tests/unit/test_a.py", "tests/unit/test_b.py",
         "--skip-lint", "--skip-mypy", "--no-cov"],
        [], 0, env_workers="2",
    )
    (tmp_path / "tests" / "unit").mkdir(parents=True, exist_ok=True)
    for name in ("test_a.py", "test_b.py"):
        (tmp_path / "tests" / "unit" / name).write_text("def test_x():\n    assert True\n", encoding="utf-8")
    verify_mod.main()
    pytest_cmds = [c for c in calls if "--collect-only" not in c]
    assert pytest_cmds
    cmd = pytest_cmds[0]
    assert "--dist" in cmd
    assert cmd[cmd.index("--dist") + 1] == "loadgroup"


def test_serial_run_has_no_dist_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.verify as verify_mod

    _, calls = _run_main_with_fake(
        monkeypatch, tmp_path,
        ["verify", "--files", "tests/unit/test_a.py", "--skip-lint", "--skip-mypy", "--no-cov"],
        [], 0,
    )
    (tmp_path / "tests" / "unit").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "unit" / "test_a.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    verify_mod.main()
    pytest_cmds = [c for c in calls if "--collect-only" not in c]
    assert pytest_cmds
    assert "--dist" not in pytest_cmds[0]


def test_deferral_lines_contract() -> None:
    lines, diags = _deferral_lines([])
    assert lines == []
    assert diags == []
    nodeid = "tests/integration/mhs/test_h.py::Test::test_one"
    lines, diags = _deferral_lines([nodeid])
    assert lines[0].startswith("DEFERRED | 1 e2e_heavy test(s) not run;")
    assert lines[1] == f"DEFERRED | {nodeid}"
    assert diags[0]["file"] == "tests/integration/mhs/test_h.py"
    assert diags[0]["error"] == f"deferred e2e_heavy: {nodeid}"


def test_coverage_failure_still_lists_deferred_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import tools.verify as verify_mod

    nodeid = "tests/integration/mhs/test_h.py::test_heavy"
    _run_main_with_fake(
        monkeypatch, tmp_path,
        ["verify", "--files", "src/example.py", "tests/integration/mhs/test_h.py",
         "--skip-lint", "--skip-mypy"],
        [nodeid], 0,
    )
    with pytest.raises(SystemExit) as exc:
        verify_mod.main()
    assert exc.value.code == 1
    out, err = capsys.readouterr()
    assert out.index(f"DEFERRED | {nodeid}") < out.index("FAIL | Diff Coverage")
    assert json.loads(err)["phase"] == "coverage"


def test_exit_five_without_deferred_tests_remains_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import tools.verify as verify_mod

    _run_main_with_fake(
        monkeypatch, tmp_path,
        ["verify", "--files", "tests/integration/mhs/test_h.py",
         "--skip-lint", "--skip-mypy", "--no-cov"],
        [], 5,
    )
    with pytest.raises(SystemExit) as exc:
        verify_mod.main()
    assert exc.value.code == 1
    out, err = capsys.readouterr()
    assert "DEFERRED |" not in out
    assert json.loads(err)["status"] == "FAIL"


def test_slow_marked_files_matches_code_forms_only(tmp_path: Path) -> None:
    deco = tmp_path / "test_deco.py"
    deco.write_text("import pytest\n@pytest.mark.slow\ndef test_a():\n    assert True\n", encoding="utf-8")
    single = tmp_path / "test_single.py"
    single.write_text("import pytest\npytestmark = pytest.mark.slow\ndef test_a():\n    assert True\n", encoding="utf-8")
    multi = tmp_path / "test_multi.py"
    multi.write_text(
        "import pytest\npytestmark = [pytest.mark.slow, pytest.mark.requires_market_lake]\n"
        "def test_a():\n    assert True\n",
        encoding="utf-8",
    )
    stringy = tmp_path / "test_stringy.py"
    stringy.write_text('x = "@pytest.mark.slow"\ndef test_a():\n    assert True\n', encoding="utf-8")
    broken = tmp_path / "test_broken.py"
    broken.write_text("def broken(:\n", encoding="utf-8")
    missing = tmp_path / "test_missing.py"
    unreadable = tmp_path / "test_unreadable.py"
    unreadable.write_bytes(b"\xff")
    assert _slow_marked_files([
        str(deco), str(single), str(multi), str(stringy), str(broken), str(missing), str(unreadable),
        f"{deco}::Test::test_a",
    ]) == sorted([str(deco), str(single), str(multi)])


def test_slow_deferral_lines_contract() -> None:
    assert _slow_deferral_lines([]) == ([], [])
    lines, diags = _slow_deferral_lines(["tests/unit/mhs/test_x.py"])
    assert PERIODIC_SLOW_GATE in lines[0]
    assert lines[1] == "DEFERRED | slow: tests/unit/mhs/test_x.py"
    assert diags[0]["file"] == "tests/unit/mhs/test_x.py"
    assert diags[0]["error"] == "deferred slow: tests/unit/mhs/test_x.py"


def test_default_run_lists_slow_target_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json as _json

    target = tmp_path / "tests" / "integration" / "mhs" / "test_h.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "import pytest\npytestmark = pytest.mark.slow\ndef test_one():\n    assert True\n",
        encoding="utf-8",
    )
    _run_main_with_fake(
        monkeypatch, tmp_path,
        ["verify", "--files", str(target), "--skip-lint", "--skip-mypy", "--no-cov"],
        [], 5,
    )
    import tools.verify as verify_mod

    verify_mod.main()
    out, err = capsys.readouterr()
    assert PERIODIC_SLOW_GATE in out
    assert f"DEFERRED | slow: {target}" in out
    payload = _json.loads(err.strip().splitlines()[-1])
    assert payload["status"] == "PASS"
    assert any("deferred slow:" in d["error"] for d in payload["diagnostics"])


def test_run_slow_suppresses_slow_deferral(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "tests" / "integration" / "mhs" / "test_h.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "import pytest\npytestmark = pytest.mark.slow\ndef test_one():\n    assert True\n",
        encoding="utf-8",
    )
    _, calls = _run_main_with_fake(
        monkeypatch, tmp_path,
        ["verify", "--files", str(target), "--skip-lint", "--skip-mypy", "--no-cov", "--run-slow"],
        [], 0,
    )
    import tools.verify as verify_mod

    verify_mod.main()
    out, _ = capsys.readouterr()
    assert "DEFERRED | slow:" not in out
    pytest_cmds = [c for c in calls if "--collect-only" not in c]
    m_indices = [i for i, x in enumerate(pytest_cmds[0]) if x == "-m"]
    assert pytest_cmds[0][m_indices[-1] + 1] == "slow"


@pytest.mark.parametrize("pytest_exit", [1, 124, 0])
def test_slow_deferral_survives_failure_diagnostics(tmp_path, monkeypatch, capsys, pytest_exit):
    import tools.verify as verify_mod

    target = tmp_path / "test_slow.py"
    target.write_text("import pytest\npytestmark = pytest.mark.slow\n", encoding="utf-8")
    _run_main_with_fake(
        monkeypatch, tmp_path,
        ["verify", "--files", str(target), "--skip-lint", "--skip-mypy"],
        [], pytest_exit,
    )
    if pytest_exit == 0:
        monkeypatch.setattr(verify_mod, "_coverage_args", lambda *a: ["--cov=src"])
        monkeypatch.setattr(verify_mod, "_check_diff_coverage", lambda *a: ([{"error": "uncovered"}], 0))
        monkeypatch.setattr(sys, "argv", [
            "verify", "--files", str(target), "src/example.py", "--skip-lint", "--skip-mypy",
        ])
    with pytest.raises(SystemExit) as exc:
        verify_mod.main()
    assert exc.value.code == 1
    out, err = capsys.readouterr()
    assert out.index(f"DEFERRED | slow: {target}") < out.index("FAIL |")
    assert any(d["error"] == f"deferred slow: {target}" for d in json.loads(err)["diagnostics"])


@pytest.mark.parametrize("exit_code", [2, 124])
def test_main_collection_error_stops_before_test_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], exit_code: int
) -> None:
    import tools.verify as verify_mod

    _run_main_with_fake(
        monkeypatch, tmp_path,
        ["verify", "--files", "tests/integration/mhs/test_h.py",
         "--skip-lint", "--skip-mypy", "--no-cov"],
        [], 0,
    )

    def failed_collection(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
        assert "--collect-only" in cmd
        return subprocess.CompletedProcess(cmd, exit_code, "invalid tier marking", "")

    monkeypatch.setattr(verify_mod, "run_cmd", failed_collection)
    with pytest.raises(SystemExit) as exc:
        verify_mod.main()
    assert exc.value.code == 1
    out, err = capsys.readouterr()
    assert "invalid tier marking" in out
    assert json.loads(err)["phase"] == "pytest-collect"
