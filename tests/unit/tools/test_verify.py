"""Verification selection, coverage scope, and process-tree lifetime contracts."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from tools.verify import _find_test_files, _integration_targets, _test_node_targets, run_cmd


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
