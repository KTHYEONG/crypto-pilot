"""Ownership fixup contract: read-only check, root-helper re-own, and refusal of root callers.

Real `find` walks real files; `id` and `docker` are faked on PATH so the caller uid can differ from
the file owners without root privileges.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy" / "crypto-ownership-fixup.sh"

_FAKE_ID = """#!/usr/bin/env bash
case "${1:-}" in
  -u) echo "$FAKE_UID" ;;
  -g) echo "$FAKE_GID" ;;
  *) exit 2 ;;
esac
"""

# Stands in for `find <path> ( ! -user U -o ! -group G ) -print -quit` once the root helper has re-owned.
_FAKE_FIND = """#!/usr/bin/env bash
if [ -f "$FAKE_STATE/reowned" ]; then exit 0; fi
echo "$1/entry.txt"
"""

_FAKE_DOCKER = """#!/usr/bin/env bash
echo "docker $*" >> "$FAKE_LOG"
if [ "${1:-}" = run ]; then
  if [ "${FAKE_NO_REOWN:-0}" != 1 ]; then : > "$FAKE_STATE/reowned"; fi
  exit 0
fi
exit 0
"""


def _write_exec(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _populate(root: Path) -> None:
    for rel in ("data/live_capture/raw", "logs", "deploy/mhs"):
        (root / rel).mkdir(parents=True, exist_ok=True)
        (root / rel / "entry.txt").write_text("x", encoding="utf-8")


def _run(
    tmp_path: Path,
    mode: str,
    *,
    fake_uid: int,
    fake_no_reown: bool = False,
    populate: bool = True,
    fake_find: bool = False,
):
    real_gid = os.getgid()
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    _write_exec(bindir / "id", _FAKE_ID)
    _write_exec(bindir / "docker", _FAKE_DOCKER)
    if fake_find:
        _write_exec(bindir / "find", _FAKE_FIND)
    log = tmp_path / "argv.log"
    log.write_text("", encoding="utf-8")
    root = tmp_path / "home" / "crypto-pilot"
    if populate:
        _populate(root)
    else:
        root.mkdir(parents=True, exist_ok=True)
    env = {
        "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}",
        "HOME": str(tmp_path),
        "CRYPTO_FIXUP_ROOT": str(root),
        "FAKE_UID": str(fake_uid),
        "FAKE_GID": str(real_gid),
        "FAKE_LOG": str(log),
        "FAKE_STATE": str(state),
        "FAKE_NO_REOWN": "1" if fake_no_reown else "0",
    }
    result = subprocess.run(  # noqa: S603 - fixed argv: the repo script under test
        [str(shutil.which("bash") or "/bin/bash"), str(SCRIPT), mode],
        capture_output=True, text=True, check=False, env=env, cwd=tmp_path, timeout=60,
    )
    docker_lines = [line for line in log.read_text(encoding="utf-8").splitlines() if line]
    return result, docker_lines


def _require_non_root_caller() -> None:
    if os.getuid() == 0:
        pytest.skip("clean-tree case needs a non-root caller so the fake uid can match file owners")


def test_check_clean_exits_zero(tmp_path) -> None:
    _require_non_root_caller()
    result, docker_lines = _run(tmp_path, "check", fake_uid=os.getuid())
    assert result.returncode == 0
    assert "foreign=0" in result.stdout
    assert "status=clean" in result.stdout
    assert docker_lines == []


def test_check_foreign_exits_10_without_touching_docker(tmp_path) -> None:
    result, docker_lines = _run(tmp_path, "check", fake_uid=os.getuid() + 1)
    assert result.returncode == 10
    assert "foreign=1" in result.stdout
    assert "status=foreign" in result.stdout
    assert docker_lines == []


def test_fix_reowns_then_rechecks(tmp_path) -> None:
    uid = os.getuid() + 1
    gid = os.getgid()
    result, docker_lines = _run(tmp_path, "fix", fake_uid=uid, fake_find=True)
    assert result.returncode == 0, result.stderr
    assert "status=fixed" in result.stdout
    assert len(docker_lines) == 3
    for line in docker_lines:
        assert "run --rm --pull never --network none --user 0:0" in line
        assert "--entrypoint /usr/bin/find" in line
        assert f"-exec chown -h {uid}:{gid} {{}} +" in line
        assert " -R " not in line


def test_fix_residual_fails(tmp_path) -> None:
    result, docker_lines = _run(tmp_path, "fix", fake_uid=os.getuid() + 1, fake_no_reown=True, fake_find=True)
    assert result.returncode != 0
    assert "status=residual" in result.stdout
    assert len(docker_lines) == 3


def test_fix_on_clean_tree_skips_docker(tmp_path) -> None:
    _require_non_root_caller()
    result, docker_lines = _run(tmp_path, "fix", fake_uid=os.getuid())
    assert result.returncode == 0
    assert "status=clean" in result.stdout
    assert docker_lines == []


def test_root_caller_refused(tmp_path) -> None:
    result, docker_lines = _run(tmp_path, "check", fake_uid=0)
    assert result.returncode != 0
    assert "root_caller" in result.stderr
    assert docker_lines == []


def test_absent_paths_are_skipped(tmp_path) -> None:
    _require_non_root_caller()
    result, _ = _run(tmp_path, "check", fake_uid=os.getuid(), populate=False)
    assert result.returncode == 0
    assert "foreign=0" in result.stdout


def test_unknown_mode_is_usage_error() -> None:
    result = subprocess.run(  # noqa: S603 - fixed argv: the repo script under test
        [str(shutil.which("bash") or "/bin/bash"), str(SCRIPT), "wipe"],
        capture_output=True, text=True, check=False, timeout=30,
    )
    assert result.returncode == 2


def test_script_never_recurses_or_escalates() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "chown -R" not in text
    assert "sudo" not in text
    assert "-delete" not in text
    assert not any(line.strip().startswith("rm ") for line in text.splitlines())
