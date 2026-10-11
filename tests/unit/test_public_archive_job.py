"""Hermetic execution guards for the daily public archive job."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "deploy/public-archive/run.sh"
FAKE_DOCKER = '''#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
with open(os.environ["CALLS"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(args) + "\\n")
command = args[args.index("public-archive") + 1:]
if command and command[-1] == "prune":
    step = "2"
else:
    step = "1"
sys.exit(int(os.environ["FAKE_RC" + step]))
'''


def _run_job(tmp_path: Path, rcs: tuple[int, int]) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    binary = tmp_path / "docker"
    binary.write_text(FAKE_DOCKER, encoding="utf-8")
    binary.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    log = tmp_path / "job.log"
    env = {
        **os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CRYPTO_PILOT_ROOT": str(tmp_path), "LOG_DIR": str(tmp_path),
        "LOG_FILE": str(log), "CALLS": str(calls),
        **{f"FAKE_RC{i}": str(rc) for i, rc in enumerate(rcs, 1)},
    }
    result = subprocess.run(  # noqa: S603 - fixed repo script with a hermetic Docker stub
        ["/bin/bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=10,
    )
    recorded = [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()]
    return result, recorded


@pytest.mark.parametrize(("rcs", "expected"), [((0, 0), 0), ((3, 0), 3), ((0, 4), 0)])
def test_job_commands_and_exit_status(tmp_path: Path, rcs: tuple[int, int], expected: int) -> None:
    result, recorded = _run_job(tmp_path, rcs)
    assert result.returncode == expected
    assert len(recorded) == 2
    assert recorded[0] == ["compose", "--profile", "archive", "run", "--rm", "public-archive"]
    assert recorded[1][-2:] == ["archive", "prune"]
    log = (tmp_path / "job.log").read_text(encoding="utf-8")
    for step, rc in enumerate(rcs, 1):
        assert f"[SYS] stage=public_archive step={step} rc={rc}" in log


def test_collect_failure_fails_job(tmp_path: Path) -> None:
    result, _ = _run_job(tmp_path, (3, 0))
    assert result.returncode == 3
    log = (tmp_path / "job.log").read_text(encoding="utf-8")
    assert "[SYS] stage=public_archive step=1 rc=3" in log
    assert "[SYS] stage=public_archive step=2 rc=0" in log


def test_prune_failure_does_not_fail_job(tmp_path: Path) -> None:
    result, _ = _run_job(tmp_path, (0, 4))
    assert result.returncode == 0
    log = (tmp_path / "job.log").read_text(encoding="utf-8")
    assert "[SYS] stage=public_archive step=2 rc=4" in log
