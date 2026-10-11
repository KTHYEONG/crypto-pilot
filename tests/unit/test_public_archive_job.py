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
if command:
    assert command[:4] == ["/app/.venv/bin/python", "-m", "src.cli.main", "archive"]
    step = "1" if command[4] == "roster" else "3"
else:
    step = "2"
sys.exit(int(os.environ["FAKE_RC" + step]))
'''


@pytest.mark.parametrize(("rcs", "expected"), [((0, 0, 0), 0), ((2, 0, 0), 2), ((0, 3, 0), 3), ((0, 0, 4), 0)])
def test_job_commands_and_exit_status(tmp_path: Path, rcs: tuple[int, int, int], expected: int) -> None:
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
    assert result.returncode == expected
    recorded = [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()]
    assert len(recorded) == 3
    assert recorded[0][-3:] == ["archive", "roster", "--write"]
    assert recorded[1] == ["compose", "--profile", "archive", "run", "--rm", "public-archive"]
    assert recorded[2][-2:] == ["archive", "prune"]
    for step, rc in enumerate(rcs, 1):
        assert f"[SYS] stage=public_archive step={step} rc={rc}" in log.read_text(encoding="utf-8")
