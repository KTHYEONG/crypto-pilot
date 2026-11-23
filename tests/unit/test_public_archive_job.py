"""Hermetic execution guards for the daily public archive job."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "deploy/public-archive/run.sh"
FAKE_DOCKER = '''#!/usr/bin/env python3
import glob, json, os, sys
args = sys.argv[1:]
if args[0] == "ps":
    sys.exit(0)
with open(os.environ["CALLS"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(args) + "\\n")
if "--raw-budget-gib" in args:
    if os.environ.get("FAKE_CLEAR_PENDING") == "1":
        for path in glob.glob(os.path.join(os.environ["FAKE_PENDING_DIR"], "*.json")):
            os.remove(path)
    sys.exit(0)
command = args[args.index("public-archive") + 1:]
step = "2" if command and command[-1] == "prune" else "1"
sys.exit(int(os.environ["FAKE_RC" + step]))
'''
FAKE_DF = '''#!/usr/bin/env bash
printf 'Avail\\n%sG\\n' "${FAKE_FREE_GB:-500}"
'''
FAKE_BACKUP = '''#!/usr/bin/env bash
echo '["backup"]' >> "$CALLS"
exit "${FAKE_BACKUP_RC:-0}"
'''


def _executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _run_job(
    tmp_path: Path,
    rcs: tuple[int, int],
    *,
    pending: int = 0,
    clear_pending: bool = False,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    _executable(tmp_path / "docker", FAKE_DOCKER)
    _executable(tmp_path / "df", FAKE_DF)
    _executable(tmp_path / "deploy/backup/crypto-pilot-backup.sh", FAKE_BACKUP)
    pending_dir = tmp_path / "data/public_archive/_jobs/pending"
    pending_dir.mkdir(parents=True)
    for index in range(pending):
        (pending_dir / f"job-{index}.json").write_text(json.dumps({"job_id": str(index), "raw_only": True}), encoding="utf-8")
    calls = tmp_path / "calls.jsonl"
    calls.touch()
    log = tmp_path / "job.log"
    env = {
        **os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CRYPTO_PILOT_ROOT": str(tmp_path), "LOG_DIR": str(tmp_path),
        "LOG_FILE": str(log), "CALLS": str(calls),
        "FAKE_PENDING_DIR": str(pending_dir), "FAKE_CLEAR_PENDING": "1" if clear_pending else "0",
        **{f"FAKE_RC{i}": str(rc) for i, rc in enumerate(rcs, 1)},
        **(extra_env or {}),
    }
    result = subprocess.run(  # noqa: S603 - fixed repo script with a hermetic Docker stub
        ["/bin/bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=10,
    )
    recorded = [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()]
    return result, recorded


def _raw_collects(recorded: list[list[str]]) -> list[list[str]]:
    return [call for call in recorded if "--raw-budget-gib" in call]


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


def test_busy_collect_fails_job(tmp_path: Path) -> None:
    result, _ = _run_job(tmp_path, (75, 0))
    assert result.returncode != 0
    assert result.returncode == 75


def test_raw_cycle_runs_collect_backup_prune(tmp_path: Path) -> None:
    result, recorded = _run_job(tmp_path, (0, 0), pending=1, clear_pending=True)
    assert result.returncode == 0
    assert recorded[0] == ["compose", "--profile", "archive", "run", "--rm", "public-archive"]
    assert recorded[1][-1] == "prune"
    raw = recorded[2]
    assert raw[raw.index("--drain-jobs-until") + 1] == "21:30"
    assert raw[raw.index("--raw-budget-gib") + 1] == "30"
    assert recorded[3] == ["backup"]
    assert recorded[4][-1] == "prune"
    assert len(recorded) == 5
    log = (tmp_path / "job.log").read_text(encoding="utf-8")
    assert "[SYS] stage=public_archive step=3 round=1 collect_rc=0 backup_rc=0 prune_rc=0 pending_raw=0" in log


def test_raw_budget_adds_chunk_to_today_spend(tmp_path: Path) -> None:
    status = tmp_path / "data/public_archive/_status"
    status.mkdir(parents=True)
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    (status / "raw_backfill_budget.json").write_text(
        json.dumps({"utc_day": today, "bytes": 5 << 30}), encoding="utf-8",
    )
    result, recorded = _run_job(tmp_path, (0, 0), pending=1, clear_pending=True)
    assert result.returncode == 0
    raw = _raw_collects(recorded)[0]
    assert raw[raw.index("--raw-budget-gib") + 1] == "35"


def test_raw_cycle_stops_at_deadline(tmp_path: Path) -> None:
    result, recorded = _run_job(
        tmp_path, (0, 0), pending=1, clear_pending=True, extra_env={"RAW_BACKFILL_DEADLINE_UTC": "00:00"},
    )
    assert result.returncode == 0
    assert len(recorded) == 2
    assert not _raw_collects(recorded)


def test_raw_cycle_low_disk_skips_collect(tmp_path: Path) -> None:
    result, recorded = _run_job(tmp_path, (0, 0), pending=1, extra_env={"FAKE_FREE_GB": "10"})
    assert result.returncode == 0
    assert not _raw_collects(recorded)
    assert ["backup"] in recorded
    assert recorded[-1][-1] == "prune"
    log = (tmp_path / "job.log").read_text(encoding="utf-8")
    assert "step=3 round=1 collect_rc=skipped" in log


def test_raw_cycle_no_progress_stops(tmp_path: Path) -> None:
    result, recorded = _run_job(tmp_path, (0, 0), pending=1, extra_env={"FAKE_FREE_GB": "10"})
    assert result.returncode == 0
    assert not _raw_collects(recorded)
    assert recorded.count(["backup"]) == 2
    log = (tmp_path / "job.log").read_text(encoding="utf-8")
    assert "step=3 round=2" in log
    assert "step=3 round=3" not in log


def test_raw_cycle_backup_failure_is_not_fatal(tmp_path: Path) -> None:
    result, _ = _run_job(
        tmp_path, (0, 0), pending=1, clear_pending=True, extra_env={"FAKE_BACKUP_RC": "1"},
    )
    assert result.returncode == 0
    log = (tmp_path / "job.log").read_text(encoding="utf-8")
    assert "backup_rc=1" in log
