"""Hermetic invariant guards for the repo-owned Drive backup writer."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy" / "backup" / "crypto-pilot-backup.sh"

TODAY = "2026-09-23"
REMOTE_ROOT = "testremote:root"

FAKE_RCLONE_SRC = """#!/usr/bin/env python3
import json, os, sys
record = os.environ["FAKE_RECORD"]
args = sys.argv[1:]
with open(record, "a", encoding="utf-8") as f:
    f.write(json.dumps(args) + "\\n")
sub = args[0] if args else ""
if sub == "lsf":
    rc = int(os.environ.get("FAKE_LSF_RC", "0"))
    out = os.environ.get("FAKE_LSF_OUTPUT", "")
    if out:
        sys.stdout.write(out)
    sys.exit(rc)
if sub == "copy":
    joined = " ".join(args)
    if "logs/live/orders" in joined:
        sys.exit(int(os.environ.get("FAKE_ORDERS_RC", "0")))
    sys.exit(int(os.environ.get("FAKE_DATA_RC", "0")))
if sub == "purge":
    sys.exit(int(os.environ.get("FAKE_PURGE_RC", "0")))
sys.exit(0)
"""


def _setup_base(tmp_path: Path, *, with_orders: bool) -> dict[str, str]:
    root = tmp_path / "root"
    data = root / "data"
    data.mkdir(parents=True)
    (data / "state").mkdir(parents=True, exist_ok=True)
    deploy_dir = root / "deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    (deploy_dir / "crypto-pilot.rclone-filter").write_text("+ /state/**\n- **\n", encoding="utf-8")
    if with_orders:
        orders = root / "logs" / "live" / "orders"
        orders.mkdir(parents=True)
        (orders / "2026-09-22.jsonl").write_text('{"o":1}\n', encoding="utf-8")
    fake = tmp_path / "fake-rclone"
    fake.write_text(FAKE_RCLONE_SRC, encoding="utf-8")
    fake.chmod(0o755)
    record = tmp_path / "rclone.jsonl"
    record.write_text("", encoding="utf-8")
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lock = tmp_path / "quant-gdrive.lock"
    lock.touch()
    return {
        "root": str(root),
        "fake": str(fake),
        "record": str(record),
        "log_dir": str(log_dir),
        "lock": str(lock),
    }


def _run(
    base: dict[str, str], tmp_path: Path, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    _ = tmp_path
    env = os.environ.copy()
    env.update(
        {
            "CRYPTO_PILOT_ROOT": base["root"],
            "RCLONE_BIN": base["fake"],
            "REMOTE_ROOT": REMOTE_ROOT,
            "QUANT_GDRIVE_LOCK": base["lock"],
            "LOG_DIR": base["log_dir"],
            "BACKUP_TODAY_UTC": TODAY,
            "VERSION_RETENTION_DAYS": "30",
            "LOCK_WAIT_SEC": "10",
            "FAKE_RECORD": base["record"],
        }
    )
    if extra_env:
        env.update(extra_env)
    return subprocess.run(  # noqa: S603 - fixed argv: repo backup script in hermetic test
        ["/bin/bash", str(SCRIPT)], capture_output=True, text=True, env=env, timeout=60
    )


def _calls(base: dict[str, str]) -> list[list[str]]:
    text = Path(base["record"]).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _log_text(base: dict[str, str], today: str = TODAY) -> str:
    return (Path(base["log_dir"]) / f"crypto-pilot-backup-{today}.log").read_text(encoding="utf-8")


def test_data_copy_is_copy_only_filtered_and_versioned(tmp_path: Path) -> None:
    base = _setup_base(tmp_path, with_orders=False)
    result = _run(base, tmp_path, {"FAKE_LSF_RC": "3"})
    assert result.returncode == 0
    calls = _calls(base)
    copies = [c for c in calls if c and c[0] == "copy"]
    assert len(copies) == 1
    data_call = copies[0]
    assert data_call[1] == f"{base['root']}/data"
    assert data_call[2] == f"{REMOTE_ROOT}/data"
    assert "--filter-from" in data_call
    assert f"{base['root']}/deploy/crypto-pilot.rclone-filter" in data_call
    assert f"{REMOTE_ROOT}/_versions/{TODAY}/data" in data_call
    for call in calls:
        assert call[0] != "sync"
        assert call[0] != "move"
        assert call[0] != "delete"
        for token in call:
            assert token not in ("--min-age", "--max-age")
            assert not token.startswith("--delete")
        if call[0] == "purge":
            assert any("_versions/" in token for token in call)


def test_order_events_copied_when_present(tmp_path: Path) -> None:
    base = _setup_base(tmp_path, with_orders=True)
    result = _run(base, tmp_path, {"FAKE_LSF_RC": "3"})
    assert result.returncode == 0
    calls = _calls(base)
    orders_calls = [c for c in calls if c and c[0] == "copy" and any("logs/live/orders" in t for t in c)]
    assert len(orders_calls) == 1
    call = orders_calls[0]
    assert f"{REMOTE_ROOT}/logs/live/orders" in call
    assert "--include" in call
    assert "*.jsonl" in call
    assert f"{REMOTE_ROOT}/_versions/{TODAY}/logs/live/orders" in call


def test_missing_orders_dir_skipped_not_failed(tmp_path: Path) -> None:
    base = _setup_base(tmp_path, with_orders=False)
    result = _run(base, tmp_path, {"FAKE_LSF_RC": "3"})
    assert result.returncode == 0
    calls = _calls(base)
    assert not [c for c in calls if c and c[0] == "copy" and any("logs/live/orders" in t for t in c)]
    log = _log_text(base)
    assert "step=orders status=skipped" in log


def test_version_prune_keys_on_folder_date_only(tmp_path: Path) -> None:
    base = _setup_base(tmp_path, with_orders=False)
    lsf_output = "2026-08-23/\n2026-08-24/\n2026-09-22/\nnotes/\n2026-13-01/\n"
    result = _run(base, tmp_path, {"FAKE_LSF_RC": "0", "FAKE_LSF_OUTPUT": lsf_output})
    assert result.returncode == 0
    calls = _calls(base)
    purges = [c for c in calls if c and c[0] == "purge"]
    assert purges == [["purge", f"{REMOTE_ROOT}/_versions/2026-08-23"]]


def test_missing_versions_root_not_error(tmp_path: Path) -> None:
    base = _setup_base(tmp_path, with_orders=False)
    result = _run(base, tmp_path, {"FAKE_LSF_RC": "3"})
    assert result.returncode == 0
    calls = _calls(base)
    assert not [c for c in calls if c and c[0] == "purge"]
    log = _log_text(base)
    assert "step=prune status=ok" in log


def test_failed_step_does_not_stop_later_steps(tmp_path: Path) -> None:
    base = _setup_base(tmp_path, with_orders=True)
    result = _run(
        base,
        tmp_path,
        {"FAKE_DATA_RC": "1", "FAKE_LSF_RC": "0", "FAKE_LSF_OUTPUT": ""},
    )
    assert result.returncode == 1
    calls = _calls(base)
    assert any(c and c[0] == "copy" and any("logs/live/orders" in t for t in c) for c in calls)
    assert any(c and c[0] == "lsf" for c in calls)
    log = _log_text(base)
    assert "step=data status=failed" in log
    assert "status=failed" in log.splitlines()[-1]


def test_held_lock_blocks_all_drive_calls(tmp_path: Path) -> None:
    base = _setup_base(tmp_path, with_orders=True)
    holder = subprocess.Popen(  # noqa: S603 - fixed argv: flock + lock path in hermetic test
        ["/usr/bin/flock", base["lock"], "sleep", "10"]
    )
    try:
        time.sleep(0.5)
        result = _run(base, tmp_path, {"LOCK_WAIT_SEC": "1"})
        assert result.returncode == 75
        assert Path(base["record"]).read_text(encoding="utf-8").strip() == ""
    finally:
        holder.terminate()
        holder.wait(timeout=10)


def test_secrets_never_leave_host(tmp_path: Path) -> None:
    base = _setup_base(tmp_path, with_orders=False)
    data = Path(base["root"]) / "data"
    (data / ".env").write_text("K=1\n", encoding="utf-8")
    (data / "a.key").write_text("K=1\n", encoding="utf-8")
    (data / "x_key.txt").write_text("K=1\n", encoding="utf-8")
    result = _run(base, tmp_path, {"FAKE_LSF_RC": "3"})
    assert result.returncode == 0
    calls = _calls(base)
    data_calls = [c for c in calls if c and c[0] == "copy" and f"{REMOTE_ROOT}/data" in c]
    assert len(data_calls) == 1
    argv = " ".join(data_calls[0])
    assert ".env*" in argv
    assert "*.key" in argv
    assert "*_key.txt" in argv
