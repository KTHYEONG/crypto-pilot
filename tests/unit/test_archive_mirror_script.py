"""Hermetic invariant guards for the archive mirror night runner (fake docker and rclone on PATH)."""

from __future__ import annotations

import fcntl
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy" / "archive-mirror" / "run.sh"
REMOTE = "testremote:archive"
DESTRUCTIVE = {"delete", "deletefile", "purge", "sync", "move", "moveto"}

FAKE_DOCKER = """#!{python}
import fcntl, json, os, sys
args = sys.argv[1:]
record = os.environ["FAKE_RECORD"]
def note(entry):
    with open(record, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\\n")
note({{"bin": "docker", "argv": args}})
with open(os.environ["QUANT_GDRIVE_LOCK"], "a") as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        note({{"violation": "drive_lock_held_during_container_run"}})
    else:
        fcntl.flock(lock, fcntl.LOCK_UN)
sub = args[args.index("mirror") + 1]
if sub == "format-doc":
    print("FORMAT")
    sys.exit(0)
if sub == "build":
    if os.environ.get("FAKE_BUILD_SLEEP"):
        import time
        time.sleep(float(os.environ["FAKE_BUILD_SLEEP"]))
    counter = os.path.join(os.environ["FAKE_STATE"], "build_count")
    count = int(open(counter).read()) if os.path.exists(counter) else 0
    with open(counter, "w") as handle:
        handle.write(str(count + 1))
    steps = os.environ.get("FAKE_BUILD_SEQ", "0|0|false").split(";")
    rc, built, more = steps[min(count, len(steps) - 1)].split("|")
    report = {{"built_objects": int(built), "built_sources": 0, "verbatim_sources": 0, "failed_sources": 0,
              "pending_sources": 0, "outbox_bytes": 0, "more": more == "true", "capacity_blocked": None,
              "anomalies": [], "listing_failed": []}}
    print(json.dumps(report))
    sys.exit(int(rc))
if sub == "confirm":
    given = args[args.index("--remote-listing") + 1]
    host = given.replace("/app/data/archive_mirror", os.environ["CRYPTO_PILOT_ROOT"] + "/data/archive_mirror")
    with open(host, encoding="utf-8") as handle:
        note({{"confirm_listing": handle.read()}})
    sys.exit(int(os.environ.get("FAKE_CONFIRM_RC", "0")))
if sub == "status":
    sys.exit(int(os.environ.get("FAKE_STATUS_RC", "0")))
sys.exit(2)
"""

FAKE_RCLONE = """#!{python}
import fcntl, json, os, sys
args = sys.argv[1:]
record = os.environ["FAKE_RECORD"]
entry = {{"bin": "rclone", "argv": args}}
if "--files-from" in args:
    with open(args[args.index("--files-from") + 1], encoding="utf-8") as handle:
        entry["files_from"] = handle.read().split()
with open(record, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(entry) + "\\n")
sub = args[0]
joined = " ".join(args)
rules = [item.split("=", 1) for item in os.environ.get("FAKE_RCLONE_FAIL", "").split(";") if item]
rc = next((int(code) for needle, code in rules if needle in joined), 0)
if sub in ("copy", "copyto") and args[2].startswith("testremote:"):
    with open(os.environ["QUANT_GDRIVE_LOCK"], "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            with open(record, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({{"violation": "unlocked_write"}}) + "\\n")
            sys.exit(98)
if sub == "size":
    sys.stdout.write(os.environ.get("FAKE_SIZE_JSON", '{{"count": 0, "bytes": 1000}}'))
elif sub == "about":
    sys.stdout.write(os.environ.get("FAKE_ABOUT_JSON", '{{"total": 2000, "used": 1000, "free": 1000}}'))
elif sub == "lsjson":
    sys.stdout.write(os.environ.get("FAKE_LSJSON", "[]"))
sys.exit(rc)
"""


def _write_exe(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def harness(tmp_path: Path) -> dict[str, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_exe(bin_dir / "docker", FAKE_DOCKER.format(python=sys.executable))
    _write_exe(bin_dir / "rclone", FAKE_RCLONE.format(python=sys.executable))
    state = tmp_path / "state"
    state.mkdir()
    root = tmp_path / "root"
    (root / "data" / "archive_mirror").mkdir(parents=True)
    return {"tmp": tmp_path, "bin": bin_dir, "state": state, "root": root, "lock": tmp_path / "drive.lock"}


def _env(harness: dict[str, Path], **overrides: str) -> dict[str, str]:
    tmp = harness["tmp"]
    env = {
        "PATH": f"{harness['bin']}:/usr/local/bin:/usr/bin:/bin",
        "HOME": str(tmp),
        "CRYPTO_PILOT_ROOT": str(harness["root"]),
        "LOG_DIR": str(tmp / "logs"),
        "RCLONE_BIN": str(harness["bin"] / "rclone"),
        "MIRROR_REMOTE": REMOTE,
        "DRIVE_ABOUT_REMOTE": "testremote:",
        "QUANT_GDRIVE_LOCK": str(harness["lock"]),
        "LOCK_WAIT_SEC": "1",
        "MIRROR_DEADLINE_UTC": "23:59",
        "FAKE_RECORD": str(tmp / "record.jsonl"),
        "FAKE_STATE": str(harness["state"]),
    }
    env.update(overrides)
    return env


def _run(harness: dict[str, Path], **overrides: str) -> tuple[subprocess.CompletedProcess[str], list[dict]]:
    result = subprocess.run(  # noqa: S603 - fixed argv: repo mirror script in hermetic test
        ["/bin/bash", str(SCRIPT)],
        env=_env(harness, **overrides),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    record = harness["tmp"] / "record.jsonl"
    calls = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()] if record.exists() else []
    return result, calls


def _argv(calls: list[dict], binary: str) -> list[list[str]]:
    return [c["argv"] for c in calls if c.get("bin") == binary]


def _builds(calls: list[dict]) -> list[list[str]]:
    return [a for a in _argv(calls, "docker") if "build" in a]


PAIRS = ("metrics/2024-01/ALL-aaaaaaaaaaaa.parquet", "verbatim/bookDepth/2024-01/B-USDT-1-bbbbbbbbbbbb.zip")
ORPHAN = "metrics/2024-02/ALL-cccccccccccc.parquet"


def _stage_outbox(root: Path) -> None:
    outbox = root / "data/archive_mirror/outbox"
    for relative in (*PAIRS, ORPHAN):
        target = outbox / "objects" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
    for relative in PAIRS:
        sidecar = outbox / "index" / Path(relative).with_suffix(".json.zst")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_bytes(b"s")


def test_only_complete_pairs_are_uploaded_and_listed(harness: dict[str, Path]) -> None:
    _stage_outbox(harness["root"])
    _, calls = _run(harness, FAKE_BUILD_SEQ="0|0|false")
    sidecars = sorted(Path(r).with_suffix(".json.zst").as_posix() for r in PAIRS)
    sent = {c["argv"][1].rsplit("/", 1)[-1]: sorted(c["files_from"]) for c in calls if "files_from" in c and c["argv"][0] == "copy"}
    assert sent == {"objects": sorted(PAIRS), "index": sidecars}
    listed = next(c["files_from"] for c in calls if "files_from" in c and c["argv"][0] == "lsjson")
    assert sorted(listed) == sorted([f"objects/{r}" for r in PAIRS] + [f"index/{s}" for s in sidecars])


def test_orphan_object_alone_is_never_uploaded(harness: dict[str, Path]) -> None:
    target = harness["root"] / "data/archive_mirror/outbox/objects" / ORPHAN
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    _, calls = _run(harness, FAKE_BUILD_SEQ="0|0|false")
    assert not [a for a in _argv(calls, "rclone") if a[0] == "copy" and "outbox" in a[1]]


def test_confirm_reads_the_listing_just_fetched(harness: dict[str, Path]) -> None:
    _stage_outbox(harness["root"])
    fetched = '[{"Path":"objects/fresh","Size":1}]'
    _, calls = _run(harness, FAKE_BUILD_SEQ="0|0|false", FAKE_LSJSON=fetched)
    assert next(c["confirm_listing"] for c in calls if "confirm_listing" in c) == fetched


def test_index_sync_failure_builds_nothing(harness: dict[str, Path]) -> None:
    result, calls = _run(harness, FAKE_RCLONE_FAIL="archive/index=1")
    assert result.returncode != 0
    assert _builds(calls) == []


def test_index_absent_directory_is_fine(harness: dict[str, Path]) -> None:
    result, calls = _run(harness, FAKE_RCLONE_FAIL="archive/index=3", FAKE_BUILD_SEQ="0|0|false")
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(_builds(calls)) == 1


@pytest.mark.parametrize("override", [{"FAKE_RCLONE_FAIL": "about --json=1"}, {"FAKE_SIZE_JSON": "not json"}])
def test_capacity_probe_failure_fails_closed(harness: dict[str, Path], override: dict[str, str]) -> None:
    result, calls = _run(harness, **override)
    assert result.returncode != 0
    assert _builds(calls) == []


def test_absent_remote_size_counts_as_zero(harness: dict[str, Path]) -> None:
    _, calls = _run(harness, FAKE_RCLONE_FAIL="size --json=3", FAKE_BUILD_SEQ="0|0|false")
    (build,) = _builds(calls)
    assert build[build.index("--remote-bytes") + 1] == "0"


def test_objects_upload_before_sidecars_with_immutable(harness: dict[str, Path]) -> None:
    _stage_outbox(harness["root"])
    _, calls = _run(harness)
    copies = _argv(calls, "rclone")
    objects = next(i for i, a in enumerate(copies) if a[0] == "copy" and a[1].endswith("outbox/objects"))
    sidecars = next(i for i, a in enumerate(copies) if a[0] == "copy" and a[1].endswith("outbox/index"))
    assert objects < sidecars
    assert "--immutable" in copies[objects]
    assert "--immutable" in copies[sidecars]


def test_upload_failure_never_confirms_and_drops_stale_listing(harness: dict[str, Path]) -> None:
    _stage_outbox(harness["root"])
    status_dir = harness["root"] / "data/archive_mirror/_status"
    status_dir.mkdir(parents=True)
    stale = status_dir / "remote_listing.json"
    stale.write_text("[]", encoding="utf-8")
    result, calls = _run(harness, FAKE_RCLONE_FAIL="outbox/objects=1")
    assert result.returncode == 1
    assert not [a for a in _argv(calls, "docker") if "confirm" in a]
    assert not [a for a in _argv(calls, "rclone") if a[0] == "lsjson"]
    assert not stale.exists()


def test_drive_lock_not_held_during_container_run(harness: dict[str, Path]) -> None:
    _stage_outbox(harness["root"])
    result, calls = _run(harness, FAKE_BUILD_SEQ="0|1|false")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not [c for c in calls if "violation" in c]


def test_format_doc_and_confirm_use_remote_listing(harness: dict[str, Path]) -> None:
    _stage_outbox(harness["root"])
    result, calls = _run(harness, FAKE_BUILD_SEQ="0|0|false")
    assert result.returncode == 0, result.stdout + result.stderr
    copyto = [a for a in _argv(calls, "rclone") if a[0] == "copyto"]
    assert len(copyto) == 1
    assert re.fullmatch(rf"{re.escape(REMOTE)}/FORMAT-v1-[0-9a-f]{{12}}\.md", copyto[0][2])
    assert "--immutable" in copyto[0]
    confirm = [a for a in _argv(calls, "docker") if "confirm" in a]
    assert confirm
    for argv in confirm:
        assert argv[argv.index("--remote-listing") + 1] == "/app/data/archive_mirror/_status/remote_listing.json"


@pytest.mark.parametrize(
    ("build_seq", "expected_builds"),
    [("0|5|false", 1), ("0|0|true", 1), ("4|0|true", 1), ("2|0|true", 1)],
)
def test_loop_stops_on_terminal_result(harness: dict[str, Path], build_seq: str, expected_builds: int) -> None:
    _, calls = _run(harness, FAKE_BUILD_SEQ=build_seq)
    assert len(_builds(calls)) == expected_builds


@pytest.mark.parametrize(("build_seq", "expected_exit"), [("5|3|true;0|2|false", 5), ("3|3|true;0|2|false", 3)])
def test_anomaly_does_not_stop_backfill(harness: dict[str, Path], build_seq: str, expected_exit: int) -> None:
    result, calls = _run(harness, FAKE_BUILD_SEQ=build_seq)
    assert len(_builds(calls)) == 2
    assert result.returncode == expected_exit


@pytest.mark.parametrize(
    ("build_seq", "status_rc", "confirm_rc", "stage", "expected"),
    [
        ("4|0|true;0|1|false", "6", "0", False, 4),
        ("5|1|true;0|1|false", "6", "0", False, 5),
        ("3|1|true;0|1|false", "6", "0", False, 6),
        ("3|1|true;0|1|false", "0", "0", False, 3),
        ("0|1|false", "0", "3", True, 1),
        ("0|1|false", "6", "3", True, 6),
    ],
)
def test_exit_priority(
    harness: dict[str, Path], build_seq: str, status_rc: str, confirm_rc: str, stage: bool, expected: int
) -> None:
    if stage:
        _stage_outbox(harness["root"])
    result, _ = _run(
        harness, FAKE_BUILD_SEQ=build_seq, FAKE_STATUS_RC=status_rc, FAKE_CONFIRM_RC=confirm_rc
    )
    assert result.returncode == expected, result.stdout + result.stderr


def test_busy_run_lock_exits_75(harness: dict[str, Path]) -> None:
    status_dir = harness["root"] / "data/archive_mirror/_status"
    status_dir.mkdir(parents=True)
    with open(status_dir / "run.lock", "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result, calls = _run(harness)
    assert result.returncode == 75
    assert not _argv(calls, "docker")


def test_passed_deadline_builds_nothing(harness: dict[str, Path]) -> None:
    result, calls = _run(harness, MIRROR_DEADLINE_UTC="00:00")
    assert result.returncode == 0
    assert _builds(calls) == []


def test_never_deletes_remote(harness: dict[str, Path]) -> None:
    _stage_outbox(harness["root"])
    _, calls = _run(harness, FAKE_BUILD_SEQ="5|3|true;0|2|false")
    assert not [a for a in _argv(calls, "rclone") if a[0] in DESTRUCTIVE]


@pytest.mark.parametrize("now", ["22:55", "23:30", "00:00", "00:14", "06:10", "06:44"])
def test_quiet_window_does_nothing(harness: dict[str, Path], now: str) -> None:
    result, calls = _run(harness, MIRROR_DEADLINE_UTC="", MIRROR_NOW_UTC=now)
    assert result.returncode == 0
    assert calls == []


@pytest.mark.parametrize(
    ("now", "end"), [("00:15", "06:10"), ("06:09", "06:10"), ("06:45", "22:55"), ("13:40", "22:55"), ("22:54", "22:55")]
)
def test_session_ends_at_the_next_quiet_window(harness: dict[str, Path], now: str, end: str) -> None:
    result, calls = _run(harness, MIRROR_DEADLINE_UTC="", MIRROR_NOW_UTC=now, FAKE_BUILD_SEQ="0|0|false")
    assert result.returncode == 0, result.stdout + result.stderr
    (build,) = _builds(calls)
    assert build[build.index("--deadline") + 1] == end


@pytest.mark.parametrize(
    "override",
    [
        {"MIRROR_DEADLINE_UTC": "", "MIRROR_QUIET_UTC": "6-7"},
        {"MIRROR_DEADLINE_UTC": "25:00"},
        {"MIRROR_DEADLINE_UTC": "", "MIRROR_NOW_UTC": "noon"},
    ],
)
def test_malformed_window_fails_closed(harness: dict[str, Path], override: dict[str, str]) -> None:
    result, calls = _run(harness, **override)
    assert result.returncode == 1
    assert calls == []


def test_busy_drive_lock_defers_without_failing_or_confirming(harness: dict[str, Path]) -> None:
    _stage_outbox(harness["root"])
    with open(harness["lock"], "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result, calls = _run(harness, LOCK_WAIT_SEC="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not [a for a in _argv(calls, "rclone") if a[0] in ("copy", "copyto") and a[2].startswith(REMOTE)]
    assert not [a for a in _argv(calls, "docker") if "confirm" in a]
    assert _builds(calls) == []


def test_upload_cut_at_max_duration_is_deferred(harness: dict[str, Path]) -> None:
    _stage_outbox(harness["root"])
    result, calls = _run(harness, FAKE_RCLONE_FAIL="outbox/objects=10")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not [a for a in _argv(calls, "docker") if "confirm" in a]
    upload = next(a for a in _argv(calls, "rclone") if a[0] == "copy" and a[1].endswith("outbox/objects"))
    assert upload[upload.index("--max-duration") + 1].endswith("s")


def test_overrunning_build_is_terminated_and_deferred(harness: dict[str, Path]) -> None:
    started = time.monotonic()
    result, calls = _run(
        harness, MIRROR_NOW_UTC="10:00", MIRROR_DEADLINE_UTC="10:01", MIRROR_OVERRUN_GRACE_SEC="-57",
        FAKE_BUILD_SLEEP="60",
    )
    assert time.monotonic() - started < 40
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(_builds(calls)) == 1


def test_one_upload_never_holds_the_drive_lock_past_the_cap(harness: dict[str, Path]) -> None:
    _stage_outbox(harness["root"])
    _, calls = _run(harness, MIRROR_NOW_UTC="07:00", MIRROR_DEADLINE_UTC="22:55")
    uploads = [a for a in _argv(calls, "rclone") if a[0] == "copy" and "outbox" in a[1]]
    assert uploads
    assert {a[a.index("--max-duration") + 1] for a in uploads} == {"1800s"}


def test_cut_index_sync_builds_nothing(harness: dict[str, Path]) -> None:
    result, calls = _run(harness, FAKE_RCLONE_FAIL="archive/index=10")
    assert result.returncode == 0
    assert _builds(calls) == []
