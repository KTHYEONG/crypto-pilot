"""Invariant guards for the blue/green capture handover helper."""

from __future__ import annotations

import ast
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.application.ops.capture_handover import (
    SlotObservation,
    decide_capture_action,
    evaluate_ready,
    main,
)

NOW = datetime(2026, 9, 26, 12, 0, 0, tzinfo=UTC)
STARTED = NOW - timedelta(minutes=5)
CONTAINER_START = STARTED - timedelta(seconds=10)


def _obs(slot: str, *, running: bool = True, fp: str | None = "sha256:img", cfg: bool = True, ready: bool = True) -> SlotObservation:
    return SlotObservation(slot=slot, running=running, fingerprint=fp, config_hash_matches=cfg, heartbeat_ready=ready)  # type: ignore[arg-type]


def _ready_payload(*, started_at: datetime = STARTED, ts: datetime = NOW, stopped: str | None = None, flush: int = 0) -> dict:
    def iso(value: datetime) -> str:
        return value.isoformat().replace("+00:00", "Z")

    return {
        "started_at": iso(started_at),
        "stopped_at": stopped,
        "ts": iso(ts),
        "rest": {"book_ticker": {"first_ok_at": iso(started_at)}},
        "ws": {"first_frame_at": iso(started_at)},
        "flush_failures": flush,
    }


def test_unchanged_ready_slot_is_kept() -> None:
    """A current ready slot needs no deploy action."""
    decision = decide_capture_action([_obs("blue"), _obs("green", running=False, fp=None, ready=False)], image_fingerprint="sha256:img", legacy_running=False)
    assert (decision.action, decision.reason) == ("keep", "unchanged")
    assert decision.new_slot is None
    assert decision.old_slot is None
    assert decision.retire_legacy is False


def test_changed_fingerprint_hands_over_to_idle_slot() -> None:
    """A stale running slot hands over to the idle slot."""
    decision = decide_capture_action([_obs("blue", fp="sha256:old"), _obs("green", running=False, fp=None, ready=False)], image_fingerprint="sha256:img", legacy_running=False)
    assert (decision.action, decision.new_slot, decision.old_slot, decision.reason) == ("handover", "green", "blue", "fingerprint_changed")


def test_green_active_hands_over_back_to_blue() -> None:
    """Handover direction follows whichever slot is running."""
    decision = decide_capture_action([_obs("blue", running=False, fp=None, ready=False), _obs("green", fp="sha256:old")], image_fingerprint="sha256:img", legacy_running=False)
    assert (decision.action, decision.new_slot, decision.old_slot) == ("handover", "blue", "green")


def test_unreadable_image_fingerprint_never_keeps() -> None:
    """Without a trusted image fingerprint the running slot cannot be proven current."""
    decision = decide_capture_action([_obs("blue"), _obs("green", running=False, fp=None, ready=False)], image_fingerprint=None, legacy_running=False)
    assert (decision.action, decision.reason) == ("handover", "image_fingerprint_unreadable")


def test_unreadable_slot_fingerprint_hands_over() -> None:
    """An unreadable running fingerprint cannot be proven current."""
    decision = decide_capture_action([_obs("blue", fp=None), _obs("green", running=False, fp=None, ready=False)], image_fingerprint="sha256:img", legacy_running=False)
    assert decision.reason == "fingerprint_unreadable"


def test_running_slot_not_ready_is_replaced() -> None:
    """A current but not-yet-ready slot hands over with reason not_ready."""
    decision = decide_capture_action([_obs("blue", ready=False), _obs("green", running=False, fp=None, ready=False)], image_fingerprint="sha256:img", legacy_running=False)
    assert (decision.action, decision.reason) == ("handover", "not_ready")


def test_config_drift_hands_over() -> None:
    """Equal fingerprints with a config-hash mismatch still hands over."""
    decision = decide_capture_action([_obs("blue", cfg=False), _obs("green", running=False, fp=None, ready=False)], image_fingerprint="sha256:img", legacy_running=False)
    assert decision.reason == "compose_config_changed"


def test_both_running_reconciles_to_current_ready_slot() -> None:
    """A crashed handover leaves both slots running; reconcile keeps the current ready one."""
    decision = decide_capture_action([_obs("blue", fp="sha256:old"), _obs("green")], image_fingerprint="sha256:img", legacy_running=False)
    assert (decision.action, decision.new_slot, decision.old_slot) == ("reconcile", "green", "blue")


def test_both_current_ready_reconciles_to_blue() -> None:
    """Two current ready slots cannot be distinguished; reconcile keeps blue."""
    decision = decide_capture_action([_obs("blue"), _obs("green")], image_fingerprint="sha256:img", legacy_running=False)
    assert (decision.action, decision.new_slot, decision.old_slot, decision.reason) == ("reconcile", "blue", "green", "both_running")


def test_both_running_without_current_keeps_ready_slot() -> None:
    """With no current slot, reconcile keeps the only READY one."""
    decision = decide_capture_action([_obs("blue", fp="sha256:old"), _obs("green", fp="sha256:older")], image_fingerprint="sha256:img", legacy_running=False)
    assert (decision.action, decision.new_slot, decision.old_slot) == ("reconcile", "blue", "green")


def test_both_running_with_nothing_ready_keeps_blue() -> None:
    """With nothing READY, reconcile keeps blue deterministically."""
    decision = decide_capture_action([_obs("blue", fp="sha256:old", ready=False), _obs("green", fp="sha256:older", ready=False)], image_fingerprint="sha256:img", legacy_running=False)
    assert (decision.action, decision.new_slot, decision.old_slot) == ("reconcile", "blue", "green")


def test_decide_rejects_observations_without_both_slots() -> None:
    """Anything but exactly one observation per slot is a programming error."""
    with pytest.raises(ValueError, match="one observation per slot"):
        decide_capture_action([_obs("blue"), _obs("blue")], image_fingerprint="sha256:img", legacy_running=False)
    with pytest.raises(ValueError, match="one observation per slot"):
        decide_capture_action([_obs("blue")], image_fingerprint="sha256:img", legacy_running=False)


def test_ready_accepts_docker_nanosecond_timestamps() -> None:
    """Docker StartedAt nanoseconds truncate to microseconds without losing READY."""
    started = "2026-09-26T11:50:10.123456789+00:00"
    container_start = "2026-09-26T11:50:00.987654321+00:00"
    payload = _ready_payload()
    payload["started_at"] = started
    payload["rest"] = {"book_ticker": {"first_ok_at": started}}
    payload["ws"] = {"first_frame_at": started}
    from datetime import datetime

    assert evaluate_ready(payload, container_started_at=datetime.fromisoformat(container_start), now=NOW, stale_s=90000) == "ready"


def test_cli_rejects_malformed_arguments() -> None:
    """Malformed CLI input exits 2 without deciding."""
    assert main([]) == 2
    assert main(["decide", "--image-fp", "sha256:img", "--slot", "blue:1:sha256:old:1:1"]) == 2
    assert main(["decide", "--image-fp", "sha256:img", "--slot", "red:1:fp:1:1", "--slot", "green:0:-:0:0"]) == 2
    assert main(["decide", "--image-fp", "sha256:img", "--slot", "blue:1", "--slot", "green:0:-:0:0"]) == 2
    assert main(["decide", "--image-fp", "sha256:img", "--slot", "blue:bogus:fp:1:1", "--slot", "green:0:-:0:0"]) == 2
    assert main(["ready", "--heartbeat", "x", "--container-started-at", "not-a-time", "--stale-s", "30"]) == 2
    assert main(["decide", "--bogus-flag"]) == 2
    assert main(["ready", "--heartbeat", "x"]) == 2
    assert main(["ready", "--heartbeat", "x", "--container-started-at", CONTAINER_START.isoformat(), "--now", "bogus", "--stale-s", "30"]) == 2
    assert main(["ready", "--heartbeat", "x", "--container-started-at", CONTAINER_START.isoformat(), "--stale-s", "bogus"]) == 2


def test_cli_ready_rejects_non_dict_payload(tmp_path: Path) -> None:
    """A heartbeat file without a JSON object waits without raising."""
    target = tmp_path / "list.json"
    target.write_text("[1, 2]", encoding="utf-8")
    assert main(["ready", "--heartbeat", str(target), "--container-started-at", CONTAINER_START.isoformat(), "--stale-s", "30"]) == 10


def test_cli_decide_without_image_fingerprint_hands_over(capsys: pytest.CaptureFixture[str]) -> None:
    """An absent or placeholder image fingerprint never keeps the running slot."""
    assert main(["decide", "--slot", "blue:1:sha256:old:1:1", "--slot", "green:0:-:0:0"]) == 0
    assert "reason=image_fingerprint_unreadable" in capsys.readouterr().out
    assert main(["decide", "--image-fp", "-", "--slot", "blue:1:sha256:old:1:1", "--slot", "green:0:-:0:0"]) == 0
    assert "reason=image_fingerprint_unreadable" in capsys.readouterr().out


def test_first_deploy_migrates_legacy_recorder() -> None:
    """No slot running with a legacy recorder starts blue and retires legacy."""
    decision = decide_capture_action([_obs("blue", running=False, fp=None, ready=False), _obs("green", running=False, fp=None, ready=False)], image_fingerprint="sha256:img", legacy_running=True)
    assert (decision.action, decision.new_slot, decision.reason) == ("start", "blue", "legacy_migration")
    assert decision.retire_legacy is True


def test_empty_host_starts_blue() -> None:
    """Nothing running starts the blue slot."""
    decision = decide_capture_action([_obs("blue", running=False, fp=None, ready=False), _obs("green", running=False, fp=None, ready=False)], image_fingerprint="sha256:img", legacy_running=False)
    assert (decision.action, decision.new_slot, decision.reason) == ("start", "blue", "not_running")


def test_ready_requires_both_streams_after_this_start() -> None:
    """A heartbeat with only REST ready is still waiting."""
    payload = _ready_payload()
    payload["ws"] = {"first_frame_at": None}
    assert evaluate_ready(payload, container_started_at=CONTAINER_START, now=NOW, stale_s=30) == "waiting"


def test_stale_file_from_previous_run_is_not_ready() -> None:
    """A fully-ready file from an earlier container start is not READY."""
    payload = _ready_payload()
    assert evaluate_ready(payload, container_started_at=STARTED + timedelta(seconds=1), now=NOW, stale_s=30) == "waiting"


def test_stale_heartbeat_timestamp_is_not_ready() -> None:
    """A heartbeat older than stale_s is not READY."""
    payload = _ready_payload(ts=NOW - timedelta(seconds=60))
    assert evaluate_ready(payload, container_started_at=CONTAINER_START, now=NOW, stale_s=30) == "waiting"


def test_flush_failures_block_readiness() -> None:
    """Any flush failure blocks READY."""
    payload = _ready_payload(flush=1)
    assert evaluate_ready(payload, container_started_at=CONTAINER_START, now=NOW, stale_s=30) == "waiting"


def test_stopped_heartbeat_is_not_ready() -> None:
    """A heartbeat with stopped_at set belongs to a dead container."""
    payload = _ready_payload(stopped=NOW.isoformat().replace("+00:00", "Z"))
    assert evaluate_ready(payload, container_started_at=CONTAINER_START, now=NOW, stale_s=30) == "waiting"


def test_missing_stream_sections_wait() -> None:
    """Missing book_ticker or ws sections wait without raising."""
    no_book = _ready_payload()
    no_book["rest"] = {}
    assert evaluate_ready(no_book, container_started_at=CONTAINER_START, now=NOW, stale_s=30) == "waiting"
    null_ok = _ready_payload()
    null_ok["rest"] = {"book_ticker": {"first_ok_at": None}}
    assert evaluate_ready(null_ok, container_started_at=CONTAINER_START, now=NOW, stale_s=30) == "waiting"
    no_ws = _ready_payload()
    no_ws["ws"] = "not-a-dict"
    assert evaluate_ready(no_ws, container_started_at=CONTAINER_START, now=NOW, stale_s=30) == "waiting"


def test_naive_container_start_waits_closed() -> None:
    """An unparseable container start can never prove freshness; wait without raising."""
    from datetime import datetime

    payload = _ready_payload()
    assert evaluate_ready(payload, container_started_at=datetime(2026, 9, 26, 11, 50), now=NOW, stale_s=30) == "waiting"


def test_missing_or_malformed_file_waits() -> None:
    """Absent or malformed heartbeats wait without raising."""
    naive = _ready_payload()
    naive["started_at"] = "2026-09-26T12:00:00"
    assert evaluate_ready(None, container_started_at=CONTAINER_START, now=NOW, stale_s=30) == "waiting"
    assert evaluate_ready({}, container_started_at=CONTAINER_START, now=NOW, stale_s=30) == "waiting"
    assert evaluate_ready(naive, container_started_at=CONTAINER_START, now=NOW, stale_s=30) == "waiting"
    bad_rest = _ready_payload()
    bad_rest["rest"] = "not-a-dict"
    assert evaluate_ready(bad_rest, container_started_at=CONTAINER_START, now=NOW, stale_s=30) == "waiting"


def test_fully_ready_heartbeat_is_ready_and_cli_matches(tmp_path: Path) -> None:
    """All conditions met evaluates ready, and the CLI exit codes match."""
    payload = _ready_payload()
    assert evaluate_ready(payload, container_started_at=CONTAINER_START, now=NOW, stale_s=30) == "ready"
    target = tmp_path / "capture_blue.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    assert main(["ready", "--heartbeat", str(target), "--container-started-at", CONTAINER_START.isoformat(), "--now", NOW.isoformat(), "--stale-s", "30"]) == 0
    assert main(["ready", "--heartbeat", str(tmp_path / "missing.json"), "--container-started-at", CONTAINER_START.isoformat(), "--now", NOW.isoformat(), "--stale-s", "30"]) == 10


def test_cli_decide_prints_one_parseable_line(capsys: pytest.CaptureFixture[str]) -> None:
    """The decide CLI prints a single machine-readable line."""
    assert main(["decide", "--image-fp", "sha256:img", "--slot", "blue:1:sha256:old:1:1", "--slot", "green:0:-:0:0"]) == 0
    assert capsys.readouterr().out.strip() == "action=handover new=green old=blue legacy=0 reason=fingerprint_changed"


def test_helper_is_stdlib_only_and_312_safe() -> None:
    """The host helper imports only stdlib modules and never src.*."""
    source = Path("src/application/ops/capture_handover.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in sys.stdlib_module_names
                assert not alias.name.startswith("src")
        elif isinstance(node, ast.ImportFrom):
            assert node.module is not None
            assert node.module.split(".")[0] in sys.stdlib_module_names
            assert not node.module.startswith("src")
