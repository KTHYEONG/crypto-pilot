"""Decide raw-first capture deploy actions and evaluate capture READY from the host."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

Slot = Literal["blue", "green"]
LEGACY_CONTAINER: str = "market-recorder"

EXIT_OK: int = 0
EXIT_NOT_READY: int = 10
EXIT_USAGE: int = 2


@dataclass(frozen=True, slots=True)
class SlotObservation:
    """What the deploy script observed about one capture container.

    Attributes:
        slot: Slot name.
        running: ``docker inspect .State.Running`` is true.
        fingerprint: ``/app/.capture_fingerprint`` of the running container, or ``None`` when unreadable.
        config_hash_matches: The compose config-hash label equals ``compose config --hash`` for the service.
        heartbeat_ready: The slot's heartbeat evaluates READY now (see ``evaluate_ready``).
    """

    slot: Slot
    running: bool
    fingerprint: str | None
    config_hash_matches: bool
    heartbeat_ready: bool


@dataclass(frozen=True, slots=True)
class CaptureDecision:
    """Deploy action for the capture tier.

    Attributes:
        action: ``keep`` (running slot is current), ``start`` (nothing usable runs; start ``new_slot``),
            ``handover`` (start ``new_slot``, wait READY, then retire ``old_slot``),
            ``reconcile`` (both slots run; retire ``old_slot`` and keep ``new_slot`` without a start).
        new_slot: Slot to start or keep; ``None`` only for ``keep``.
        old_slot: Slot to retire; ``None`` when nothing must be retired.
        retire_legacy: The legacy ``market-recorder`` container runs and must be retired after READY.
        reason: Machine-readable reason (``not_running``, ``unchanged``, ``fingerprint_changed``,
            ``fingerprint_unreadable``, ``image_fingerprint_unreadable``, ``compose_config_changed``,
            ``both_running``, ``legacy_migration``).
    """

    action: Literal["keep", "start", "handover", "reconcile"]
    new_slot: Slot | None
    old_slot: Slot | None
    retire_legacy: bool
    reason: str


def _is_current(fp: str | None, image_fp: str | None) -> bool:
    return fp is not None and image_fp is not None and fp == image_fp


def decide_capture_action(
    slots: Sequence[SlotObservation], *, image_fingerprint: str | None, legacy_running: bool
) -> CaptureDecision:
    """Choose the capture deploy action from observed state; pure and total.

    Why: the bash script must not embed branching logic that tests cannot reach; every combination
    of (0/1/2 running slots, readable/unreadable fingerprints, config drift, legacy recorder) maps to
    exactly one action here.

    Raises:
        ValueError: ``slots`` does not contain exactly one observation per slot name.
    """
    names = sorted(o.slot for o in slots)
    if names != ["blue", "green"]:
        raise ValueError(f"slots must contain exactly one observation per slot name, got {names}")
    by_slot = {o.slot: o for o in slots}
    running = [o for o in slots if o.running]
    if len(running) == 2:
        current_ready = [o for o in running if o.heartbeat_ready and _is_current(o.fingerprint, image_fingerprint)]
        if len(current_ready) == 1:
            keep = current_ready[0]
        elif len(current_ready) > 1:
            keep = by_slot["blue"]
        else:
            ready = [o for o in running if o.heartbeat_ready]
            keep = ready[0] if len(ready) == 1 else by_slot["blue"]
        old = by_slot["green"] if keep.slot == "blue" else by_slot["blue"]
        return CaptureDecision(
            action="reconcile", new_slot=keep.slot, old_slot=old.slot, retire_legacy=legacy_running, reason="both_running"
        )
    if len(running) == 1:
        obs = running[0]
        idle: Slot = "green" if obs.slot == "blue" else "blue"
        if (
            image_fingerprint is not None
            and obs.fingerprint == image_fingerprint
            and obs.config_hash_matches
            and obs.heartbeat_ready
        ):
            return CaptureDecision(
                action="keep", new_slot=None, old_slot=None, retire_legacy=legacy_running, reason="unchanged"
            )
        if obs.fingerprint is None:
            reason = "fingerprint_unreadable"
        elif image_fingerprint is None:
            reason = "image_fingerprint_unreadable"
        elif obs.fingerprint != image_fingerprint:
            reason = "fingerprint_changed"
        elif not obs.config_hash_matches:
            reason = "compose_config_changed"
        else:
            reason = "not_ready"
        return CaptureDecision(
            action="handover", new_slot=idle, old_slot=obs.slot, retire_legacy=legacy_running, reason=reason
        )
    if legacy_running:
        return CaptureDecision(
            action="start", new_slot="blue", old_slot=None, retire_legacy=True, reason="legacy_migration"
        )
    return CaptureDecision(action="start", new_slot="blue", old_slot=None, retire_legacy=False, reason="not_running")


def _parse_ts(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    # Truncate sub-microsecond fractions (docker nanoseconds) to microseconds.
    if "." in text:
        t_pos = text.find("T")
        tz_pos = max(text.rfind("+", t_pos + 1 if t_pos >= 0 else 0), text.rfind("-", t_pos + 1 if t_pos >= 0 else 0))
        if tz_pos > 0:
            head, tail = text[:tz_pos], text[tz_pos:]
            whole, dot, frac_digits = head.partition(".")
            if dot and len(frac_digits) > 6:
                text = whole + "." + frac_digits[:6] + tail
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def evaluate_ready(
    heartbeat: Mapping[str, object] | None,
    *,
    container_started_at: datetime,
    now: datetime,
    stale_s: float,
) -> Literal["ready", "waiting"]:
    """READY per the shared contract, fenced against stale files from an earlier run of the same slot.

    READY ⇔ heartbeat present and parseable, ``started_at`` ≥ ``container_started_at`` (the file belongs
    to this container), ``now - ts`` ≤ ``stale_s``, ``stopped_at`` is null,
    ``rest.book_ticker.first_ok_at`` ≥ ``started_at``, ``ws.first_frame_at`` ≥ ``started_at`` and
    ``flush_failures == 0``. Anything else, including malformed or missing fields, is ``waiting``,
    never an exception, because the file may be mid-replace or absent in the first seconds.
    """
    try:
        if not isinstance(heartbeat, Mapping):
            return "waiting"
        started_at = _parse_ts(heartbeat.get("started_at"))
        ts = _parse_ts(heartbeat.get("ts"))
        if started_at is None or ts is None:
            return "waiting"
        if started_at < container_started_at:
            return "waiting"
        if (now - ts).total_seconds() > stale_s:
            return "waiting"
        if heartbeat.get("stopped_at") is not None:
            return "waiting"
        flush_failures = heartbeat.get("flush_failures")
        if flush_failures != 0:
            return "waiting"
        rest = heartbeat.get("rest")
        if not isinstance(rest, Mapping):
            return "waiting"
        book = rest.get("book_ticker")
        if not isinstance(book, Mapping):
            return "waiting"
        first_ok_at = _parse_ts(book.get("first_ok_at"))
        if first_ok_at is None or first_ok_at < started_at:
            return "waiting"
        ws = heartbeat.get("ws")
        if not isinstance(ws, Mapping):
            return "waiting"
        first_frame_at = _parse_ts(ws.get("first_frame_at"))
        if first_frame_at is None or first_frame_at < started_at:
            return "waiting"
        return "ready"
    except Exception:
        return "waiting"


def _parse_bool_token(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "ready"):
        return True
    if lowered in ("0", "false", "no", "waiting"):
        return False
    raise ValueError(f"invalid boolean token: {value!r}")


def _parse_slot_token(token: str) -> SlotObservation:
    pieces = token.split(":")
    if len(pieces) < 5:
        raise ValueError(f"invalid --slot {token!r}; want NAME:RUNNING:FP:CFG_MATCH:READY")
    name = pieces[0]
    if name not in ("blue", "green"):
        raise ValueError(f"invalid slot name {name!r}")
    running = _parse_bool_token(pieces[1])
    ready_raw = pieces[-1]
    cfg_raw = pieces[-2]
    fp_raw = ":".join(pieces[2:-2])
    fingerprint: str | None = None if fp_raw in ("", "-", "None", "null", "absent") else fp_raw
    config_match = _parse_bool_token(cfg_raw)
    heartbeat_ready = _parse_bool_token(ready_raw)
    slot_name: Slot = "blue" if name == "blue" else "green"
    return SlotObservation(
        slot=slot_name,
        running=running,
        fingerprint=fingerprint,
        config_hash_matches=config_match,
        heartbeat_ready=heartbeat_ready,
    )


def _normalize_image_fp(value: str | None) -> str | None:
    if value is None:
        return None
    if value.strip() in ("", "-", "None", "null", "absent"):
        return None
    return value.strip()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI.

    ``decide --image-fp FP [--legacy-running] --slot NAME:RUNNING:FP:CFG_MATCH:READY ...`` prints one
    line ``action=<a> new=<slot|-> old=<slot|-> legacy=<0|1> reason=<r>`` and returns 0.
    ``ready --heartbeat PATH --container-started-at ISO [--now ISO] --stale-s S`` returns 0 when READY,
    10 when waiting, and prints nothing. Malformed arguments return 2.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in ("decide", "ready"):
        return EXIT_USAGE
    command = args[0]
    rest = args[1:]
    if command == "decide":
        parser = argparse.ArgumentParser(prog="capture_handover decide")
        parser.add_argument("--image-fp", default=None)
        parser.add_argument("--legacy-running", action="store_true")
        parser.add_argument("--slot", action="append", default=[])
        try:
            parsed = parser.parse_args(rest)
        except SystemExit:
            return EXIT_USAGE
        try:
            observations = [_parse_slot_token(token) for token in parsed.slot]
            decision = decide_capture_action(
                observations,
                image_fingerprint=_normalize_image_fp(parsed.image_fp),
                legacy_running=bool(parsed.legacy_running),
            )
        except ValueError:
            return EXIT_USAGE
        new = decision.new_slot if decision.new_slot is not None else "-"
        old = decision.old_slot if decision.old_slot is not None else "-"
        legacy = "1" if decision.retire_legacy else "0"
        sys.stdout.write(f"action={decision.action} new={new} old={old} legacy={legacy} reason={decision.reason}\n")
        return EXIT_OK
    parser = argparse.ArgumentParser(prog="capture_handover ready")
    parser.add_argument("--heartbeat", required=True)
    parser.add_argument("--container-started-at", required=True)
    parser.add_argument("--now", default=None)
    parser.add_argument("--stale-s", required=True)
    try:
        parsed = parser.parse_args(rest)
    except SystemExit:
        return EXIT_USAGE
    try:
        container_started_at = _parse_ts(parsed.container_started_at)
        if container_started_at is None:
            return EXIT_USAGE
        now = _parse_ts(parsed.now) if parsed.now is not None else datetime.now(UTC)
        if now is None:
            return EXIT_USAGE
        stale_s = float(parsed.stale_s)
    except (ValueError, TypeError):
        return EXIT_USAGE
    try:
        raw = Path(parsed.heartbeat).read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, ValueError):
        return EXIT_NOT_READY
    if not isinstance(payload, dict):
        return EXIT_NOT_READY
    verdict = evaluate_ready(payload, container_started_at=container_started_at, now=now, stale_s=stale_s)
    return EXIT_OK if verdict == "ready" else EXIT_NOT_READY


if __name__ == "__main__":  # pragma: no cover - host entry point
    raise SystemExit(main())
