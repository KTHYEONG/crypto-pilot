"""CI deploy gate: proceed only when the live daemon is idle."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

BUSY_STAGES: frozenset[str] = frozenset({"refresh", "signal", "execute"})

EXIT_PROCEED: int = 0
EXIT_WAIT: int = 10

DEFAULT_MAX_WAIT_S: float = 3600.0
DEFAULT_STALE_AFTER_S: float = 2700.0


@dataclass(frozen=True, slots=True)
class GateDecision:
    action: Literal["proceed", "wait", "proceed_stale", "proceed_timeout"]
    reason: str


def decide_deploy(
    heartbeat: Mapping[str, object] | None,
    *,
    now: datetime,
    waited_s: float,
    max_wait_s: float,
    stale_after_s: float,
) -> GateDecision:
    """Preserve the daemon idle deployment decision independently of runtime dependencies. Args: observed heartbeat, aware current time, elapsed wait and existing timeout/staleness controls. Returns: the existing action and diagnostic reason. Raises: ValueError for a naive current time."""
    if now.tzinfo is None:
        raise ValueError("now must be tz-aware")
    ts_raw = heartbeat.get("ts") if heartbeat is not None else None
    if not isinstance(ts_raw, str):
        return GateDecision("proceed", "no_heartbeat")
    try:
        ts = datetime.fromisoformat(ts_raw)
    except ValueError:
        return GateDecision("proceed", "no_heartbeat")
    if ts.tzinfo is None:
        return GateDecision("proceed", "no_heartbeat")
    stage = heartbeat.get("stage") if heartbeat is not None else None
    if stage not in BUSY_STAGES:
        return GateDecision("proceed", "idle")
    age_s = (now - ts).total_seconds()
    if age_s > stale_after_s:
        return GateDecision("proceed_stale", f"stale:{stage} age_s={int(age_s)}")
    if waited_s >= max_wait_s:
        return GateDecision("proceed_timeout", f"max_wait:{stage} waited_s={int(waited_s)}")
    return GateDecision("wait", f"busy:{stage}")


def main(argv: Sequence[str] | None = None) -> int:
    """Provide the dependency-free deployment gate entry point. Args: existing heartbeat and wait CLI arguments. Returns: 10 for wait, otherwise 0. Raises: existing argument and file-read errors."""
    parser = argparse.ArgumentParser(description="Deploy gate: wait for daemon idle.")
    parser.add_argument("--heartbeat-file", required=True)
    parser.add_argument("--waited-s", type=float, required=True)
    parser.add_argument("--max-wait-s", type=float, default=DEFAULT_MAX_WAIT_S)
    parser.add_argument("--stale-after-s", type=float, default=DEFAULT_STALE_AFTER_S)
    parser.add_argument("--now", default=None)
    args = parser.parse_args(argv)
    raw = Path(args.heartbeat_file).read_text(encoding="utf-8")
    if not raw.strip():
        heartbeat: dict[str, object] | None = None
    else:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            heartbeat = None
        else:
            heartbeat = parsed if isinstance(parsed, dict) else None
    now = datetime.fromisoformat(args.now) if args.now is not None else datetime.now(UTC)
    decision = decide_deploy(
        heartbeat,
        now=now,
        waited_s=args.waited_s,
        max_wait_s=args.max_wait_s,
        stale_after_s=args.stale_after_s,
    )
    print(f"action={decision.action} reason={decision.reason}")  # noqa: T201 - contract-mandated CI output
    return EXIT_WAIT if decision.action == "wait" else EXIT_PROCEED


if __name__ == "__main__":
    raise SystemExit(main())
