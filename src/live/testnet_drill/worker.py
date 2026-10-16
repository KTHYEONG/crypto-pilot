"""Child-process executor for kill/restart drill scenarios.

The only module that terminates a process: ``os._exit`` is injected into the session transport (``KILL_*`` rules)
and into the audit hook (audit-event kills). The child never tears down, so the journal, audit and venue state it
leaves behind are exactly what the parent's restart sequence must recover.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

KILL_EXIT_CODE = 137


def emit_milestone(event: str, **fields: Any) -> None:
    """Write one redacted JSON milestone line to stdout."""
    from src.live.testnet_drill.faults import redact

    payload = {"event": event, **fields}
    sys.stdout.write(redact(json.dumps(payload, sort_keys=True, default=str)) + "\n")
    sys.stdout.flush()


def install_audit_hook(
    audit: Any,
    event_name: str,
    action: Callable[[], None],
    *,
    fields: Mapping[str, Any] | None = None,
) -> Callable[[], None]:
    """Run ``action`` after the named audit event (matching ``fields``) has been written and flushed.

    Returns an undo callable restoring the original ``record``.
    """
    original = audit.record
    expected = dict(fields or {})

    def fire(event: str, kwargs: Mapping[str, Any]) -> None:
        if event == event_name and all(kwargs.get(key) == value for key, value in expected.items()):
            action()

    if isinstance(audit, type):

        def unbound(self: Any, event: str, *args: Any, **kwargs: Any) -> Any:
            result = original(self, event, *args, **kwargs)
            fire(event, kwargs)
            return result

        setattr(audit, "record", unbound)  # noqa: B010 - class-level hook, undone when the worker returns
        return lambda: setattr(audit, "record", original)

    def wrapped(event: str, *args: Any, **kwargs: Any) -> Any:
        result = original(event, *args, **kwargs)
        fire(event, kwargs)
        return result

    setattr(audit, "record", wrapped)  # noqa: B010 - instance-level hook
    return lambda: setattr(audit, "record", original)


def install_audit_kill(
    audit: Any,
    event_name: str,
    kill: Callable[[int], NoReturn],
    *,
    exit_code: int = KILL_EXIT_CODE,
    fields: Mapping[str, Any] | None = None,
) -> Callable[[], None]:
    """Trigger kill after the named audit event is durably written; returns the undo callable."""
    return install_audit_hook(audit, event_name, lambda: kill(exit_code), fields=fields)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="testnet-drill-worker")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--rules", required=True)
    return parser.parse_args(list(argv) if argv is not None else None)


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rules(raw: Any) -> tuple[Any, ...]:
    from src.live.testnet_drill.faults import FaultAction, FaultRule

    if not isinstance(raw, list):
        raise ValueError("rules must be a JSON list")
    return tuple(
        FaultRule(
            method=str(r["method"]),
            path=str(r["path"]),
            params=tuple((str(k), str(v)) for k, v in r.get("params", ())),
            action=FaultAction(str(r["action"])),
            occurrence=r.get("occurrence"),
            times=int(r.get("times", 1)),
            status=int(r.get("status", 200)),
            headers=tuple((str(k), str(v)) for k, v in r.get("headers", ())),
            body=str(r.get("body", "{}")).encode(),
            delay_s=float(r.get("delay_s", 0.0)),
            truncate_to=int(r.get("truncate_to", 0)),
            exit_code=int(r.get("exit_code", KILL_EXIT_CODE)),
        )
        for r in raw
    )


def _perform(session: Any, script: Mapping[str, Any], stop: Callable[[], bool]) -> dict[str, Any]:
    from src.live.testnet_drill.execution import cancel_probe, make_intent, run_cycle, settle_journal, submit_probe

    action = str(script["action"])
    params = dict(script.get("params", {}))
    if action == "probe":
        submit_probe(session, params)
        return {}
    if action == "settle":
        submit_probe(session, params)
        settle_journal(session)
        return {}
    if action == "cancel":
        submit_probe(session, params)
        cancel_probe(session, str(params["symbol"]), str(params["newClientOrderId"]))
        return {}
    if action == "cycle":
        symbols = script.get("symbols") or [session.config.symbols[0]]
        anchor = int(script.get("anchor_offset_bps", 0))
        trace = run_cycle(
            session, [make_intent(session, str(symbol), anchor_offset_bps=anchor) for symbol in symbols], stop=stop
        )
        return {"escaped": trace.escaped, "outcomes": [o.status for o in trace.outcomes]}
    if action == "runner_cycle":
        import pandas as pd

        from src.live.runner import run_shadow_cycle
        from src.live.testnet_drill.scenarios.cycle_e2e import cycle_now
        from src.live.testnet_drill.seams import production_bound

        decision_time = pd.Timestamp(str(script["decision_time"]))
        with production_bound(session):
            report = run_shadow_cycle(
                session.settings, decision_time, Path(str(script["weights_path"])), now=cycle_now(decision_time)
            )
        return {"cycle_status": report.status}
    if action == "recovery":
        from src.live.testnet_drill.scenarios.restart_recovery import restart_recovery

        restart_recovery(session)
        return {}
    raise ValueError(f"unknown worker action {action!r}")


def main(
    argv: Sequence[str] | None = None,
    *,
    kill: Callable[[int], NoReturn] | None = None,
    inner: Any | None = None,
    settings: Any | None = None,
) -> int:
    """Child entrypoint: run a scripted intent set against the drill session, honoring KILL_* rules.

    Returns 0 on a clean finish and 1 on any error; a configured kill never returns (exit 137).
    """
    args = _parse_args(argv)
    workspace = Path(args.session_dir)
    kill_fn: Callable[[int], NoReturn] = kill if kill is not None else os._exit

    def announced_kill(code: int) -> NoReturn:
        emit_milestone("killed_at", exit_code=code)
        kill_fn(code)

    emit_milestone("started", session_dir=str(workspace))
    previous_handler: Any = None
    undo_hooks: list[Callable[[], None]] = []
    stop_requested: list[bool] = []
    session: Any = None
    try:
        script = _load_json(args.script)
        rules = _rules(_load_json(args.rules))
        from src.live.settings import LiveSettings
        from src.live.testnet_drill.config import config_from_json
        from src.live.testnet_drill.session import open_drill_session

        config = config_from_json(dict(script["config"]))
        session = open_drill_session(
            settings if settings is not None else LiveSettings(),
            config,
            run_id=args.run_id,
            rules=rules,
            workspace=workspace,
            inner=inner,
            kill=announced_kill,
        )
        kill_on = script.get("kill_on_audit_event")
        if kill_on:
            # The runner writes through its own AuditLog instances, so a runner cycle is hooked at class level.
            from src.live.audit import AuditLog

            target = AuditLog if script.get("action") == "runner_cycle" else session.audit
            undo_hooks.append(
                install_audit_kill(target, str(kill_on["event"]), announced_kill, fields=kill_on.get("fields"))
            )
        previous_handler = signal.signal(signal.SIGTERM, lambda signum, frame: stop_requested.append(True))
        sigterm_on = script.get("sigterm_on_audit_event")
        if sigterm_on:
            install_audit_hook(session.audit, str(sigterm_on), lambda: os.kill(os.getpid(), signal.SIGTERM))
        result = _perform(session, script, lambda: bool(stop_requested))
    except Exception as exc:
        emit_milestone("finished", status="error", error=type(exc).__name__)
        return 1
    finally:
        for undo in reversed(undo_hooks):
            undo()
        if previous_handler is not None:
            signal.signal(signal.SIGTERM, previous_handler)
        if session is not None:
            from src.live.testnet_drill.session import close_session_handles

            close_session_handles(session)
    emit_milestone("finished", status="clean", **result)
    return 0


if __name__ == "__main__":  # pragma: no cover - child-process entry
    raise SystemExit(main())
