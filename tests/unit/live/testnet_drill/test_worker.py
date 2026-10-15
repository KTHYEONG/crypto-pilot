"""Worker child-process invariants."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import pytest

from src.live.testnet_drill import worker


class _KillFiredError(BaseException):
    pass


def _kill(code: int) -> None:
    raise _KillFiredError(code)


class _Audit:
    def __init__(self, path):
        self._path = path

    def record(self, event, **fields):
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": event, **fields}) + "\n")
        return True


def test_kill_after_durable_audit(tmp_path) -> None:
    audit = _Audit(tmp_path / "audit.jsonl")
    worker.install_audit_kill(audit, "fill", _kill)
    audit.record("order_posted", order_id="x")
    with pytest.raises(_KillFiredError):
        audit.record("fill", order_id="x")
    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["event"] == "fill"


def test_audit_kill_honors_field_predicate(tmp_path) -> None:
    audit = _Audit(tmp_path / "audit.jsonl")
    worker.install_audit_kill(audit, "order_cancelled", _kill, fields={"reason": "repeg"})
    audit.record("order_cancelled", reason="timeout")
    with pytest.raises(_KillFiredError):
        audit.record("order_cancelled", reason="repeg")


def _write(tmp_path, script, rules):
    script_path, rules_path = tmp_path / "script.json", tmp_path / "rules.json"
    script_path.write_text(json.dumps(script), encoding="utf-8")
    rules_path.write_text(json.dumps(rules), encoding="utf-8")
    return script_path, rules_path


def _argv(session, script_path, rules_path):
    return ["--session-dir", str(session.workspace), "--run-id", session.run_id,
            "--script", str(script_path), "--rules", str(rules_path)]


def test_kill_before_delivery_leaves_journal_submit(exchange_session, tmp_path) -> None:
    from src.live.order_journal import OrderJournal
    from src.live.testnet_drill.config import config_to_json
    from src.live.testnet_drill.scenarios.lifecycle_runs import _limit

    session, venue, _ = exchange_session()
    params = _limit(session, distance=True)
    script_path, rules_path = _write(
        tmp_path,
        {"action": "probe", "params": params, "config": config_to_json(session.config)},
        [{"method": "POST", "path": "/fapi/v1/order", "action": "kill_before_delivery"}],
    )
    posts_before = sum(1 for _, method, _, _ in venue.calls if method == "POST")
    buffer = io.StringIO()
    with redirect_stdout(buffer), pytest.raises(_KillFiredError):
        worker.main(_argv(session, script_path, rules_path), kill=_kill, inner=venue, settings=session.settings)
    journal = OrderJournal(session.journal.path)
    assert any(s.client_order_id == params["newClientOrderId"] for s in journal.submits())
    assert journal.terminal(params["newClientOrderId"]) is None
    assert sum(1 for _, method, _, _ in venue.calls if method == "POST") == posts_before
    milestones = [json.loads(line) for line in buffer.getvalue().splitlines()]
    assert [m["event"] for m in milestones] == ["started", "killed_at"]
    assert milestones[-1]["exit_code"] == 137


def test_no_secrets_on_stdout(tmp_path) -> None:
    probe_key = "order-key-abc"  # noqa: S105 - redaction target, not a real credential
    probe_sig = "deadbeef"  # noqa: S105 - redaction target, not a real credential
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        worker.emit_milestone("started", session_dir=str(tmp_path), note=f"api_key={probe_key} signature={probe_sig}")
    out = buffer.getvalue()
    assert probe_key not in out
    assert probe_sig not in out
    assert json.loads(out.strip())["event"] == "started"


def test_os_exit_is_confined_to_worker() -> None:
    import inspect

    import src.live.testnet_drill.faults as faults
    import src.live.testnet_drill.session as session_module

    assert "os._exit" in inspect.getsource(worker)
    assert "os._exit" not in inspect.getsource(session_module)
    assert "os._exit" not in inspect.getsource(faults)


def test_main_error_path_returns_nonzero(tmp_path, capsys) -> None:
    missing = tmp_path / "missing.json"
    code = worker.main(["--session-dir", str(tmp_path / "ws"), "--run-id", "w1", "--script", str(missing),
                        "--rules", str(missing)])
    assert code == 1
    assert '"finished"' in capsys.readouterr().out


def test_unknown_action_fails_cleanly(exchange_session, tmp_path, capsys) -> None:
    from src.live.testnet_drill.config import config_to_json

    session, venue, _ = exchange_session()
    script_path, rules_path = _write(tmp_path, {"action": "nope", "config": config_to_json(session.config)}, [])
    assert worker.main(_argv(session, script_path, rules_path), kill=_kill, inner=venue, settings=session.settings) == 1
    assert '"error": "ValueError"' in capsys.readouterr().out


def test_clean_finish_returns_zero_without_teardown(exchange_session, tmp_path, capsys) -> None:
    from src.live.testnet_drill.config import config_to_json
    from src.live.testnet_drill.scenarios.lifecycle_runs import _limit

    session, venue, _ = exchange_session()
    params = _limit(session, distance=True)
    script_path, rules_path = _write(tmp_path, {"action": "probe", "params": params,
                                                "config": config_to_json(session.config)}, [])
    assert worker.main(_argv(session, script_path, rules_path), kill=_kill, inner=venue, settings=session.settings) == 0
    assert '"clean"' in capsys.readouterr().out
    assert any(row["clientOrderId"] == params["newClientOrderId"] for row in session.client.open_orders())


def test_config_round_trip() -> None:
    from src.live.testnet_drill.config import config_from_json, config_to_json
    from tests.unit.live.testnet_drill._stubs import make_config

    config = make_config(symbols=("BTCUSDT", "ETHUSDT"))
    assert config_from_json(json.loads(json.dumps(config_to_json(config)))) == config


@pytest.mark.parametrize("action", ["settle", "cancel", "recovery", "runner_cycle"])
def test_actions_finish_cleanly_without_kill_rules(exchange_session, tmp_path, capsys, action) -> None:
    from src.live.testnet_drill.config import config_to_json
    from src.live.testnet_drill.scenarios.cycle_e2e import write_weights_artifact
    from src.live.testnet_drill.scenarios.lifecycle_runs import _limit

    session, venue, _ = exchange_session(config={"symbols": ("BTCUSDT", "ETHUSDT")})
    script = {"action": action, "params": _limit(session, distance=True), "config": config_to_json(session.config)}
    if action == "runner_cycle":
        import pandas as pd

        day = pd.Timestamp.now(tz="UTC").normalize()
        script["weights_path"] = str(write_weights_artifact(session, {day: {"BTCUSDT": 0.0}}))
        script["decision_time"] = day.isoformat()
    script_path, rules_path = _write(tmp_path, script, [])
    assert worker.main(_argv(session, script_path, rules_path), kill=_kill, inner=venue, settings=session.settings) == 0
    finished = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert finished["status"] == "clean"


def test_rules_payload_must_be_a_list(exchange_session, tmp_path, capsys) -> None:
    from src.live.testnet_drill.config import config_to_json

    session, venue, _ = exchange_session()
    script_path, rules_path = _write(tmp_path, {"action": "probe", "config": config_to_json(session.config)}, {"x": 1})
    assert worker.main(_argv(session, script_path, rules_path), kill=_kill, inner=venue, settings=session.settings) == 1


def test_parent_transport_kill_is_refused(exchange_session) -> None:
    from src.live.testnet_drill.guard import DrillRefused

    session, _, _ = exchange_session()
    with pytest.raises(DrillRefused, match="parent_kill"):
        session.transport._kill(137)
