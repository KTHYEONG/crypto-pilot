# ruff: noqa
"""Deprecated shim: tests moved to test_runner_*.py (P0 split)."""


def test_runner_market_client_uses_testnet_venue_under_live_testnet(tmp_path, monkeypatch) -> None:
    import pandas as pd

    import src.live.runner as runner_mod
    from src.live.settings import TESTNET_FAPI_URL, ExecutionMode, LiveSettings

    captured: list[str] = []

    class _FakeClient:
        def __init__(self, base_url, api_key, api_secret, mode, audit, *, recv_window_ms):
            captured.append(base_url)

    monkeypatch.setattr(runner_mod, "BinanceFuturesRestClient", _FakeClient)
    monkeypatch.setattr(runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl")
    settings = LiveSettings(
        mode=ExecutionMode.LIVE_TESTNET,
        order_api_key="testnet-key",
        order_api_secret="testnet-secret",
    )
    decision = pd.Timestamp("2026-09-15 00:00Z")

    runner_mod._market_client(settings, decision)
    runner_mod._order_client(settings, decision)

    # Then: 필터/호가 조회와 주문이 같은 테스트넷 베뉴
    assert captured == [TESTNET_FAPI_URL, TESTNET_FAPI_URL]


def test_order_client_ignores_testnet_order_key_under_paper_mode(tmp_path, monkeypatch) -> None:
    import pandas as pd

    import src.live.runner as runner_mod
    from src.live.settings import ExecutionMode, LiveSettings

    captured: list[tuple[str, str]] = []

    class _FakeClient:
        def __init__(self, base_url, api_key, api_secret, mode, audit, *, recv_window_ms):
            captured.append((base_url, api_key.get_secret_value()))

    monkeypatch.setattr(runner_mod, "BinanceFuturesRestClient", _FakeClient)
    monkeypatch.setattr(runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl")
    settings = LiveSettings(
        mode=ExecutionMode.PAPER,
        api_key="mainnet-key",
        api_secret="mainnet-secret",
        order_api_key="testnet-key",
        order_api_secret="testnet-secret",
    )
    decision = pd.Timestamp("2026-09-16 00:00Z")

    runner_mod._order_client(settings, decision)

    # Then: PAPER는 order_base_url이 메인넷이므로 테스트넷 전용 order_api_key를 절대 보내지 않는다.
    assert captured == [(settings.order_base_url, "mainnet-key")]


def test_live_runner_metadata_carries_configured_digest() -> None:
    import pandas as pd
    from src.live.runner import _execution_quality_metadata
    from src.live.settings import LiveSettings
    now = pd.Timestamp('2026-09-14', tz='UTC')
    settings = LiveSettings(strategy_digest='abc')
    assert _execution_quality_metadata(settings, now) == {'strategy_digest': 'abc', 'observed_at': now}


def _seed_policy_cycle_artifact(tmp_path) -> tuple:
    import pandas as pd

    from tests.unit.live._runner_stubs import DECISION_TIME, NOW

    frame = pd.DataFrame(
        {"AAAUSDT": [0.02], "BUSDT": [-0.02]},
        index=pd.DatetimeIndex([DECISION_TIME]),
    )
    path = tmp_path / "deployed_target_weights.parquet"
    frame.to_parquet(path, index=True)
    closes = pd.DataFrame(
        100.0, index=pd.DatetimeIndex(frame.index), columns=list(frame.columns), dtype="float64",
    )
    from src.live.deployed_weights import decision_ohlcv_close_path

    closes.to_parquet(decision_ohlcv_close_path(path), index=True)
    return path, DECISION_TIME, NOW


def _install_policy_cycle_stubs(tmp_path, monkeypatch, captured: dict, *, journal_fills: bool = False) -> None:
    import src.live.orderbook as ob_mod
    import src.live.runner as runner_mod
    from src.live.executor import ExecutionOutcome
    from tests.unit.live._runner_stubs import StubMarketClient, StubOrderClient

    monkeypatch.setattr(
        runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient()
    )
    monkeypatch.setattr(
        runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient()
    )

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        from decimal import Decimal

        import pandas as pd

        from tests.unit.live._runner_stubs import NOW

        captured["policy"] = policy
        outcomes = [
            ExecutionOutcome(
                symbol=intent.symbol,
                filled_qty=intent.quantity,
                unfilled_qty=Decimal("0"),
                avg_fill_price=Decimal("100"),
                chases=0,
                status="FILLED",
            )
            for intent in intents
        ]
        if journal_fills:
            journal = kwargs.get("journal")
            attempt = kwargs.get("attempt")
            if journal is not None and attempt is not None:
                for outcome in outcomes:
                    if outcome.filled_qty <= 0:
                        continue
                    side = next((i.side for i in intents if i.symbol == outcome.symbol), "BUY")
                    journal.record_fill(
                        kind="execution",
                        attempt_seq=attempt.attempt_seq,
                        symbol=outcome.symbol,
                        side=side,
                        quantity=outcome.filled_qty,
                        price=Decimal("100"),
                        fee_bps=5.0,
                        liquidity="taker",
                        reason="timeout_taker",
                        filled_at=pd.Timestamp(NOW),
                        client_order_id=None,
                        leg_index=0,
                        cumulative_executed_qty=None,
                        simulated=True,
                    )
        return tuple(outcomes)

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)
    monkeypatch.setattr(
        runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl"
    )
    monkeypatch.setattr(ob_mod, "capture_order_books", lambda *a, **k: [])
    monkeypatch.setattr(ob_mod, "append_order_book_snapshots", lambda *a, **k: [])


def test_runner_selects_strict_passive_policy_from_settings(tmp_path, monkeypatch) -> None:
    import src.live.runner as runner_mod
    from src.live.settings import LiveSettings
    from tests.unit.live._runner_stubs import DECISION_TIME, NOW

    path, _, _ = _seed_policy_cycle_artifact(tmp_path)
    captured: dict = {}
    _install_policy_cycle_stubs(tmp_path, monkeypatch, captured)
    settings = LiveSettings(
        ledger_path=str(tmp_path / "ledger_strict.json"),
        execution_policy="strict_passive",
    )
    runner_mod.run_shadow_cycle(settings, DECISION_TIME, path, now=NOW)

    assert captured["policy"].passive_pricing == "anchored"
    assert captured["policy"].passive_deadline_s == 1800.0

    captured.clear()
    plain_settings = LiveSettings(ledger_path=str(tmp_path / "ledger_taker.json"))
    runner_mod.run_shadow_cycle(plain_settings, DECISION_TIME, path, now=NOW)

    assert captured["policy"].passive_pricing == "touch_chase"
    assert captured["policy"].passive_deadline_s == 180.0


def _seed_run_cycle_artifact(tmp_path, monkeypatch, record_run_id: str):
    import pandas as pd

    import src.live.settings as settings_mod
    from src.live.settings import LiveSettings
    from tests.unit.live._runner_stubs import DECISION_TIME, NOW

    monkeypatch.setattr(settings_mod, "DATA_DIR", tmp_path / "data")
    settings = LiveSettings(record_run_id=record_run_id)
    weights_path = tmp_path / "data" / "state" / "runs" / record_run_id / "target_weights.parquet"
    frame = pd.DataFrame(
        {"AAAUSDT": [0.02], "BUSDT": [-0.02]},
        index=pd.DatetimeIndex([DECISION_TIME]),
    )
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(weights_path, index=True)
    closes = pd.DataFrame(
        100.0, index=pd.DatetimeIndex(frame.index), columns=list(frame.columns), dtype="float64",
    )
    from src.live.deployed_weights import decision_ohlcv_close_path

    closes.to_parquet(decision_ohlcv_close_path(weights_path), index=True)
    return settings, weights_path, DECISION_TIME, NOW


def test_run_manifest_written_once_on_first_cycle(tmp_path, monkeypatch) -> None:
    import json

    import src.live.runner as runner_mod
    from src.mhs.frozen_research_candidate import FROZEN_MHS_TOP20_V2

    record_run_id = "frozen_top20_v2_bayes_maker_20260922"
    settings, weights_path, decision_time, now = _seed_run_cycle_artifact(tmp_path, monkeypatch, record_run_id)
    captured: dict = {}
    _install_policy_cycle_stubs(tmp_path, monkeypatch, captured)
    report = runner_mod.run_shadow_cycle(settings, decision_time, weights_path, now=now)
    assert report.status == "COMPLETE"

    manifest_path = tmp_path / "data" / "state" / "runs" / record_run_id / "run_manifest.json"
    assert manifest_path.exists()
    first = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert first["run_id"] == record_run_id
    assert first["strategy_id"] == FROZEN_MHS_TOP20_V2.strategy_id
    from src.mhs.params import FROZEN_GROWTH_NAME_CLIP

    assert first["name_clip"] == FROZEN_GROWTH_NAME_CLIP
    assert first["execution_policy"] == settings.execution_policy
    assert first["paper_fill_model"] == settings.paper_fill_model
    assert first["mode"] == settings.mode.value
    assert first["seed_equity_usdt"] == settings.notional_equity_usdt
    assert first["unit_bootstrap_sha256"]
    assert "started_at" in first

    second = runner_mod.run_shadow_cycle(settings, decision_time, weights_path, now=now)
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["started_at"] == first["started_at"]
    assert second.status == "COMPLETE"


def test_run_manifest_mismatch_halts(tmp_path, monkeypatch) -> None:
    import json

    import pytest

    import src.live.runner as runner_mod
    from src.common.errors import DataIntegrityError

    record_run_id = "frozen_top20_v2_bayes_maker_20260922"
    settings, weights_path, decision_time, now = _seed_run_cycle_artifact(tmp_path, monkeypatch, record_run_id)
    manifest_path = tmp_path / "data" / "state" / "runs" / record_run_id / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = runner_mod._current_run_manifest(settings, now)
    payload["execution_policy"] = "taker_parity"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataIntegrityError):
        runner_mod._assert_run_manifest_compatible(
            settings.model_copy(update={"execution_policy": "strict_passive"})
        )

    captured: dict = {}
    _install_policy_cycle_stubs(tmp_path, monkeypatch, captured)
    strict_settings = settings.model_copy(update={"execution_policy": "strict_passive"})
    report = runner_mod.run_shadow_cycle(strict_settings, decision_time, weights_path, now=now)
    assert report.status == "HALT"
    assert "mismatch" in (report.reason or "")


def test_order_journal_follows_settings_path(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    import src.live.runner as runner_mod

    record_run_id = "frozen_top20_v2_bayes_maker_20260922"
    settings, weights_path, decision_time, now = _seed_run_cycle_artifact(tmp_path, monkeypatch, record_run_id)
    seen: dict = {}
    real_journal = runner_mod.OrderJournal

    def _capturing_journal(path):
        seen["path"] = Path(path)
        return real_journal(Path(path))

    monkeypatch.setattr(runner_mod, "OrderJournal", _capturing_journal)
    captured: dict = {}
    _install_policy_cycle_stubs(tmp_path, monkeypatch, captured)
    runner_mod.run_shadow_cycle(settings, decision_time, weights_path, now=now)

    assert seen["path"] == Path(settings.order_journal_path)
    journal = real_journal(seen["path"])
    from decimal import Decimal

    journal.record_submit(
        "wired-cid",
        "AAAUSDT",
        journal.next_submit_seq(),
        attempt_seq=0,
        side="BUY",
        quantity=Decimal("1"),
        reduce_only=False,
        leg_index=0,
    )
    assert seen["path"].exists()


def test_missing_unit_bootstrap_yields_null_manifest_hash(tmp_path, monkeypatch) -> None:
    import src.live.runner as runner_mod

    record_run_id = "frozen_top20_v2_bayes_maker_20260922"
    settings, _, _, now = _seed_run_cycle_artifact(tmp_path, monkeypatch, record_run_id)
    missing = settings.model_copy(update={"unit_bootstrap_path": str(tmp_path / "nope.parquet")})
    assert runner_mod._unit_bootstrap_sha256(missing) is None
    payload = runner_mod._current_run_manifest(missing, now)
    assert payload["unit_bootstrap_sha256"] is None


def test_resealed_bootstrap_keeps_manifest_digest(tmp_path) -> None:
    import base64
    import hashlib
    import io
    import pandas as pd
    from pydantic import SecretStr
    import src.live.runner as runner_mod
    from src.live.crypto import derive_key, open_bytes, seal_bytes
    from src.live.settings import LiveSettings

    key_b64 = base64.b64encode(b"0" * 32).decode()
    key = derive_key(SecretStr(key_b64))
    frame = pd.DataFrame({"ret": [0.01, -0.02]}, index=pd.date_range("2026-01-01", periods=2, tz="UTC"))
    buf = io.BytesIO()
    frame.to_parquet(buf, index=True)
    plaintext = buf.getvalue()
    blob = seal_bytes(plaintext, key)
    assert open_bytes(blob, key) == plaintext
    p1 = tmp_path / "b1.parquet.enc"
    p2 = tmp_path / "b2.parquet.enc"
    p1.write_bytes(blob)
    p2.write_bytes(blob)
    s1 = LiveSettings(unit_bootstrap_path=str(p1), artifact_key=key_b64)
    s2 = LiveSettings(unit_bootstrap_path=str(p2), artifact_key=key_b64)
    assert runner_mod._unit_bootstrap_sha256(s1) == runner_mod._unit_bootstrap_sha256(s2) == hashlib.sha256(plaintext).hexdigest()


def test_manifest_records_applied_clip(tmp_path) -> None:
    import pandas as pd
    import src.live.runner as runner_mod
    from src.live.settings import LiveSettings
    from src.mhs.params import FROZEN_GROWTH_NAME_CLIP

    now = pd.Timestamp("2026-09-22 00:00Z")
    settings = LiveSettings(unit_bootstrap_path=str(tmp_path / "nope.parquet"))
    payload = runner_mod._current_run_manifest(settings, now)
    assert payload["name_clip"] == FROZEN_GROWTH_NAME_CLIP


def test_legacy_ciphertext_digest_migrates_once(tmp_path, monkeypatch) -> None:
    import base64
    import hashlib
    import io
    import json
    import pandas as pd
    import pytest
    from pydantic import SecretStr
    import src.live.runner as runner_mod
    import src.live.settings as settings_mod
    from src.live.crypto import derive_key, seal_bytes
    from src.live.settings import LiveSettings
    from src.common.errors import DataIntegrityError

    monkeypatch.setattr(settings_mod, "DATA_DIR", tmp_path / "data")
    key_b64 = base64.b64encode(b"1" * 32).decode()
    key = derive_key(SecretStr(key_b64))
    frame = pd.DataFrame({"ret": [0.03]}, index=pd.date_range("2026-01-01", periods=1, tz="UTC"))
    buf = io.BytesIO()
    frame.to_parquet(buf, index=True)
    blob = seal_bytes(buf.getvalue(), key)
    boot = tmp_path / "boot.parquet.enc"
    boot.write_bytes(blob)
    record_run_id = "legacy_digest_run_20260922"
    settings = LiveSettings(record_run_id=record_run_id, unit_bootstrap_path=str(boot), artifact_key=key_b64)
    now = pd.Timestamp("2026-09-22 00:00Z")
    manifest_path = tmp_path / "data" / "state" / "runs" / record_run_id / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    current = runner_mod._current_run_manifest(settings, now)
    legacy = dict(current)
    legacy["unit_bootstrap_sha256"] = hashlib.sha256(blob).hexdigest()
    assert legacy["unit_bootstrap_sha256"] != current["unit_bootstrap_sha256"]
    manifest_path.write_text(json.dumps(legacy), encoding="utf-8")
    runner_mod._assert_run_manifest_compatible(settings)
    migrated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert migrated["unit_bootstrap_sha256"] == current["unit_bootstrap_sha256"]
    other = dict(current)
    other["unit_bootstrap_sha256"] = hashlib.sha256(b"different").hexdigest()
    manifest_path.write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        runner_mod._assert_run_manifest_compatible(settings)


def test_legacy_null_clip_manifest_migrates_once(tmp_path, monkeypatch) -> None:
    import json
    import pandas as pd
    import pytest
    import src.live.runner as runner_mod
    import src.live.settings as settings_mod
    from src.live.settings import LiveSettings
    from src.common.errors import DataIntegrityError

    monkeypatch.setattr(settings_mod, "DATA_DIR", tmp_path / "data")
    record_run_id = "legacy_clip_run_20260922"
    settings = LiveSettings(record_run_id=record_run_id, unit_bootstrap_path=str(tmp_path / "nope.parquet"))
    now = pd.Timestamp("2026-09-22 00:00Z")
    manifest_path = tmp_path / "data" / "state" / "runs" / record_run_id / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    current = runner_mod._current_run_manifest(settings, now)
    legacy = dict(current)
    legacy["name_clip"] = None
    manifest_path.write_text(json.dumps(legacy), encoding="utf-8")
    runner_mod._assert_run_manifest_compatible(settings)
    migrated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert migrated["name_clip"] == 0.05
    bad = dict(current)
    bad["name_clip"] = 0.1
    manifest_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        runner_mod._assert_run_manifest_compatible(settings)


def test_sealed_bootstrap_wrong_key_raises(tmp_path) -> None:
    import base64
    import io
    import pandas as pd
    import pytest
    from pydantic import SecretStr
    import src.live.runner as runner_mod
    from src.live.crypto import derive_key, seal_bytes
    from src.live.settings import LiveSettings
    from src.common.errors import DataIntegrityError

    key_b64 = base64.b64encode(b"2" * 32).decode()
    wrong_b64 = base64.b64encode(b"3" * 32).decode()
    frame = pd.DataFrame({"ret": [0.01]}, index=pd.date_range("2026-01-01", periods=1, tz="UTC"))
    buf = io.BytesIO()
    frame.to_parquet(buf, index=True)
    blob = seal_bytes(buf.getvalue(), derive_key(SecretStr(key_b64)))
    boot = tmp_path / "boot.parquet.enc"
    boot.write_bytes(blob)
    settings = LiveSettings(unit_bootstrap_path=str(boot), artifact_key=wrong_b64)
    with pytest.raises(DataIntegrityError):
        runner_mod._unit_bootstrap_sha256(settings)


def _audit_events_by_name(tmp_path, event: str) -> list:
    import json

    path = tmp_path / "shadow_cycle.jsonl"
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record.get("event") == event:
            events.append(record)
    return events


def _seed_live_tax_cycle(tmp_path, monkeypatch):
    import src.live.runner as runner_mod
    from src.live.settings import LiveSettings

    path, decision_time, now = _seed_policy_cycle_artifact(tmp_path)
    captured: dict = {}
    _install_policy_cycle_stubs(tmp_path, monkeypatch, captured, journal_fills=True)
    settings = LiveSettings(
        mode="live_testnet",
        order_api_key="testnet-key",
        order_api_secret="testnet-secret",
        ledger_path=str(tmp_path / "ledger_live_tax.json"),
        order_journal_path=str(tmp_path / "order_journal.jsonl"),
        fills_dir=str(tmp_path / "fills"),
        tax_ledger_dir=str(tmp_path / "tax"),
        execution_quality_dir=str(tmp_path / "eq"),
        portfolio_state_dir=str(tmp_path / "portfolio"),
    )
    return runner_mod, settings, path, decision_time, now


def test_live_tax_issues_audited_without_halting(tmp_path, monkeypatch) -> None:
    """Live collection issue: cycle COMPLETE with one tax_collect_issue audit event."""
    from src.live.tax_ledger import TaxCollectionIssue

    runner_mod, settings, path, decision_time, now = _seed_live_tax_cycle(tmp_path, monkeypatch)
    calls: list = []

    def _fake_collect(client, symbols, tax_dir, mode, *, now, settings=None):
        calls.append((symbols, str(tax_dir), mode))
        return 2, (TaxCollectionIssue(stream="trades:AAAUSDT", stage="fetch", detail="boom"),)

    monkeypatch.setattr(runner_mod, "collect_and_persist_live_tax", _fake_collect)
    report = runner_mod.run_shadow_cycle(settings, decision_time, path, now=now)
    assert report.status == "COMPLETE"
    assert len(calls) == 1
    events = _audit_events_by_name(tmp_path, "tax_collect_issue")
    assert len(events) == 1
    assert events[0]["stream"] == "trades:AAAUSDT"
    assert events[0]["stage"] == "fetch"


def test_live_tax_invalid_watermark_audited_once(tmp_path, monkeypatch) -> None:
    """Corrupt watermark: one tax_watermark_invalid audit event and the cycle is COMPLETE."""
    from src.common.errors import DataIntegrityError

    runner_mod, settings, path, decision_time, now = _seed_live_tax_cycle(tmp_path, monkeypatch)

    def _fake_collect(client, symbols, tax_dir, mode, *, now, settings=None):
        raise DataIntegrityError("tax watermark unreadable")

    monkeypatch.setattr(runner_mod, "collect_and_persist_live_tax", _fake_collect)
    report = runner_mod.run_shadow_cycle(settings, decision_time, path, now=now)
    assert report.status == "COMPLETE"
    events = _audit_events_by_name(tmp_path, "tax_watermark_invalid")
    assert len(events) == 1


def test_reject_cluster_alerts_once(tmp_path) -> None:
    """Spec 02: one intent_reject_cluster alert for a code shared by >= threshold symbols."""
    import json
    from decimal import Decimal

    import pandas as pd

    from src.live.audit import AuditLog
    from src.live.executor import ExecutionOutcome
    from src.live.runner import _alert_reject_cluster
    from src.live.settings import LiveSettings

    def _rejected(symbol, code):
        return ExecutionOutcome(symbol=symbol, filled_qty=Decimal(0), unfilled_qty=Decimal(1),
                                avg_fill_price=None, chases=0, status="REJECTED", reject_code=code)

    audit_path = tmp_path / "audit.jsonl"
    audit = AuditLog(audit_path)
    settings = LiveSettings()
    assert settings.reject_cluster_alert_min_symbols == 3
    outcomes = [_rejected("AAAUSDT", -1013), _rejected("BBBUSDT", -1013), _rejected("CCCUSDT", -1013)]
    _alert_reject_cluster(settings, outcomes, audit,
                          decision_time=pd.Timestamp("2026-09-14", tz="UTC"),
                          now=pd.Timestamp("2026-09-14 01:00", tz="UTC"))
    events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    clusters = [e for e in events if e["event"] == "intent_reject_cluster"]
    assert len(clusters) == 1
    assert clusters[0]["code"] == -1013
    assert clusters[0]["symbols"] == 3


def test_scattered_rejections_do_not_alert(tmp_path) -> None:
    """Spec 02: rejections spread across codes below threshold stay silent."""
    from decimal import Decimal

    import pandas as pd

    from src.live.audit import AuditLog
    from src.live.executor import ExecutionOutcome
    from src.live.runner import _alert_reject_cluster
    from src.live.settings import LiveSettings

    def _rejected(symbol, code):
        return ExecutionOutcome(symbol=symbol, filled_qty=Decimal(0), unfilled_qty=Decimal(1),
                                avg_fill_price=None, chases=0, status="REJECTED", reject_code=code)

    audit_path = tmp_path / "audit.jsonl"
    audit = AuditLog(audit_path)
    outcomes = [_rejected("AAAUSDT", -1013), _rejected("BBBUSDT", -4164), _rejected("CCCUSDT", -4164)]
    _alert_reject_cluster(LiveSettings(), outcomes, audit,
                          decision_time=pd.Timestamp("2026-09-14", tz="UTC"),
                          now=pd.Timestamp("2026-09-14 01:00", tz="UTC"))
    import json

    lines = audit_path.read_text(encoding="utf-8").splitlines() if audit_path.exists() else []
    events = [json.loads(line) for line in lines]
    assert [e for e in events if e["event"] == "intent_reject_cluster"] == []


def test_sizing_mark_fallbacks_audited(tmp_path) -> None:
    """Spec 02: nonzero held symbols absent from live marks are audited once each."""
    import json
    from decimal import Decimal

    from src.live.audit import AuditLog
    from src.live.runner import _audit_sizing_mark_fallbacks

    audit_path = tmp_path / "audit.jsonl"
    audit = AuditLog(audit_path)
    _audit_sizing_mark_fallbacks(
        audit,
        {"AAAUSDT": Decimal("5"), "BBBUSDT": Decimal("0"), "CCCUSDT": Decimal("-2")},
        {"AAAUSDT": Decimal("100")},
        {"CCCUSDT": Decimal("99")},
    )
    events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    fallbacks = [e for e in events if e["event"] == "sizing_mark_fallback"]
    assert [(e["symbol"], e["has_fallback"]) for e in fallbacks] == [("CCCUSDT", True)]


def test_risk_blocked_intents_audited(tmp_path) -> None:
    """Spec 02: withheld risk-increasing intents leave one audit row each."""
    import json
    from decimal import Decimal

    from src.live.audit import AuditLog
    from src.live.planner import OrderIntent
    from src.live.runner import _audit_risk_blocked

    audit_path = tmp_path / "audit.jsonl"
    audit = AuditLog(audit_path)
    _audit_risk_blocked(audit, [])
    assert not audit_path.exists()
    blocked = [
        OrderIntent(symbol="AAAUSDT", side="BUY", quantity=Decimal("1"), reduce_only=False,
                    target_qty=Decimal("1"), current_qty=Decimal("0"),
                    client_order_prefix="run1", leg_index=1, decision_price=Decimal("100")),
    ]
    _audit_risk_blocked(audit, blocked)
    events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert [(e["event"], e["symbol"]) for e in events] == [("intent_risk_control_blocked", "AAAUSDT")]


def test_reject_cluster_alerts_on_abort_path(tmp_path, monkeypatch) -> None:
    """Spec 02: the abort path still emits the cluster alert once outcomes persist."""
    import json

    import pandas as pd

    import src.live.runner as runner_mod
    from src.live.audit import AuditLog  # noqa: F401
    from src.live.errors import LiveTradingError
    from src.live.executor import ExecutionOutcome
    from src.live.runner import run_shadow_cycle
    from src.live.settings import LiveSettings
    from tests.unit.live._runner_stubs import DECISION_TIME, NOW, StubMarketClient, StubOrderClient

    frame = pd.DataFrame(
        {"AAAUSDT": [0.02], "BUSDT": [-0.02]},
        index=pd.DatetimeIndex([DECISION_TIME]),
    )
    weights_path = tmp_path / "deployed_target_weights_abort.parquet"
    frame.to_parquet(weights_path, index=True)
    from src.live.deployed_weights import decision_ohlcv_close_path

    closes = pd.DataFrame(
        100.0, index=pd.DatetimeIndex(frame.index), columns=list(frame.columns), dtype="float64",
    )
    closes.to_parquet(decision_ohlcv_close_path(weights_path), index=True)

    monkeypatch.setattr(
        runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient()
    )
    monkeypatch.setattr(
        runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient()
    )
    monkeypatch.setattr(
        runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl"
    )
    import src.live.orderbook as ob_mod

    monkeypatch.setattr(ob_mod, "capture_order_books", lambda *a, **k: [])
    monkeypatch.setattr(ob_mod, "append_order_book_snapshots", lambda *a, **k: [])

    from decimal import Decimal

    def fake_raise(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, outcome_sink=None, **kwargs):
        outcomes = [
            ExecutionOutcome(symbol=i.symbol, filled_qty=Decimal(0), unfilled_qty=i.quantity,
                             avg_fill_price=None, chases=0, status="REJECTED", reject_code=-1013)
            for i in intents
        ]
        if outcome_sink is not None:
            outcome_sink[:] = outcomes
        exc = LiveTradingError("boom")
        exc.partial_outcomes = tuple(outcomes)
        raise exc

    monkeypatch.setattr(runner_mod, "execute_intents", fake_raise)
    settings = LiveSettings(
        notional_equity_usdt=2000.0,
        ledger_path=str(tmp_path / "ledger_abort.json"),
        order_journal_path=str(tmp_path / "order_journal.jsonl"),
        fills_dir=str(tmp_path / "fills"),
        tax_ledger_dir=str(tmp_path / "tax"),
        execution_quality_dir=str(tmp_path / "eq"),
        portfolio_state_dir=str(tmp_path / "portfolio"),
        reject_cluster_alert_min_symbols=2,
    )
    report = run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)
    assert report.status == "HALT"
    audit_path = tmp_path / "shadow_cycle.jsonl"
    assert audit_path.exists()
    events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    clusters = [e for e in events if e["event"] == "intent_reject_cluster"]
    assert len(clusters) == 1
    assert clusters[0]["code"] == -1013


def test_corrupt_funding_shard_alerts_and_halts(tmp_path, monkeypatch) -> None:
    """PAPER funding shard with mid-file corruption: HALT + one tax_ledger_corrupt dispatch."""
    import json
    from decimal import Decimal

    import pandas as pd

    import src.live.runner as runner_mod
    from src.live.ledger import LedgerState, PositionSnapshot, save_ledger
    from src.live.settings import LiveSettings
    from src.live.tax_ledger import _tax_shard_path
    from tests.unit.live._runner_stubs import DECISION_TIME, NOW, StubMarketClient, StubOrderClient

    path, _, _ = _seed_policy_cycle_artifact(tmp_path)
    captured: dict = {}
    _install_policy_cycle_stubs(tmp_path, monkeypatch, captured)
    monkeypatch.setattr(runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient())
    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient())

    t0 = NOW - pd.Timedelta(days=2)
    t1 = t0 + pd.Timedelta(hours=8)
    funding = pd.Series([0.0001], index=pd.DatetimeIndex([t1]))
    closes = pd.Series([100.0], index=pd.DatetimeIndex([t1.floor("h")]))
    monkeypatch.setattr(runner_mod, "_load_paper_funding", lambda symbols: {"AAAUSDT": funding})
    monkeypatch.setattr(runner_mod, "_load_paper_trade_closes", lambda symbols: {"AAAUSDT": closes})

    tax_dir = tmp_path / "tax"
    tax_dir.mkdir(parents=True, exist_ok=True)
    shard = _tax_shard_path(tax_dir, t1)
    row = {"record_id": "seed:1", "kind": "FUNDING_FEE", "event_time": t0.isoformat(),
           "symbol": "AAAUSDT", "side": "", "quantity": 1.0, "price": 100.0, "quote_qty": 100.0,
           "fee": 0.0, "fee_asset": "USDT", "realized_pnl": 0.1, "income_asset": "USDT",
           "is_maker": False, "venue_id": 0, "source": "simulated", "mode": "paper", "income_type": ""}
    shard.write_text(json.dumps(row) + "\n" + "{bad}\n" + json.dumps(row) + "\n", encoding="utf-8")

    ledger_path = tmp_path / "ledger_corrupt.json"
    save_ledger(
        ledger_path,
        LedgerState(
            positions={"AAAUSDT": Decimal("1")},
            equity_high_water_mark=Decimal("2000"),
            cash_usdt=Decimal("2000"),
            funding_watermarks={"AAAUSDT": t0},
            position_history=(PositionSnapshot(effective_from=t0, positions={"AAAUSDT": Decimal("1")}),),
        ),
    )
    settings = LiveSettings(
        mode="paper",
        ledger_path=str(ledger_path),
        order_journal_path=str(tmp_path / "journal.jsonl"),
        fills_dir=str(tmp_path / "fills"),
        tax_ledger_dir=str(tax_dir),
        execution_quality_dir=str(tmp_path / "eq"),
        portfolio_state_dir=str(tmp_path / "port"),
    )
    calls: list = []

    def _fake_dispatch(settings, *, event, detail, decision_time, dedupe_key, now):
        calls.append({"event": event, "detail": detail, "dedupe_key": dedupe_key})
        return True

    monkeypatch.setattr(runner_mod, "dispatch_alert", _fake_dispatch)
    report = runner_mod.run_shadow_cycle(settings, DECISION_TIME, path, now=NOW)
    assert report.status == "HALT"
    assert len(calls) == 1
    assert calls[0]["event"] == "tax_ledger_corrupt"
    assert "line=2" in calls[0]["detail"]


def test_live_collection_receives_wall_clock_now(tmp_path, monkeypatch) -> None:
    """LIVE collection gets now=now_ts, not decision_time."""
    import src.live.runner as runner_mod

    runner_mod, settings, path, decision_time, now = _seed_live_tax_cycle(tmp_path, monkeypatch)
    seen: dict = {}

    def _fake_collect(client, symbols, tax_dir, mode, *, now, settings=None):
        seen["now"] = now
        return 0, ()

    monkeypatch.setattr(runner_mod, "collect_and_persist_live_tax", _fake_collect)
    report = runner_mod.run_shadow_cycle(settings, decision_time, path, now=now)
    assert report.status == "COMPLETE"
    assert seen["now"] == now
    assert seen["now"] != decision_time


def test_live_tax_retention_gap_alerts(tmp_path, monkeypatch) -> None:
    """A retention_gap collection issue (venue history already expired) is alerted, not only audited."""
    from src.live.tax_ledger import TaxCollectionIssue

    runner_mod, settings, path, decision_time, now = _seed_live_tax_cycle(tmp_path, monkeypatch)
    alerts: list = []

    def _fake_collect(client, symbols, tax_dir, mode, *, now, settings=None):
        return 0, (TaxCollectionIssue(stream="income", stage="retention_gap", detail="uncovered [a..b]"),)

    monkeypatch.setattr(runner_mod, "collect_and_persist_live_tax", _fake_collect)
    monkeypatch.setattr(runner_mod, "dispatch_alert", lambda settings, **kw: alerts.append(kw))

    report = runner_mod.run_shadow_cycle(settings, decision_time, path, now=now)

    assert report.status == "COMPLETE"
    gap = [a for a in alerts if a["event"] == "tax_income_gap"]
    assert len(gap) == 1
    assert gap[0]["detail"] == "uncovered [a..b]"
