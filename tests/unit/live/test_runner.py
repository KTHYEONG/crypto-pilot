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


def _install_policy_cycle_stubs(tmp_path, monkeypatch, captured: dict) -> None:
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

        captured["policy"] = policy
        return tuple(
            ExecutionOutcome(
                symbol=intent.symbol,
                filled_qty=intent.quantity,
                unfilled_qty=Decimal("0"),
                avg_fill_price=Decimal("100"),
                chases=0,
                status="FILLED",
            )
            for intent in intents
        )

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
    journal.record_submit("wired-cid", "AAAUSDT", journal.next_submit_seq())
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
