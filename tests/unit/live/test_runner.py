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
