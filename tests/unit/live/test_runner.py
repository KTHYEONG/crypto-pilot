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
