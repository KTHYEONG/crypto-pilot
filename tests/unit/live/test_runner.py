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
    settings = LiveSettings(mode=ExecutionMode.LIVE_TESTNET)
    decision = pd.Timestamp("2026-09-15 00:00Z")

    runner_mod._market_client(settings, decision)
    runner_mod._order_client(settings, decision)

    # Then: 필터/호가 조회와 주문이 같은 테스트넷 베뉴
    assert captured == [TESTNET_FAPI_URL, TESTNET_FAPI_URL]
