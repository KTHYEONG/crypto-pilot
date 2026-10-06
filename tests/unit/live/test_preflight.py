"""SCENARIO_LIVE_38/39: preflight 게이트는 GET-only이며 8개 점검을 모두 수행하고
어떤 개별 실패에도 예외를 던지지 않는다(I-PREFLIGHT-TOTAL)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.live.preflight import run_preflight
from src.live.settings import LiveSettings

_EXPECTED_CHECK_NAMES = (
    "artifact_readable",
    "artifact_covers_decision_time",
    "venue_exchange_info",
    "venue_rate_limits",
    "account_configuration",
    "position_reconciliation",
    "venue_leverage_plan",
    "tax_collection_ready",
)


class StubMarketClient:
    def exchange_info(self) -> dict[str, Any]:
        return {
            "symbols": [],
            "rateLimits": [
                {"rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "intervalNum": 1, "limit": 2400},
                {"rateLimitType": "ORDERS", "interval": "MINUTE", "intervalNum": 1, "limit": 1200},
                {"rateLimitType": "ORDERS", "interval": "SECOND", "intervalNum": 10, "limit": 300},
            ],
        }


class StubOrderClient:
    def request(self, method: str, path: str, params=None, *, signed=False) -> Any:
        if path == "/fapi/v2/account":
            # Real Binance omits dualSidePosition from this payload.
            return {
                "totalWalletBalance": "2000",
                "availableBalance": "1900",
                "totalInitialMargin": "10",
                "totalUnrealizedProfit": "0",
                "multiAssetsMargin": "false",
            }
        if path == "/fapi/v2/positionRisk":
            return []
        if path == "/fapi/v1/positionSide/dual":
            return {"dualSidePosition": False}
        raise AssertionError(f"unexpected path {path}")


class StubAllFailClient:
    """모든 호출이 OSError로 실패하며, 변이 메서드 호출 여부를 기록한다."""

    def __init__(self, mutation_calls: list[str]) -> None:
        self._mutation_calls = mutation_calls

    def exchange_info(self) -> Any:
        raise OSError("network unreachable")

    def request(self, method: str, path: str, params=None, *, signed=False) -> Any:
        raise OSError("network unreachable")

    def sync_server_time(self) -> None:
        raise OSError("network unreachable")

    def open_orders(self) -> list[Any]:
        raise OSError("network unreachable")

    def new_order(self, params: Any) -> Any:
        self._mutation_calls.append("new_order")
        raise AssertionError("preflight must never place an order")

    def cancel_order(self, *args: Any, **kwargs: Any) -> Any:
        self._mutation_calls.append("cancel_order")
        raise AssertionError("preflight must never cancel an order")


def _write_artifact(path, index: pd.DatetimeIndex) -> None:
    frame = pd.DataFrame({"AAAUSDT": [0.02] * len(index)}, index=index)
    frame.to_parquet(path, index=True)


def test_SCENARIO_LIVE_38_PREFLIGHT_FLAGS_STALE_ARTIFACT(tmp_path) -> None:
    now = pd.Timestamp("2026-08-27 00:00Z")
    stale_ts = now.normalize() - pd.Timedelta(days=30)

    stale_artifact = tmp_path / "stale.parquet"
    _write_artifact(stale_artifact, pd.DatetimeIndex([stale_ts]))

    settings = LiveSettings(mode="shadow", ledger_path=str(tmp_path / "ledger.json"))
    report = run_preflight(
        settings,
        stale_artifact,
        now=now,
        market_client=StubMarketClient(),
        order_client=StubOrderClient(),
    )
    by_name = {c.name: c for c in report.checks}
    assert by_name["artifact_readable"].passed is True
    coverage = by_name["artifact_covers_decision_time"]
    assert coverage.passed is False
    assert "staleness_hours=720.0" in coverage.detail

    fresh_artifact = tmp_path / "fresh.parquet"
    _write_artifact(fresh_artifact, pd.DatetimeIndex([now.normalize()]))
    report2 = run_preflight(
        settings,
        fresh_artifact,
        now=now,
        market_client=StubMarketClient(),
        order_client=StubOrderClient(),
    )
    coverage2 = {c.name: c for c in report2.checks}["artifact_covers_decision_time"]
    assert coverage2.passed is True
    assert "staleness_hours=0.0" in coverage2.detail


def test_SCENARIO_LIVE_39_PREFLIGHT_AGGREGATES_ALL_CHECKS_WITHOUT_RAISING(tmp_path) -> None:
    missing_artifact = tmp_path / "does_not_exist.parquet"
    ledger_path = tmp_path / "state" / "ledger.json"
    settings = LiveSettings(mode="shadow", ledger_path=str(ledger_path))

    mutation_calls: list[str] = []
    market_client = StubAllFailClient(mutation_calls)
    order_client = StubAllFailClient(mutation_calls)

    report = run_preflight(
        settings,
        missing_artifact,
        market_client=market_client,
        order_client=order_client,
    )

    assert report.passed is False
    assert len(report.checks) == 8
    assert tuple(c.name for c in report.checks) == (
        "artifact_readable",
        "artifact_covers_decision_time",
        "venue_exchange_info",
        "venue_rate_limits",
        "account_configuration",
        "position_reconciliation",
        "venue_leverage_plan",
        "tax_collection_ready",
    )
    assert all(c.passed is False for c in report.checks if c.name != "tax_collection_ready")
    assert {c.name: c for c in report.checks}["tax_collection_ready"].detail == (
        "suppressed mode: simulated tax records"
    )
    assert mutation_calls == []
    assert not ledger_path.exists()


def test_preflight_venue_leverage_plan_reports_pending_changes_in_live_mode(tmp_path) -> None:
    class LeverageAwareClient(StubOrderClient):
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def request(self, method, path, params=None, *, signed=False):
            self.calls.append((method, path))
            if path == "/fapi/v1/leverageBracket":
                return [{"symbol": "AAAUSDT", "brackets": [
                    {"bracket": 1, "initialLeverage": 20, "notionalCap": 50000, "notionalFloor": 0, "maintMarginRatio": 0.01, "cum": 0},
                ]}]
            if path == "/fapi/v2/positionRisk":
                return [{"symbol": "AAAUSDT", "positionAmt": "0", "marginType": "isolated", "leverage": "20"}]
            return super().request(method, path, params, signed=signed)
    now = pd.Timestamp("2026-08-27 00:00Z")
    artifact = tmp_path / "weights.parquet"
    _write_artifact(artifact, pd.DatetimeIndex([now.normalize()]))
    settings = LiveSettings(mode="live_testnet", ledger_path=str(tmp_path / "ledger.json"))
    client = LeverageAwareClient()

    report = run_preflight(settings, artifact, now=now, market_client=StubMarketClient(), order_client=client)

    by_name = {c.name: c for c in report.checks}
    assert by_name["venue_leverage_plan"].passed is True
    assert by_name["venue_leverage_plan"].detail == "symbols=1 margin_type_changes=1 leverage_changes=1"
    assert {method for method, _ in client.calls} == {"GET"}
    assert ("GET", "/fapi/v1/leverageBracket") in client.calls


def test_preflight_venue_leverage_plan_skips_venue_in_suppressed_mode(tmp_path) -> None:
    now = pd.Timestamp("2026-08-27 00:00Z")
    artifact = tmp_path / "weights.parquet"
    _write_artifact(artifact, pd.DatetimeIndex([now.normalize()]))
    settings = LiveSettings(mode="shadow", ledger_path=str(tmp_path / "ledger.json"))

    report = run_preflight(settings, artifact, now=now, market_client=StubMarketClient(), order_client=StubOrderClient())

    by_name = {c.name: c for c in report.checks}
    assert by_name["venue_leverage_plan"].passed is True
    assert by_name["venue_leverage_plan"].detail == "suppressed mode: venue leverage untouched"


def test_preflight_venue_leverage_plan_fails_without_artifact_in_live_mode(tmp_path) -> None:
    class LeverageAwareClient(StubOrderClient):
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def request(self, method, path, params=None, *, signed=False):
            self.calls.append((method, path))
            if path == "/fapi/v1/leverageBracket":
                return [{"symbol": "AAAUSDT", "brackets": [
                    {"bracket": 1, "initialLeverage": 20, "notionalCap": 50000, "notionalFloor": 0, "maintMarginRatio": 0.01, "cum": 0},
                ]}]
            if path == "/fapi/v2/positionRisk":
                return [{"symbol": "AAAUSDT", "positionAmt": "0", "marginType": "isolated", "leverage": "20"}]
            return super().request(method, path, params, signed=signed)
    now = pd.Timestamp("2026-08-27 00:00Z")
    settings = LiveSettings(mode="live_testnet", ledger_path=str(tmp_path / "ledger.json"))

    report = run_preflight(
        settings, tmp_path / "missing.parquet", now=now,
        market_client=StubMarketClient(), order_client=LeverageAwareClient(),
    )

    by_name = {c.name: c for c in report.checks}
    assert by_name["account_configuration"].passed is True
    assert by_name["venue_leverage_plan"].passed is False
    assert "missing artifact" in by_name["venue_leverage_plan"].detail


def _live_settings_for_tax_gate(tmp_path, mode, *, enabled=True):
    kwargs = {"mode": mode, "ledger_path": str(tmp_path / "ledger.json"), "tax_collection_enabled": enabled}
    if mode in ("live_testnet", "live_mainnet"):
        kwargs.update(order_api_key="k", order_api_secret="s")  # noqa: S106 - hermetic test credential
    if mode == "live_mainnet":
        from src.live.settings import MAINNET_TRADING_ACK

        kwargs["mainnet_trading_ack"] = MAINNET_TRADING_ACK
    return LiveSettings(**kwargs)


def test_tax_collection_check_mainnet_fails_without_collection(tmp_path) -> None:
    from src.live.preflight import run_preflight, tax_collection_check

    settings = _live_settings_for_tax_gate(tmp_path, "live_mainnet", enabled=False)
    check = tax_collection_check(settings, tmp_path / "tax")
    assert check.name == "tax_collection_ready"
    assert check.passed is False
    assert check.detail.startswith("[RISK] live_mainnet requires LIVE_TAX_COLLECTION_ENABLED=true")

    now = pd.Timestamp("2026-08-27 00:00Z")
    artifact = tmp_path / "weights.parquet"
    _write_artifact(artifact, pd.DatetimeIndex([now.normalize()]))
    report = run_preflight(
        settings, artifact, now=now,
        market_client=StubMarketClient(), order_client=StubOrderClient(),
    )
    assert report.passed is False
    assert any(c.name == "tax_collection_ready" and not c.passed for c in report.checks)


def test_tax_collection_check_testnet_warns_only(tmp_path, caplog) -> None:
    import logging
    from src.live.preflight import run_preflight, tax_collection_check

    settings = _live_settings_for_tax_gate(tmp_path, "live_testnet", enabled=False)
    check = tax_collection_check(settings, tmp_path / "tax")
    assert check.passed is True
    assert check.detail.startswith("WARNING:")

    now = pd.Timestamp("2026-08-27 00:00Z")
    artifact = tmp_path / "weights.parquet"
    _write_artifact(artifact, pd.DatetimeIndex([now.normalize()]))
    with caplog.at_level(logging.WARNING, logger="LivePreflight"):
        report = run_preflight(
            settings, artifact, now=now,
            market_client=StubMarketClient(), order_client=StubOrderClient(),
        )
    by_name = {c.name: c for c in report.checks}
    assert by_name["tax_collection_ready"].passed is True
    assert by_name["tax_collection_ready"].detail.startswith("WARNING:")
    assert "[RISK]" in caplog.text


def test_tax_collection_check_enabled_and_suppressed_pass(tmp_path) -> None:
    from src.live.preflight import tax_collection_check

    live = _live_settings_for_tax_gate(tmp_path, "live_mainnet", enabled=True)
    assert tax_collection_check(live, tmp_path / "tax").passed is True
    paper = _live_settings_for_tax_gate(tmp_path, "paper", enabled=False)
    check = tax_collection_check(paper, tmp_path / "tax")
    assert check.passed is True
    assert check.detail == "suppressed mode: simulated tax records"


def test_preflight_reports_account_scoped_dir(tmp_path, monkeypatch) -> None:
    import src.live.tax_ledger as tax_mod
    from src.live.preflight import run_preflight

    venue_root = tmp_path / "venue"
    monkeypatch.setattr(tax_mod, "default_venue_tax_ledger_root", lambda: venue_root)
    settings = _live_settings_for_tax_gate(tmp_path, "live_testnet", enabled=True)
    settings.record_run_id = "run_a_12345678"
    now = pd.Timestamp("2026-08-27 00:00Z")
    artifact = tmp_path / "weights.parquet"
    _write_artifact(artifact, pd.DatetimeIndex([now.normalize()]))
    report = run_preflight(
        settings, artifact, now=now,
        market_client=StubMarketClient(), order_client=StubOrderClient(),
    )
    check = {c.name: c for c in report.checks}["tax_collection_ready"]
    assert check.detail.endswith(f"tax_dir={venue_root / 'testnet'}")
    assert "runs" not in check.detail

