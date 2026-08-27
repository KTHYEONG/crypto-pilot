"""SCENARIO_LIVE_38/39: preflight 게이트는 GET-only이며 6개 점검을 모두 수행하고
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
            return {
                "totalWalletBalance": "2000",
                "availableBalance": "1900",
                "totalInitialMargin": "10",
                "totalUnrealizedProfit": "0",
                "dualSidePosition": "false",
                "multiAssetsMargin": "false",
            }
        if path == "/fapi/v2/positionRisk":
            return []
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
    assert len(report.checks) == 6
    assert tuple(c.name for c in report.checks) == _EXPECTED_CHECK_NAMES
    assert all(c.passed is False for c in report.checks)
    assert mutation_calls == []
    assert not ledger_path.exists()


#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_38_PREFLIGHT_FLAGS_STALE_ARTIFACT",
    "SCENARIO_LIVE_39_PREFLIGHT_AGGREGATES_ALL_CHECKS_WITHOUT_RAISING",
)
