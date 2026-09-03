# ruff: noqa
"""Live runner tests - shadow_cycle."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import src.live.runner as runner_mod
from src.live.account import (
    AccountSnapshot,
    assert_drawdown_within_limit,
    assert_suppressed_venue_flat,
    resolve_sizing_equity,
)
from src.live.errors import ReconciliationBreach, RiskGateBreach, VenueError
from src.live.executor import ExecutionOutcome
from src.live.ledger import LedgerState, load_ledger, save_ledger
from src.live.runner import check_risk_gates, run_shadow_cycle
from src.live.settings import LiveSettings

from tests.unit.live._runner_stubs import DECISION_TIME, NOW, StubMarketClient, StubOrderClient

@pytest.fixture(autouse=True)
def _maybe_disable_orderbook_capture(monkeypatch, request):  # noqa: ARG001
    if "captures_orderbook" in request.node.name:
        return
    import src.live.orderbook as ob_mod

    monkeypatch.setattr(ob_mod, "capture_order_books", lambda *a, **k: [])
    monkeypatch.setattr(ob_mod, "append_order_book_snapshots", lambda *a, **k: [])

@pytest.fixture
def artifact(tmp_path):
    frame = pd.DataFrame(
        {"AAAUSDT": [0.02], "BUSDT": [-0.02]},
        index=pd.DatetimeIndex([DECISION_TIME]),
    )
    path = tmp_path / "deployed_target_weights.parquet"
    frame.to_parquet(path, index=True)
    return path

@pytest.fixture
def live_env(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient()
    )
    monkeypatch.setattr(
        runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient()
    )
    calls: list[Any] = []

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        calls.extend(intents)
        outcomes = tuple(
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
        audit.record("intents_executed", count=len(outcomes))
        return outcomes

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)
    monkeypatch.setattr(
        runner_mod,
        "default_audit_log_path",
        lambda name, for_date=None: tmp_path / f"{name}.jsonl",
    )
    import src.live.orderbook as ob_mod

    monkeypatch.setattr(ob_mod, "capture_order_books", lambda *a, **k: [])
    monkeypatch.setattr(ob_mod, "append_order_book_snapshots", lambda *a, **k: [])
    return calls

def test_SCENARIO_LIVE_29_CYCLE_REPORTS_MIN_NOTIONAL_DROP(artifact, live_env, tmp_path) -> None:
    """SCENARIO_LIVE_29_CYCLE_REPORTS_MIN_NOTIONAL_DROP: at equity $2,000,
    three symbols whose weights place them below minNotional surface in
    CycleReport.dropped_notional_fraction as dropped target notional over
    total target notional (strictly inside (0, 1)); a cycle with no drops
    reports exactly 0.0."""

    class FiveSymbolMarketClient(StubMarketClient):
        def exchange_info(self) -> dict[str, Any]:
            payload = super().exchange_info()
            template = payload["symbols"][0]
            for symbol in ("DRP1USDT", "DRP2USDT", "DRP3USDT"):
                extra = dict(template)
                extra["symbol"] = symbol
                payload["symbols"].append(extra)
            return payload

    monkeypatch_market = FiveSymbolMarketClient()
    original_market = runner_mod._market_client
    runner_mod._market_client = lambda settings, decision_time: monkeypatch_market  # type: ignore[assignment, misc]
    try:
        weights = pd.DataFrame(
            {
                "AAAUSDT": [0.02],
                "BUSDT": [-0.02],
                "DRP1USDT": [0.0001],
                "DRP2USDT": [0.0001],
                "DRP3USDT": [0.0001],
            },
            index=pd.DatetimeIndex([DECISION_TIME]),
        )
        drop_path = artifact.parent / "with_drops.parquet"
        weights.to_parquet(drop_path, index=True)

        settings = LiveSettings(
            notional_equity_usdt=2000.0, ledger_path=str(tmp_path / "ledger_drop.json"),
        )
        report = run_shadow_cycle(settings, DECISION_TIME, drop_path, now=NOW)
    finally:
        runner_mod._market_client = original_market

    assert report.status == "COMPLETE"
    # 분자: 3 * $2000 * 0.0001 = $0.6, 분모: 유지 노셔널 $80 + 드롭 $0.6.
    from decimal import Decimal as _D

    dropped = _D("0.6")
    total = _D("80") + dropped
    expected = float(dropped / total)
    assert 0.0 < report.dropped_notional_fraction < 1.0
    assert report.dropped_notional_fraction == pytest.approx(expected, rel=1e-9)

    no_drop_report = run_shadow_cycle(
        LiveSettings(notional_equity_usdt=2000.0, ledger_path=str(tmp_path / "ledger_clean.json")),
        DECISION_TIME,
        artifact,
        now=NOW,
    )
    assert no_drop_report.status == "COMPLETE"
    assert no_drop_report.dropped_notional_fraction == 0.0



def test_SCENARIO_LIVE_RUNNER_WRITES_EXECUTION_QUALITY_AND_NEVER_HALTS_ON_ITS_FAILURE(
    artifact, monkeypatch, tmp_path
) -> None:
    """Run shadow cycle writes execution quality and never halts on its failure."""
    import pandas as pd

    monkeypatch.setattr(
        runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient()
    )
    monkeypatch.setattr(
        runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient()
    )
    monkeypatch.setattr(
        runner_mod,
        "default_audit_log_path",
        lambda name, for_date=None: tmp_path / f"{name}.jsonl",
    )

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
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

    eq_dir = tmp_path / "eq_quality"
    ledger_path = tmp_path / "ledger_runner_eq.json"
    settings = LiveSettings(
        mode="paper",
        notional_equity_usdt=2000.0,
        ledger_path=str(ledger_path),
        execution_quality_dir=str(eq_dir),
    )
    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)
    assert report.status == "COMPLETE"
    # At least one record persisted with mark
    shards = list(eq_dir.glob("*.parquet"))
    assert len(shards) >= 1
    df = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    assert len(df) >= 1
    # mark_price_at_decision populated (proving runner marks now reach storage)
    assert df["mark_price_at_decision"].notna().any()

    # Monkeypatch append to raise OSError still returns COMPLETE and ledger written
    def raise_oserror(records, history_dir):
        raise OSError("disk full")

    monkeypatch.setattr(runner_mod, "append_execution_quality", raise_oserror)
    ledger_path2 = tmp_path / "ledger_runner_eq2.json"
    settings2 = LiveSettings(
        mode="paper",
        notional_equity_usdt=2000.0,
        ledger_path=str(ledger_path2),
        execution_quality_dir=str(tmp_path / "eq_quality2"),
    )
    report2 = run_shadow_cycle(settings2, DECISION_TIME, artifact, now=NOW)
    assert report2.status == "COMPLETE"
    assert ledger_path2.exists()



def test_SCENARIO_LIVE_35_PAPER_MULTI_DAY_CYCLES_DO_NOT_HALT(tmp_path, monkeypatch) -> None:
    """SCENARIO_LIVE_35_PAPER_MULTI_DAY_CYCLES_DO_NOT_HALT: PAPER cycles on
    unchanged target weights never HALT, and settle to 0 intents from the
    second cycle onward once the internal ledger already holds the target."""
    monkeypatch.setattr(
        runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient()
    )
    monkeypatch.setattr(
        runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient()
    )
    monkeypatch.setattr(
        runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl"
    )

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        return tuple(
            ExecutionOutcome(
                symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"),
                avg_fill_price=Decimal("100"), chases=0, status="FILLED",
            )
            for i in intents
        )

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)

    days = [DECISION_TIME + pd.Timedelta(days=i) for i in range(3)]
    weights = pd.DataFrame(
        {"AAAUSDT": [0.02] * 3, "BUSDT": [-0.02] * 3}, index=pd.DatetimeIndex(days)
    )
    weights_path = tmp_path / "paper_weights.parquet"
    weights.to_parquet(weights_path, index=True)

    ledger_path = tmp_path / "ledger_paper_multi.json"
    settings = LiveSettings(mode="paper", notional_equity_usdt=2000.0, ledger_path=str(ledger_path))

    reports = []
    ledger_snapshots = []
    for day in days:
        report = run_shadow_cycle(settings, day, weights_path, now=day + pd.Timedelta(hours=2))
        reports.append(report)
        ledger_snapshots.append(dict(load_ledger(ledger_path).positions))

    assert [r.status for r in reports] == ["COMPLETE", "COMPLETE", "COMPLETE"]
    assert reports[0].intent_count > 0
    assert reports[1].intent_count == 0
    assert reports[2].intent_count == 0
    assert ledger_snapshots[0] == ledger_snapshots[2]



def test_SCENARIO_LIVE_47_RUNNER_PERSISTS_PORTFOLIO_STATE_PAPER_VS_LIVE(
    tmp_path, monkeypatch
) -> None:
    """SCENARIO_LIVE_47: PAPER writes a virtual_mtm row with cash populated;
    live_testnet writes a venue row with wallet_balance populated; a write
    failure never changes the cycle status."""
    monkeypatch.setattr(
        runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient()
    )
    monkeypatch.setattr(
        runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl"
    )

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        return tuple(
            ExecutionOutcome(
                symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"),
                avg_fill_price=Decimal("100"), chases=0, status="FILLED",
            )
            for i in intents
        )

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)

    weights = pd.DataFrame(
        {"AAAUSDT": [0.02], "BUSDT": [-0.02]}, index=pd.DatetimeIndex([DECISION_TIME])
    )
    weights_path = tmp_path / "weights.parquet"
    weights.to_parquet(weights_path, index=True)

    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient())
    paper_dir = tmp_path / "portfolio_paper"
    paper_settings = LiveSettings(
        mode="paper", notional_equity_usdt=2000.0,
        ledger_path=str(tmp_path / "ledger_paper.json"),
        portfolio_state_dir=str(paper_dir),
    )
    paper_report = run_shadow_cycle(paper_settings, DECISION_TIME, weights_path, now=NOW)
    assert paper_report.status == "COMPLETE"
    paper_df = pd.read_parquet(paper_dir / "active.parquet")
    assert (paper_df["equity_source"] == "virtual_mtm").all()
    assert paper_df["cash_usdt"].notna().all()

    class VenueMatchClient(StubOrderClient):
        def request(self, method, path, params=None, *, signed=False):
            if path == "/fapi/v2/positionRisk":
                return [{"symbol": "AAAUSDT", "positionAmt": "0.4"}]
            return super().request(method, path, params, signed=signed)

    save_ledger(
        tmp_path / "ledger_live.json",
        LedgerState(positions={"AAAUSDT": Decimal("0.4")}, equity_high_water_mark=Decimal(0)),
    )
    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: VenueMatchClient())
    live_dir = tmp_path / "portfolio_live"
    live_settings = LiveSettings(
        mode="live_testnet", notional_equity_usdt=2000.0,
        ledger_path=str(tmp_path / "ledger_live.json"),
        portfolio_state_dir=str(live_dir),
    )
    live_report = run_shadow_cycle(live_settings, DECISION_TIME, weights_path, now=NOW)
    assert live_report.status == "COMPLETE"
    live_df = pd.read_parquet(live_dir / "active.parquet")
    assert (live_df["equity_source"] == "venue").all()
    assert live_df["wallet_balance_usdt"].notna().all()
    assert live_df["cash_usdt"].isna().all()

    def raise_oserror(record, history_dir):
        raise OSError("disk full")

    monkeypatch.setattr(runner_mod, "append_portfolio_state", raise_oserror)
    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient())
    fail_settings = LiveSettings(
        mode="paper", notional_equity_usdt=2000.0,
        ledger_path=str(tmp_path / "ledger_paper2.json"),
        portfolio_state_dir=str(tmp_path / "portfolio_fail"),
    )
    fail_report = run_shadow_cycle(fail_settings, DECISION_TIME, weights_path, now=NOW)
    assert fail_report.status == "COMPLETE"



def test_SCENARIO_LIVE_49_MARK_FETCH_FALLBACK_TOLERATES_UNKNOWN_SYMBOL() -> None:
    """SCENARIO_LIVE_49: _marks_from_tickers' per-symbol fallback skips a
    symbol whose lookup fails and still returns marks for the rest."""
    from src.live.runner import _marks_from_tickers

    class PartialFailClient:
        def book_ticker(self, symbol: str) -> dict[str, str]:
            if symbol == "DEADUSDT":
                raise AssertionError("unknown symbol")
            return {"bidPrice": "100.00", "askPrice": "101.00"}

    marks = _marks_from_tickers(PartialFailClient(), ["AAAUSDT", "DEADUSDT", "BUSDT"])
    assert set(marks) == {"AAAUSDT", "BUSDT"}
    assert marks["AAAUSDT"] == Decimal("100.50")



def test_SCENARIO_PARITY_09_runner_wiring_and_failsoft(tmp_path, monkeypatch):
    """SCENARIO_PARITY_09-runner-wiring-and-failsoft"""
    import pandas as pd
    
    from decimal import Decimal
    from src.live.runner import run_shadow_cycle
    from src.live.settings import LiveSettings
    import src.live.runner as runner_mod
    from src.live.executor import ExecutionOutcome

    # Setup artifact
    decision_time = pd.Timestamp("2026-01-01 00:00Z")
    frame = pd.DataFrame({"AAAUSDT": [0.02]}, index=pd.DatetimeIndex([decision_time]))
    artifact = tmp_path / "deployed_target_weights.parquet"
    frame.to_parquet(artifact, index=True)

    class MarketClient:
        def exchange_info(self):
            return {
                "symbols": [{
                    "symbol": "AAAUSDT",
                    "contractType": "PERPETUAL",
                    "quoteAsset": "USDT",
                    "status": "TRADING",
                    "quantityPrecision": 3,
                    "pricePrecision": 2,
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "100000"},
                        {"filterType": "MIN_NOTIONAL", "minNotional": "1"},
                    ],
                }],
                "rateLimits": [
                    {"rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "intervalNum": 1, "limit": 2400},
                    {"rateLimitType": "ORDERS", "interval": "MINUTE", "intervalNum": 1, "limit": 1200},
                    {"rateLimitType": "ORDERS", "interval": "SECOND", "intervalNum": 10, "limit": 300},
                ],
            }
        def book_ticker(self, s):
            return {"bidPrice": "100.00", "askPrice": "101.00"}
        def book_tickers(self):
            return {"AAAUSDT": {"bidPrice": "100.00", "askPrice": "101.00"}}
    class OrderClient:
        def request(self, method, path, params=None, *, signed=False):
            if path == "/fapi/v2/account":
                return {"totalWalletBalance": "2000", "availableBalance": "1900", "totalInitialMargin": "10", "totalUnrealizedProfit": "0", "dualSidePosition": "false", "multiAssetsMargin": "false"}
            if path == "/fapi/v2/positionRisk":
                return []
            raise AssertionError(path)
        def sync_server_time(self):
            return None
        def open_orders(self):
            return []
    monkeypatch.setattr(runner_mod, "_market_client", lambda s, dt: MarketClient())
    monkeypatch.setattr(runner_mod, "_order_client", lambda s, dt: OrderClient())
    monkeypatch.setattr(runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl")

    def fake_execute(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        outcomes=[]
        for intent in intents:
            outcomes.append(ExecutionOutcome(symbol=intent.symbol, filled_qty=Decimal("0.5"), unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100"), chases=0, status="FILLED", fills=((Decimal("0.5"), Decimal("100"), 2.0, "maker_fill", "maker"),), maker_qty=Decimal("0.5"), taker_qty=Decimal("0")))  # noqa: PERF401
        return tuple(outcomes)
    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute)

    fills_dir = tmp_path / "fills"
    eq_dir = tmp_path / "eq"
    ledger_path = tmp_path / "ledger.json"
    settings = LiveSettings(mode="paper", notional_equity_usdt=2000.0, ledger_path=str(ledger_path), fills_dir=str(fills_dir), execution_quality_dir=str(eq_dir))
    report = run_shadow_cycle(settings, decision_time, artifact, now=decision_time+pd.Timedelta(hours=2))
    assert report.status == "COMPLETE"
    # check fills written
    from src.live.fills import load_fills
    df = load_fills(fills_dir)
    assert len(df) == len(report.outcomes)  # one fill per outcome
    assert (df["mode"] == "paper").all()
    assert (df["run_id"] == decision_time.strftime("%Y%m%d")).all()
    # failsoft: monkeypatch append_fills to raise
    def raise_oserror(events, d):
        raise OSError("disk full")
    monkeypatch.setattr(runner_mod, "append_fills", raise_oserror)
    ledger_path2 = tmp_path / "ledger2.json"
    settings2 = LiveSettings(mode="paper", notional_equity_usdt=2000.0, ledger_path=str(ledger_path2), fills_dir=str(tmp_path / "fills2"))
    report2 = run_shadow_cycle(settings2, decision_time, artifact, now=decision_time+pd.Timedelta(hours=2))
    assert report2.status == "COMPLETE"
    # audit log should contain fills_write_failed
    audit_path = tmp_path / "shadow_cycle.jsonl"
    if audit_path.exists():
        txt = audit_path.read_text()
        assert "fills_write_failed" in txt



def test_SCENARIO_REC_10_runner_failsoft_collect(tmp_path, monkeypatch):
    import pandas as pd
    from decimal import Decimal
    import src.live.runner as runner_mod
    from src.live.runner import run_shadow_cycle
    from src.live.settings import LiveSettings
    import json

    decision_time = pd.Timestamp("2026-01-01 00:00Z")
    artifact = tmp_path / "weights.parquet"
    pd.DataFrame({"A": [0.02]}, index=pd.DatetimeIndex([decision_time])).to_parquet(artifact, index=True)

    class MarketClient:
        def exchange_info(self):
            return {
                "symbols": [{"symbol": "A", "contractType": "PERPETUAL", "quoteAsset": "USDT", "status": "TRADING", "quantityPrecision": 3, "pricePrecision": 2, "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01"}, {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "100000"}, {"filterType": "MIN_NOTIONAL", "minNotional": "1"}]}],
                "rateLimits": [{"rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "intervalNum": 1, "limit": 2400}, {"rateLimitType": "ORDERS", "interval": "MINUTE", "intervalNum": 1, "limit": 1200}, {"rateLimitType": "ORDERS", "interval": "SECOND", "intervalNum": 10, "limit": 300}],
            }

        def book_tickers(self):
            return {"A": {"symbol": "A", "bidPrice": "100", "askPrice": "101"}}

        def premium_index(self):
            raise OSError("premium fail")

        def book_ticker(self, s):
            return {"bidPrice": "100", "askPrice": "101"}

    class OrderClient:
        def request(self, method, path, params=None, *, signed=False):
            if path == "/fapi/v2/account":
                return {"totalWalletBalance": "2000", "availableBalance": "1900", "totalInitialMargin": "10", "totalUnrealizedProfit": "0", "dualSidePosition": "false", "multiAssetsMargin": "false"}
            if path == "/fapi/v2/positionRisk":
                return []
            raise AssertionError(path)

        def sync_server_time(self):
            return None

        def open_orders(self):
            return []

    monkeypatch.setattr(runner_mod, "_market_client", lambda s, dt: MarketClient())
    monkeypatch.setattr(runner_mod, "_order_client", lambda s, dt: OrderClient())
    monkeypatch.setattr(runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl")

    def fake_execute(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        from src.live.executor import ExecutionOutcome

        return tuple(ExecutionOutcome(symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100"), chases=0, status="FILLED", fills=((Decimal("0.5"), Decimal("100"), 2.0, "maker_fill", "maker"),)) for i in intents)

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute)
    # monkeypatch microstructure and tax to fail
    monkeypatch.setattr(runner_mod, "append_microstructure", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    monkeypatch.setattr(runner_mod, "append_tax_records", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))

    settings = LiveSettings(mode="paper", notional_equity_usdt=2000.0, ledger_path=str(tmp_path / "ledger.json"), fills_dir=str(tmp_path / "fills"), execution_quality_dir=str(tmp_path / "eq"))
    report = run_shadow_cycle(settings, decision_time, artifact, now=decision_time + pd.Timedelta(hours=2))
    assert report.status == "COMPLETE"
    audit_path = tmp_path / "shadow_cycle.jsonl"
    txt = audit_path.read_text(encoding="utf-8")
    assert "microstructure_write_failed" in txt
    assert "tax_ledger_write_failed" in txt
    # premium_index failure should still have intent
    assert report.intent_count > 0



def test_run_shadow_cycle_paper_mode_records_immediate_taker_fills(tmp_path, monkeypatch) -> None:
    import src.live.runner as runner_mod
    from src.live.runner import run_shadow_cycle
    from src.live.settings import ExecutionMode, LiveSettings

    decision_time = pd.Timestamp("2026-08-24 00:00Z")
    now = decision_time + pd.Timedelta(hours=2)
    artifact = tmp_path / "weights.parquet"
    pd.DataFrame({"AAAUSDT": [0.02]}, index=pd.DatetimeIndex([decision_time])).to_parquet(artifact, index=True)

    class MarketClient:
        def exchange_info(self):
            return {
                "symbols": [{
                    "symbol": "AAAUSDT",
                    "contractType": "PERPETUAL",
                    "quoteAsset": "USDT",
                    "status": "TRADING",
                    "quantityPrecision": 3,
                    "pricePrecision": 2,
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "100000"},
                        {"filterType": "MIN_NOTIONAL", "minNotional": "1"},
                    ],
                }],
                "rateLimits": [
                    {"rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "intervalNum": 1, "limit": 2400},
                    {"rateLimitType": "ORDERS", "interval": "MINUTE", "intervalNum": 1, "limit": 1200},
                    {"rateLimitType": "ORDERS", "interval": "SECOND", "intervalNum": 10, "limit": 300},
                ],
            }

        def book_ticker(self, symbol):
            return {"bidPrice": "100.00", "askPrice": "102.00"}

        def book_tickers(self):
            return {"AAAUSDT": {"bidPrice": "100.00", "askPrice": "102.00", "symbol": "AAAUSDT"}}

        def premium_index(self):
            return {}

        def depth(self, symbol, *, limit=20):
            return {"lastUpdateId": 1, "bids": [["100.00", "1"]], "asks": [["102.00", "1"]]}

    class OrderClient:
        def __init__(self):
            self.new_order_called = False

        def request(self, method, path, params=None, *, signed=False):
            if path == "/fapi/v2/account":
                return {"totalWalletBalance": "2000", "availableBalance": "1900", "totalInitialMargin": "10", "totalUnrealizedProfit": "0", "dualSidePosition": "false", "multiAssetsMargin": "false"}
            if path == "/fapi/v2/positionRisk":
                return []
            raise AssertionError(path)

        def sync_server_time(self):
            return None

        def open_orders(self):
            return []

        def book_tickers(self):
            return {"AAAUSDT": {"bidPrice": "100.00", "askPrice": "102.00", "symbol": "AAAUSDT"}}

        def book_ticker(self, symbol):
            return {"bidPrice": "100.00", "askPrice": "102.00"}

        def new_order(self, params):
            self.new_order_called = True
            raise AssertionError("should not be called for immediate_taker")

        def cancel_order(self, *a, **k):
            raise AssertionError("cancel not called")

        def query_order(self, *a, **k):
            raise AssertionError("query not called")

    order_client = OrderClient()
    monkeypatch.setattr(runner_mod, "_market_client", lambda s, dt: MarketClient())
    monkeypatch.setattr(runner_mod, "_order_client", lambda s, dt: order_client)
    monkeypatch.setattr(runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl")

    ledger_path = tmp_path / "ledger.json"
    fills_dir = tmp_path / "fills"
    settings = LiveSettings(mode=ExecutionMode.PAPER, notional_equity_usdt=2000.0, ledger_path=str(ledger_path), fills_dir=str(fills_dir), orderbook_capture_enabled=False, microstructure_dir=str(tmp_path / "micro"), execution_quality_dir=str(tmp_path / "eq"), portfolio_state_dir=str(tmp_path / "port"), tax_ledger_dir=str(tmp_path / "tax"))
    # fees: default maker 2 taker 5 slippage 3 => 8
    report = run_shadow_cycle(settings, decision_time, artifact, now=now)
    assert report.status == "COMPLETE"
    assert len(report.outcomes) == 1
    oc = report.outcomes[0]
    assert oc.status == "FILLED"
    # book-mid = 101
    assert oc.avg_fill_price == Decimal("101.00") or oc.avg_fill_price == Decimal("101")
    # fills parquet
    from src.live.fills import load_fills

    df = load_fills(fills_dir)
    assert len(df) == 1
    assert df.iloc[0]["reason"] == "immediate_taker"
    assert df.iloc[0]["liquidity"] == "taker"
    assert float(df.iloc[0]["fee_bps"]) == pytest.approx(8.0)
    assert order_client.new_order_called is False
    # ledger reflects filled qty
    from src.live.ledger import load_ledger

    state = load_ledger(ledger_path)
    assert state.positions.get("AAAUSDT", Decimal("0")) != Decimal("0")



def test_run_shadow_cycle_captures_orderbook_and_never_halts_on_failure(tmp_path, monkeypatch) -> None:
    import src.live.runner as runner_mod
    from src.live.runner import run_shadow_cycle
    from src.live.settings import LiveSettings
    import json

    decision_time = pd.Timestamp("2026-08-24 00:00Z")
    now = decision_time + pd.Timedelta(hours=2)
    artifact = tmp_path / "weights.parquet"
    pd.DataFrame({"AAAUSDT": [0.02]}, index=pd.DatetimeIndex([decision_time])).to_parquet(artifact, index=True)

    class MarketClientOK:
        def exchange_info(self):
            return {
                "symbols": [{
                    "symbol": "AAAUSDT",
                    "contractType": "PERPETUAL",
                    "quoteAsset": "USDT",
                    "status": "TRADING",
                    "quantityPrecision": 3,
                    "pricePrecision": 2,
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "100000"},
                        {"filterType": "MIN_NOTIONAL", "minNotional": "1"},
                    ],
                }],
                "rateLimits": [
                    {"rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "intervalNum": 1, "limit": 2400},
                    {"rateLimitType": "ORDERS", "interval": "MINUTE", "intervalNum": 1, "limit": 1200},
                    {"rateLimitType": "ORDERS", "interval": "SECOND", "intervalNum": 10, "limit": 300},
                ],
            }

        def book_ticker(self, symbol):
            return {"bidPrice": "100.00", "askPrice": "101.00"}

        def book_tickers(self):
            return {"AAAUSDT": {"bidPrice": "100.00", "askPrice": "101.00", "symbol": "AAAUSDT"}}

        def premium_index(self):
            return {}

        def depth(self, symbol, *, limit=20):
            return {"lastUpdateId": 1, "bids": [["100", "1"]], "asks": [["101", "1"]]}

    class OrderClient:
        def request(self, method, path, params=None, *, signed=False):
            if path == "/fapi/v2/account":
                return {"totalWalletBalance": "2000", "availableBalance": "1900", "totalInitialMargin": "10", "totalUnrealizedProfit": "0", "dualSidePosition": "false", "multiAssetsMargin": "false"}
            if path == "/fapi/v2/positionRisk":
                return []
            raise AssertionError(path)

        def sync_server_time(self):
            return None

        def open_orders(self):
            return []

        def book_ticker(self, s):
            return {"bidPrice": "100.00", "askPrice": "101.00"}

        def book_tickers(self):
            return {"AAAUSDT": {"bidPrice": "100.00", "askPrice": "101.00"}}

    # first run success
    orderbook_dir = tmp_path / "ob"

    def fake_exec(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        from src.live.executor import ExecutionOutcome as EO

        return tuple(EO(symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100"), chases=0, status="FILLED", fills=((i.quantity, Decimal("100"), 2.0, "maker_fill", "maker"),)) for i in intents)

    monkeypatch.setattr(runner_mod, "_market_client", lambda s, dt: MarketClientOK())
    monkeypatch.setattr(runner_mod, "_order_client", lambda s, dt: OrderClient())
    monkeypatch.setattr(runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl")
    monkeypatch.setattr(runner_mod, "execute_intents", fake_exec)

    ledger_path = tmp_path / "ledger.json"
    settings = LiveSettings(
        notional_equity_usdt=2000.0,
        ledger_path=str(ledger_path),
        orderbook_capture_enabled=True,
        orderbook_capture_dir=str(orderbook_dir),
        orderbook_capture_duration_s=0,
        orderbook_capture_interval_s=10,
        microstructure_dir=str(tmp_path / "micro"),
        execution_quality_dir=str(tmp_path / "eq"),
        portfolio_state_dir=str(tmp_path / "port"),
        fills_dir=str(tmp_path / "fills"),
        tax_ledger_dir=str(tmp_path / "tax"),
    )
    report = run_shadow_cycle(settings, decision_time, artifact, now=now)
    assert report.status == "COMPLETE"
    files = list(orderbook_dir.glob("live_orderbook_*.parquet"))
    assert len(files) >= 1

    # second run where depth raises
    class MarketClientFail(MarketClientOK):
        def depth(self, symbol, *, limit=20):
            raise RuntimeError("depth fail")

    # second run where depth raises -> simulate whole capture failure via patching orderbook capture to raise
    monkeypatch.setattr(runner_mod, "_market_client", lambda s, dt: MarketClientFail())
    import src.live.orderbook as ob_mod

    monkeypatch.setattr(ob_mod, "capture_order_books", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("depth fail")))
    # need fresh ledger for second decision time to avoid duplicate? use same ledger but different decision_time
    decision_time2 = pd.Timestamp("2026-08-25 00:00Z")
    artifact2 = tmp_path / "weights2.parquet"
    pd.DataFrame({"AAAUSDT": [0.02]}, index=pd.DatetimeIndex([decision_time2])).to_parquet(artifact2, index=True)
    ledger_path2 = tmp_path / "ledger2.json"
    settings2 = LiveSettings(
        notional_equity_usdt=2000.0,
        ledger_path=str(ledger_path2),
        orderbook_capture_enabled=True,
        orderbook_capture_dir=str(tmp_path / "ob2"),
        orderbook_capture_duration_s=0,
        orderbook_capture_interval_s=10,
        microstructure_dir=str(tmp_path / "micro2"),
        execution_quality_dir=str(tmp_path / "eq2"),
        portfolio_state_dir=str(tmp_path / "port2"),
        fills_dir=str(tmp_path / "fills2"),
        tax_ledger_dir=str(tmp_path / "tax2"),
    )
    report2 = run_shadow_cycle(settings2, decision_time2, artifact2, now=decision_time2 + pd.Timedelta(hours=2))
    assert report2.status == "COMPLETE"
    audit_path = tmp_path / "shadow_cycle.jsonl"
    txt = audit_path.read_text(encoding="utf-8")
    assert "orderbook_capture_failed" in txt



def test_run_shadow_cycle_shadow_mode_does_not_use_immediate_taker(tmp_path, monkeypatch) -> None:
    import src.live.runner as runner_mod
    from src.live.runner import run_shadow_cycle
    from src.live.settings import LiveSettings, ExecutionMode
    from src.live.executor import ExecutionOutcome

    decision_time = pd.Timestamp("2026-08-24 00:00Z")
    now = decision_time + pd.Timedelta(hours=2)
    artifact = tmp_path / "weights.parquet"
    pd.DataFrame({"AAAUSDT": [0.02]}, index=pd.DatetimeIndex([decision_time])).to_parquet(artifact, index=True)

    class MarketClient:
        def exchange_info(self):
            return {
                "symbols": [{
                    "symbol": "AAAUSDT",
                    "contractType": "PERPETUAL",
                    "quoteAsset": "USDT",
                    "status": "TRADING",
                    "quantityPrecision": 3,
                    "pricePrecision": 2,
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "100000"},
                        {"filterType": "MIN_NOTIONAL", "minNotional": "1"},
                    ],
                }],
                "rateLimits": [
                    {"rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "intervalNum": 1, "limit": 2400},
                    {"rateLimitType": "ORDERS", "interval": "MINUTE", "intervalNum": 1, "limit": 1200},
                    {"rateLimitType": "ORDERS", "interval": "SECOND", "intervalNum": 10, "limit": 300},
                ],
            }

        def book_ticker(self, s):
            return {"bidPrice": "100", "askPrice": "101"}

        def book_tickers(self):
            return {"AAAUSDT": {"bidPrice": "100", "askPrice": "101", "symbol": "AAAUSDT"}}

        def premium_index(self):
            return {}

        def depth(self, symbol, *, limit=20):
            return {"lastUpdateId": 1, "bids": [["100", "1"]], "asks": [["101", "1"]]}

    class OrderClient:
        def request(self, method, path, params=None, *, signed=False):
            if path == "/fapi/v2/account":
                return {"totalWalletBalance": "2000", "availableBalance": "1900", "totalInitialMargin": "10", "totalUnrealizedProfit": "0", "dualSidePosition": "false", "multiAssetsMargin": "false"}
            if path == "/fapi/v2/positionRisk":
                return []
            raise AssertionError(path)

        def sync_server_time(self):
            return None

        def open_orders(self):
            return []

    monkeypatch.setattr(runner_mod, "_market_client", lambda s, dt: MarketClient())
    monkeypatch.setattr(runner_mod, "_order_client", lambda s, dt: OrderClient())
    monkeypatch.setattr(runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl")

    captured_kwargs: dict = {}

    def fake_execute(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        captured_kwargs.update(kwargs)
        # return SHADOW outcome
        return tuple(
            ExecutionOutcome(symbol=i.symbol, filled_qty=Decimal("0"), unfilled_qty=i.quantity, avg_fill_price=None, chases=0, status="SHADOW")
            for i in intents
        )

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute)
    settings = LiveSettings(
        mode=ExecutionMode.SHADOW,
        notional_equity_usdt=2000.0,
        ledger_path=str(tmp_path / "ledger.json"),
        fills_dir=str(tmp_path / "fills"),
        microstructure_dir=str(tmp_path / "micro"),
        execution_quality_dir=str(tmp_path / "eq"),
        portfolio_state_dir=str(tmp_path / "port"),
        tax_ledger_dir=str(tmp_path / "tax"),
        orderbook_capture_enabled=False,
    )
    report = run_shadow_cycle(settings, decision_time, artifact, now=now)
    assert report.status == "COMPLETE"
    # outcomes should be SHADOW
    assert all(o.status == "SHADOW" for o in report.outcomes)
    assert captured_kwargs.get("paper_fill_model") is None
    # no immediate_taker in fills
    from src.live.fills import load_fills

    df = load_fills(tmp_path / "fills")
    if not df.empty:
        assert not (df["reason"] == "immediate_taker").any()
# SCENARIO_REC_10-runner-failsoft-collect
