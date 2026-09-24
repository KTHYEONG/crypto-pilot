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
def _maybe_disable_orderbook_capture(monkeypatch):
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
    _seed_close_artifact(path)
    return path


def _seed_close_artifact(weights_path: Path, close: float = 100.0) -> Path:
    """Mirror every weights row into the new decision-close artifact for tests."""
    from src.live.deployed_weights import decision_ohlcv_close_path

    frame = pd.read_parquet(Path(weights_path))
    closes = pd.DataFrame(
        float(close), index=pd.DatetimeIndex(frame.index), columns=list(frame.columns), dtype="float64",
    )
    out = decision_ohlcv_close_path(Path(weights_path))
    closes.to_parquet(out, index=True)
    return out

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
        drop_path = artifact.parent / "deployed_target_weights_drops.parquet"
        weights.to_parquet(drop_path, index=True)
        _seed_close_artifact(drop_path)

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
    from decimal import Decimal
    import pandas as pd
    import src.live.runner as runner_mod
    from src.live.executor import ExecutionOutcome
    from src.live.ledger import load_ledger
    from src.live.runner import run_shadow_cycle
    from src.live.settings import LiveSettings
    from tests.unit.live._runner_stubs import DECISION_TIME, StubMarketClient, StubOrderClient
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
    # PAPER funding accrual under the parity contract: stub per-symbol funding
    # I/O (empty series -> zero delta) so the multi-day continuity assertions
    # exercise sizing/ledger logic, not the on-disk funding store.
    epochs = pd.date_range(DECISION_TIME - pd.Timedelta(days=1), DECISION_TIME + pd.Timedelta(days=3), freq="4h")
    monkeypatch.setattr(runner_mod, "_load_paper_funding", lambda symbols: {s: pd.Series(0.0, index=epochs) for s in symbols})
    monkeypatch.setattr(runner_mod, "_load_paper_trade_closes", lambda symbols: {s: pd.Series(100.0, index=epochs) for s in symbols})

    days = [DECISION_TIME + pd.Timedelta(days=i) for i in range(3)]
    weights = pd.DataFrame(
        {"AAAUSDT": [0.02] * 3, "BUSDT": [-0.02] * 3}, index=pd.DatetimeIndex(days)
    )
    weights_path = tmp_path / "deployed_target_weights_paper.parquet"
    weights.to_parquet(weights_path, index=True)
    # Seed at the stub book-mid so continuity is exercised without anchor/mark divergence dust.
    _seed_close_artifact(weights_path, close=100.5)

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
    weights_path = tmp_path / "deployed_target_weights.parquet"
    weights.to_parquet(weights_path, index=True)
    _seed_close_artifact(weights_path)

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
    _seed_close_artifact(artifact)

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
            outcomes.append(ExecutionOutcome(symbol=intent.symbol, filled_qty=Decimal("0.5"), unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100"), chases=0, status="FILLED", fills=((Decimal("0.5"), Decimal("100"), 2.0, "maker_fill", "maker", pd.Timestamp("2026-01-01 00:00Z")),), maker_qty=Decimal("0.5"), taker_qty=Decimal("0")))  # noqa: PERF401
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
    artifact = tmp_path / "deployed_target_weights.parquet"
    pd.DataFrame({"A": [0.02]}, index=pd.DatetimeIndex([decision_time])).to_parquet(artifact, index=True)
    _seed_close_artifact(artifact)

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

        return tuple(ExecutionOutcome(symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100"), chases=0, status="FILLED", fills=((Decimal("0.5"), Decimal("100"), 2.0, "maker_fill", "maker", pd.Timestamp("2026-01-01 00:00Z")),)) for i in intents)

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
    artifact = tmp_path / "deployed_target_weights.parquet"
    pd.DataFrame({"AAAUSDT": [0.02]}, index=pd.DatetimeIndex([decision_time])).to_parquet(artifact, index=True)
    _seed_close_artifact(artifact)

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



def test_run_shadow_cycle_shadow_mode_does_not_use_immediate_taker(tmp_path, monkeypatch) -> None:
    import src.live.runner as runner_mod
    from src.live.runner import run_shadow_cycle
    from src.live.settings import LiveSettings, ExecutionMode
    from src.live.executor import ExecutionOutcome

    decision_time = pd.Timestamp("2026-08-24 00:00Z")
    now = decision_time + pd.Timedelta(hours=2)
    artifact = tmp_path / "deployed_target_weights.parquet"
    pd.DataFrame({"AAAUSDT": [0.02]}, index=pd.DatetimeIndex([decision_time])).to_parquet(artifact, index=True)
    _seed_close_artifact(artifact)

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


def test_run_shadow_cycle_paper_accrues_funding_and_uses_parity_policy(tmp_path, monkeypatch) -> None:
    from decimal import Decimal
    import pandas as pd
    import src.live.runner as runner_mod
    from src.live.executor import EXECUTION_BAR_SECONDS, ExecutionOutcome
    from src.live.ledger import LedgerState, load_ledger, save_ledger
    from src.live.settings import ExecutionMode, LiveSettings

    class MarketClient:
        def exchange_info(self):
            return {
                "symbols": [{
                    "symbol": "AAAUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT", "status": "TRADING",
                    "quantityPrecision": 3, "pricePrecision": 2,
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

    decision_time = pd.Timestamp("2026-08-24 00:00Z")
    now = decision_time + pd.Timedelta(hours=2)
    artifact = tmp_path / "deployed_target_weights.parquet"
    pd.DataFrame({"AAAUSDT": [0.02]}, index=pd.DatetimeIndex([decision_time])).to_parquet(artifact, index=True)
    _seed_close_artifact(artifact)
    ledger_path = tmp_path / "ledger.json"
    save_ledger(ledger_path, LedgerState(positions={"AAAUSDT": Decimal("1")}, equity_high_water_mark=Decimal("2000"), cash_usdt=Decimal("1900"), funding_accrued_through=decision_time - pd.Timedelta(hours=23)))
    funding = pd.Series([0.5, 0.001], index=pd.DatetimeIndex([decision_time - pd.Timedelta(hours=24), decision_time]))
    captured: dict[str, object] = {}
    real_equity = runner_mod.resolve_sizing_equity

    def equity_spy(snapshot, cap_usdt, **kwargs):
        captured["cash_usdt"] = kwargs["cash_usdt"]
        return real_equity(snapshot, cap_usdt, **kwargs)

    def fake_execute(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        captured["policy"] = policy
        return tuple(ExecutionOutcome(symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"), avg_fill_price=Decimal("101"), chases=0, status="FILLED") for i in intents)

    monkeypatch.setattr(runner_mod, "_market_client", lambda s, dt: MarketClient())
    monkeypatch.setattr(runner_mod, "_order_client", lambda s, dt: OrderClient())
    monkeypatch.setattr(runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl")
    monkeypatch.setattr(runner_mod, "_load_paper_funding", lambda symbols: {"AAAUSDT": funding})
    monkeypatch.setattr(runner_mod, "_load_paper_trade_closes", lambda symbols: {"AAAUSDT": pd.Series([101.0], index=pd.DatetimeIndex([decision_time]))})
    monkeypatch.setattr(runner_mod, "resolve_sizing_equity", equity_spy)
    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute)

    settings = LiveSettings(mode=ExecutionMode.PAPER, notional_equity_usdt=2000.0, ledger_path=str(ledger_path), fills_dir=str(tmp_path / "fills"), orderbook_capture_enabled=False, microstructure_dir=str(tmp_path / "micro"), execution_quality_dir=str(tmp_path / "eq"), portfolio_state_dir=str(tmp_path / "port"), tax_ledger_dir=str(tmp_path / "tax"))
    report = runner_mod.run_shadow_cycle(settings, decision_time, artifact, now=now)
    assert report.status == "COMPLETE"
    assert captured["cash_usdt"] == Decimal("1899.899")
    assert captured["policy"].passive_deadline_s == EXECUTION_BAR_SECONDS
    assert load_ledger(ledger_path).funding_accrued_through == now
def test_accrue_ledger_funding_requires_cash(tmp_path, monkeypatch) -> None:
    from decimal import Decimal
    import pandas as pd
    import pytest
    import src.live.runner as runner_mod
    from src.common.errors import DataIntegrityError
    from src.live.ledger import LedgerState

    epoch = pd.Timestamp("2026-09-02 00:00", tz="UTC")
    state = LedgerState(
        positions={"AAAUSDT": Decimal("1")},
        equity_high_water_mark=Decimal("2000"),
        cash_usdt=None,
        funding_accrued_through=pd.Timestamp("2026-09-01 01:03", tz="UTC"),
    )
    monkeypatch.setattr(runner_mod, "_load_paper_trade_closes", lambda symbols: {"AAAUSDT": pd.Series([100.0], index=pd.DatetimeIndex([epoch]))})
    monkeypatch.setattr(runner_mod, "_load_paper_funding", lambda symbols: {"AAAUSDT": pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"))})
    quiet, accrual, _ = runner_mod._accrue_ledger_funding(state, pd.Timestamp("2026-09-02 01:03", tz="UTC"), tmp_path / "quiet.json")
    assert quiet.cash_usdt is None
    assert accrual.cash_delta == Decimal(0)

    monkeypatch.setattr(runner_mod, "_load_paper_funding", lambda symbols: {"AAAUSDT": pd.Series([0.001], index=pd.DatetimeIndex([epoch]))})
    with pytest.raises(DataIntegrityError, match="requires cash_usdt"):
        runner_mod._accrue_ledger_funding(state, pd.Timestamp("2026-09-02 01:03", tz="UTC"), tmp_path / "ledger.json")
def test_load_paper_funding_skips_missing_symbols(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.common.paths as paths

    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-09-01 08:00"], utc=True),
            "funding_rate": [0.001],
        }
    )
    frame.to_parquet(tmp_path / "AAAUSDT.parquet", index=False)
    monkeypatch.setattr(paths, "funding_path", lambda symbol: tmp_path / f"{symbol}.parquet")
    out = runner_mod._load_paper_funding(["AAAUSDT", "MISSINGUSDT"])
    assert list(out) == ["AAAUSDT"]
    assert float(out["AAAUSDT"].iloc[0]) == 0.001

"""Replacement + new depth-recorder cycle tests (spliced into test_runner_shadow_cycle.py)."""
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

import src.live.runner as runner_mod
from src.live.depth_capture import DepthCaptureSummary
from src.live.executor import ExecutionOutcome
from src.live.runner import run_shadow_cycle
from src.live.settings import LiveSettings

from tests.unit.live._runner_stubs import DECISION_TIME, NOW, StubMarketClient, StubOrderClient


class _StubDepthRecorder:
    instances: list[_StubDepthRecorder] = []

    def __init__(self, symbols: Any, **kwargs: Any) -> None:
        self.symbols = list(symbols)
        self.kwargs = kwargs
        self.starts = 0
        self.stops: list[float] = []
        _StubDepthRecorder.instances.append(self)

    def start(self) -> None:
        self.starts += 1

    def stop(self, *, post_window_s: float, shutdown: Any = None) -> DepthCaptureSummary:
        self.stops.append(float(post_window_s))
        return DepthCaptureSummary(
            rows=len(self.symbols),
            symbols_requested=len(self.symbols),
            symbols_seen=len(self.symbols),
            reconnects=0,
            parts=1,
        )


def _three_symbol_market():
    class _Market(StubMarketClient):
        def exchange_info(self) -> dict[str, Any]:
            payload = super().exchange_info()
            template = payload["symbols"][0]
            extra = dict(template)
            extra["symbol"] = "CCCUSDT"
            payload["symbols"].append(extra)
            return payload

        def book_ticker(self, symbol: str) -> dict[str, str]:
            return {"bidPrice": "100.00", "askPrice": "101.00"}

    return _Market()


def _three_symbol_weights(tmp_path: Path) -> Path:
    path = tmp_path / "deployed_target_weights_depth.parquet"
    pd.DataFrame(
        {"AAAUSDT": [0.03], "BUSDT": [0.02], "CCCUSDT": [0.01]},
        index=pd.DatetimeIndex([DECISION_TIME]),
    ).to_parquet(path, index=True)
    closes = pd.DataFrame(
        100.0, index=pd.DatetimeIndex([DECISION_TIME]),
        columns=["AAAUSDT", "BUSDT", "CCCUSDT"], dtype="float64",
    )
    from src.live.deployed_weights import decision_ohlcv_close_path

    closes.to_parquet(decision_ohlcv_close_path(path), index=True)
    return path


def _depth_cycle_settings(tmp_path: Path, monkeypatch: Any, **overrides: Any) -> tuple[Any, list[Any]]:
    import src.common.paths as paths_mod
    import src.live.settings as settings_mod

    monkeypatch.setattr(paths_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "_market_client", lambda s, dt: _three_symbol_market())
    monkeypatch.setattr(runner_mod, "_order_client", lambda s, dt: StubOrderClient())
    monkeypatch.setattr(
        runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl"
    )
    monkeypatch.setattr(runner_mod, "ExecutionDepthRecorder", _StubDepthRecorder)
    _StubDepthRecorder.instances.clear()
    calls: list[Any] = []

    def fake_execute(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        calls.extend(intents)
        return tuple(
            ExecutionOutcome(
                symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"),
                avg_fill_price=Decimal("100"), chases=0, status="FILLED",
            )
            for i in intents
        )

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute)
    params: dict[str, Any] = {
        "mode": "paper",
        "notional_equity_usdt": 2000.0,
        "ledger_path": str(tmp_path / "ledger.json"),
        "order_journal_path": str(tmp_path / "order_journal.jsonl"),
        "fills_dir": str(tmp_path / "fills"),
        "execution_quality_dir": str(tmp_path / "eq"),
        "portfolio_state_dir": str(tmp_path / "port"),
        "microstructure_dir": str(tmp_path / "micro"),
        "tax_ledger_dir": str(tmp_path / "tax"),
        "record_run_id": "depthtest01",
        "exec_depth_post_window_s": 0.0,
    }
    params.update(overrides)
    return LiveSettings(**params), calls


def test_run_shadow_cycle_depth_recorder_brackets_execution(tmp_path, monkeypatch) -> None:
    settings, calls = _depth_cycle_settings(tmp_path, monkeypatch)
    weights_path = _three_symbol_weights(tmp_path)
    report = run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)
    assert report.status == "COMPLETE"
    assert len(_StubDepthRecorder.instances) == 1
    recorder = _StubDepthRecorder.instances[0]
    assert recorder.starts == 1
    expected = sorted(calls, key=lambda i: abs(i.quantity * Decimal("100.50")), reverse=True)
    assert recorder.symbols == [i.symbol for i in expected]
    assert len(recorder.symbols) == 3
    assert recorder.stops == [settings.exec_depth_post_window_s]
    assert recorder.kwargs["run_id"] == DECISION_TIME.strftime("%Y%m%d")
    lines = (tmp_path / "shadow_cycle.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line)["event"] for line in lines]
    assert "exec_depth_capture" in events


def test_run_shadow_cycle_execution_failure_stops_recorder_immediately(tmp_path, monkeypatch) -> None:
    from src.live.errors import LiveTradingError

    settings, _ = _depth_cycle_settings(tmp_path, monkeypatch)

    def _boom(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        raise LiveTradingError("venue down")

    monkeypatch.setattr(runner_mod, "execute_intents", _boom)
    weights_path = _three_symbol_weights(tmp_path)
    report = run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)
    assert report.status == "HALT"
    assert len(_StubDepthRecorder.instances) == 1
    assert _StubDepthRecorder.instances[0].stops == [0.0]


def test_run_shadow_cycle_pre_execution_failure_stops_recorder_immediately(tmp_path, monkeypatch) -> None:
    settings, calls = _depth_cycle_settings(tmp_path, monkeypatch)

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise OSError("policy build failed")

    # 캡처 시작 뒤, 주문 집행 전 단계의 실패
    monkeypatch.setattr(runner_mod, "_uncovered_positions", _boom)
    weights_path = _three_symbol_weights(tmp_path)
    report = run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)
    assert report.status == "HALT"
    assert calls == []
    assert len(_StubDepthRecorder.instances) == 1
    assert _StubDepthRecorder.instances[0].stops == [0.0]


def test_run_shadow_cycle_never_calls_rest_depth_capture(tmp_path, monkeypatch) -> None:
    import src.live.orderbook as ob_mod

    settings, _ = _depth_cycle_settings(tmp_path, monkeypatch)

    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("REST depth capture must not run in the cycle")

    monkeypatch.setattr(ob_mod, "capture_order_books", _forbidden)
    weights_path = _three_symbol_weights(tmp_path)
    report = run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)
    assert report.status == "COMPLETE"
    assert list((tmp_path / "ob").glob("*.parquet")) == [] if (tmp_path / "ob").exists() else True


def test_run_shadow_cycle_audit_mirrored_into_run_directory(tmp_path, monkeypatch) -> None:
    settings, _ = _depth_cycle_settings(tmp_path, monkeypatch)
    weights_path = _three_symbol_weights(tmp_path)
    report = run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)
    assert report.status == "COMPLETE"
    mirror = tmp_path / "state" / "runs" / "depthtest01" / "audit" / "2026-08-24.jsonl"
    assert mirror.exists()
    assert mirror.read_bytes() == (tmp_path / "shadow_cycle.jsonl").read_bytes()
def test_run_shadow_cycle_complete_stamps_ledger_and_rerun_does_not_trade(artifact, live_env, tmp_path) -> None:
    from src.live.ledger import load_ledger
    from src.live.runner import run_shadow_cycle
    from src.live.settings import LiveSettings
    from tests.unit.live._runner_stubs import DECISION_TIME, NOW

    ledger_path = tmp_path / "ledger_once.json"
    settings = LiveSettings(notional_equity_usdt=2000.0, ledger_path=str(ledger_path))

    first = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)
    executed_after_first = len(live_env)
    positions_after_first = dict(load_ledger(ledger_path).positions)
    second = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)

    assert first.status == "COMPLETE"
    assert first.reason is None
    assert executed_after_first > 0
    assert load_ledger(ledger_path).last_executed_decision_time == DECISION_TIME
    assert (second.status, second.reason, second.intent_count) == ("COMPLETE", "already_executed", 0)
    assert len(live_env) == executed_after_first
    assert dict(load_ledger(ledger_path).positions) == positions_after_first



def test_load_paper_trade_closes_reads_close_by_completion_and_skips_missing(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.common.paths as paths
    import src.live.runner as runner_mod

    bars = pd.to_datetime(["2026-09-01 01:00", "2026-09-01 00:00", "2026-09-01 01:00"], utc=True)
    pd.DataFrame({
        "timestamp": (bars - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms"),
        "open": [11.0, 10.0, 12.0], "high": 1.0, "low": 1.0, "close": [21.0, 20.0, 22.0],
    }).to_parquet(tmp_path / "AUSDT.parquet", index=False)
    monkeypatch.setattr(paths, "ohlcv_path", lambda symbol, timeframe: tmp_path / f"{symbol}.parquet")

    out = runner_mod._load_paper_trade_closes(["AUSDT", "MISSINGUSDT"])

    assert list(out) == ["AUSDT"]
    assert out["AUSDT"].index.tolist() == list(pd.to_datetime(["2026-09-01 01:00", "2026-09-01 02:00"], utc=True))
    assert out["AUSDT"].tolist() == [20.0, 22.0]


def test_enforce_funding_lag_alerts_then_halts(monkeypatch) -> None:
    from decimal import Decimal
    import pandas as pd
    import pytest
    import src.live.runner as runner_mod
    from src.common.errors import DataIntegrityError
    from src.live.ledger import FundingAccrual
    from src.live.settings import LiveSettings

    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(runner_mod, "post_alert", lambda url, *, event, detail, decision_time, now: sent.append((event, detail)) or False)
    monkeypatch.setattr(runner_mod, "send_email_alert", lambda **kwargs: False)
    now = pd.Timestamp("2026-09-14 01:26Z")
    decision = pd.Timestamp("2026-09-14 00:00Z")
    fresh = FundingAccrual(Decimal(0), {}, {"AUSDT": pd.Timedelta(hours=9)}, {"AUSDT": pd.Timedelta(hours=4)})
    lagging = FundingAccrual(Decimal(0), {}, {"AUSDT": pd.Timedelta(hours=9, minutes=1), "BUSDT": pd.Timedelta(hours=17)}, {"AUSDT": pd.Timedelta(hours=4), "BUSDT": pd.Timedelta(hours=8)})
    stale = FundingAccrual(Decimal(0), {}, {"AUSDT": pd.Timedelta(hours=25), "BUSDT": pd.Timedelta(hours=30)}, {"AUSDT": pd.Timedelta(hours=4), "BUSDT": pd.Timedelta(hours=8)})

    runner_mod._enforce_funding_lag(fresh, LiveSettings(), decision, now)
    assert sent == []
    runner_mod._enforce_funding_lag(lagging, LiveSettings(), decision, now)
    assert sent == [("paper_funding_lag", "symbols=1 sample=AUSDT")]
    with pytest.raises(DataIntegrityError, match=r"paper funding lag exceeded symbols=AUSDT,BUSDT max_lag_h=30.0"):
        runner_mod._enforce_funding_lag(stale, LiveSettings(), decision, now)


def test_settle_delisted_paper_positions_keeps_unresolved_without_synthetic_settlement(tmp_path, monkeypatch) -> None:
    from decimal import Decimal
    import pandas as pd
    import pytest
    import src.live.runner as runner_mod
    from src.common.errors import DataIntegrityError
    from src.live.audit import AuditLog
    from src.live.ledger import LedgerState
    from src.live.settings import LiveSettings

    audit = AuditLog(tmp_path / "audit.jsonl")
    now = pd.Timestamp("2026-09-14 01:26Z")
    delivery = {"AUSDT": pd.Timestamp("2026-09-10 08:30Z")}
    path = tmp_path / "ledger.json"
    before = LedgerState(positions={"AUSDT": Decimal("1")}, cash_usdt=Decimal("100"))

    with pytest.raises(DataIntegrityError, match="paper delisted holding unresolved symbols=AUSDT"):
        runner_mod._settle_delisted_paper_positions(
            before, delivery, now, path, audit, LiveSettings(), now.normalize(),
        )
    # No synthetic liquidation is booked: ledger untouched, reason audited.
    assert not path.exists()
    audit.close()
    assert "paper_delisted_unresolved" in (tmp_path / "audit.jsonl").read_text(encoding="utf-8")

    # Flat position needs no halt.
    flat = runner_mod._settle_delisted_paper_positions(
        LedgerState(positions={}, cash_usdt=Decimal("100")), delivery, now, path, audit, LiveSettings(), now.normalize(),
    )
    assert flat.positions == {}


def test_run_shadow_cycle_paper_halts_on_delisted_holding_without_synthetic_settlement(tmp_path, monkeypatch) -> None:
    from decimal import Decimal
    import pandas as pd
    import src.live.runner as runner_mod
    from src.live.executor import ExecutionOutcome
    from src.live.ledger import LedgerState, PositionSnapshot, load_ledger, save_ledger
    from src.live.settings import ExecutionMode, LiveSettings
    from tests.unit.live._runner_stubs import DECISION_TIME, NOW, StubMarketClient, StubOrderClient

    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(runner_mod, "post_alert", lambda url, *, event, detail, decision_time, now: alerts.append((event, detail)) or False)
    monkeypatch.setattr(runner_mod, "send_email_alert", lambda **kwargs: False)
    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient())
    monkeypatch.setattr(runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl")

    def fake_execute(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        return tuple(
            ExecutionOutcome(symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100"), chases=0, status="FILLED")
            for i in intents
        )

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute)
    ledger_path = tmp_path / "ledger.json"
    settings = LiveSettings(
        mode=ExecutionMode.PAPER, notional_equity_usdt=2000.0, ledger_path=str(ledger_path),
        fills_dir=str(tmp_path / "fills"), orderbook_capture_enabled=False, microstructure_dir=str(tmp_path / "micro"),
        execution_quality_dir=str(tmp_path / "eq"), portfolio_state_dir=str(tmp_path / "port"), tax_ledger_dir=str(tmp_path / "tax"),
    )
    weights_path = tmp_path / "deployed_target_weights.parquet"
    pd.DataFrame({"BUSDT": [-0.02]}, index=pd.DatetimeIndex([DECISION_TIME])).to_parquet(weights_path, index=True)
    _seed_close_artifact(weights_path)

    delivery_ms = int(DECISION_TIME.value // 1_000_000)

    class DelistingMarketClient(StubMarketClient):
        def exchange_info(self):
            info = super().exchange_info()
            for entry in info["symbols"]:
                if entry["symbol"] == "AAAUSDT":
                    entry["status"] = "SETTLING"
                    entry["deliveryDate"] = delivery_ms
            return info

        def book_tickers(self):
            return {"BUSDT": {"symbol": "BUSDT", "bidPrice": "100.00", "askPrice": "101.00"}}

        def premium_index(self):
            return {}

    monkeypatch.setattr(runner_mod, "_market_client", lambda settings, decision_time: DelistingMarketClient())
    monkeypatch.setattr(runner_mod, "_load_paper_funding", lambda symbols: {})
    monkeypatch.setattr(runner_mod, "_load_paper_trade_closes", lambda symbols: {"AAAUSDT": pd.Series([90.0], index=pd.DatetimeIndex([DECISION_TIME]))})
    executed: list[object] = []
    real_execute = fake_execute

    def guarded_execute(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        executed.extend(intents)
        return real_execute(client, intents, filters, policy, audit, clock, sleep_fn, rate_limits=rate_limits, **kwargs)

    monkeypatch.setattr(runner_mod, "execute_intents", guarded_execute)
    captured: dict[str, object] = {}
    real_equity = runner_mod.resolve_sizing_equity

    def equity_spy(snapshot, cap_usdt, **kwargs):
        captured["cash_usdt"] = kwargs["cash_usdt"]
        captured["positions"] = dict(kwargs["positions"])
        return real_equity(snapshot, cap_usdt, **kwargs)

    monkeypatch.setattr(runner_mod, "resolve_sizing_equity", equity_spy)
    save_ledger(ledger_path, LedgerState(
        positions={"AAAUSDT": Decimal("1")}, equity_high_water_mark=Decimal("2000"), cash_usdt=Decimal("1900"),
        funding_accrued_through=NOW - pd.Timedelta(hours=1),
    ))

    report = runner_mod.run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)

    # Delisting announcement alone supplies no settlement price: position and
    # cash stay unresolved, new risk is stopped (HALT, no orders executed).
    assert report.status == "HALT"
    assert "paper delisted holding unresolved" in report.reason
    assert executed == []
    assert load_ledger(ledger_path).positions == {"AAAUSDT": Decimal("1")}
    assert load_ledger(ledger_path).cash_usdt == Decimal("1900")
    assert [event for event, _ in alerts] == ["paper_delisted_unresolved"]
    assert "paper_delisted_unresolved" in (tmp_path / "shadow_cycle.jsonl").read_text(encoding="utf-8")


def test_run_shadow_cycle_paper_halts_when_held_funding_lag_exceeds_24h(tmp_path, monkeypatch) -> None:
    from decimal import Decimal
    import pandas as pd
    import src.live.runner as runner_mod
    from src.live.executor import ExecutionOutcome
    from src.live.ledger import LedgerState, PositionSnapshot, load_ledger, save_ledger
    from src.live.settings import ExecutionMode, LiveSettings
    from tests.unit.live._runner_stubs import DECISION_TIME, NOW, StubMarketClient, StubOrderClient

    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(runner_mod, "post_alert", lambda url, *, event, detail, decision_time, now: alerts.append((event, detail)) or False)
    monkeypatch.setattr(runner_mod, "send_email_alert", lambda **kwargs: False)
    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient())
    monkeypatch.setattr(runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl")

    def fake_execute(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        return tuple(
            ExecutionOutcome(symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100"), chases=0, status="FILLED")
            for i in intents
        )

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute)
    ledger_path = tmp_path / "ledger.json"
    settings = LiveSettings(
        mode=ExecutionMode.PAPER, notional_equity_usdt=2000.0, ledger_path=str(ledger_path),
        fills_dir=str(tmp_path / "fills"), orderbook_capture_enabled=False, microstructure_dir=str(tmp_path / "micro"),
        execution_quality_dir=str(tmp_path / "eq"), portfolio_state_dir=str(tmp_path / "port"), tax_ledger_dir=str(tmp_path / "tax"),
    )
    weights_path = tmp_path / "deployed_target_weights.parquet"
    pd.DataFrame({"BUSDT": [-0.02]}, index=pd.DatetimeIndex([DECISION_TIME])).to_parquet(weights_path, index=True)
    _seed_close_artifact(weights_path)

    calls: list[object] = []
    monkeypatch.setattr(runner_mod, "execute_intents", lambda *a, **k: calls.append(a) or ())
    monkeypatch.setattr(runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient())
    stale_epoch = NOW - pd.Timedelta(hours=30)
    monkeypatch.setattr(runner_mod, "_load_paper_funding", lambda symbols: {"AAAUSDT": pd.Series([0.001], index=pd.DatetimeIndex([stale_epoch]))})
    monkeypatch.setattr(runner_mod, "_load_paper_trade_closes", lambda symbols: {})
    save_ledger(ledger_path, LedgerState(
        positions={"AAAUSDT": Decimal("1")}, equity_high_water_mark=Decimal("2000"), cash_usdt=Decimal("1900"),
        funding_watermarks={"AAAUSDT": stale_epoch},
        position_history=(PositionSnapshot(effective_from=NOW - pd.Timedelta(hours=50), positions={"AAAUSDT": Decimal("1")}),),
    ))

    report = runner_mod.run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)

    assert report.status == "HALT"
    assert "paper funding lag exceeded symbols=AAAUSDT max_lag_h=30.0" in report.reason
    assert calls == []


def test_run_shadow_cycle_applies_orphan_settlements_before_reconcile(tmp_path, monkeypatch) -> None:
    from decimal import Decimal
    import pandas as pd
    import src.live.runner as runner_mod
    from src.live.executor import OrphanSettlement
    from src.live.ledger import load_ledger
    from src.live.settings import ExecutionMode, LiveSettings
    from tests.unit.live._runner_stubs import DECISION_TIME, NOW, StubMarketClient, StubOrderClient

    monkeypatch.setattr(runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient())
    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient())
    monkeypatch.setattr(runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl")
    monkeypatch.setattr(
        runner_mod,
        "cancel_orphan_orders",
        lambda client, run_id, audit, **kwargs: [
            OrphanSettlement(
                symbol="AAAUSDT",
                client_order_id=f"{run_id}-0",
                side="BUY",
                executed_qty=Decimal("0.5"),
                avg_price=Decimal("100"),
            )
        ],
    )
    monkeypatch.setattr(runner_mod, "execute_intents", lambda *args, **kwargs: ())
    monkeypatch.setattr(runner_mod, "_load_paper_funding", lambda symbols: {})
    monkeypatch.setattr(runner_mod, "_load_paper_trade_closes", lambda symbols: {})

    ledger_path = tmp_path / "ledger.json"
    settings = LiveSettings(
        mode=ExecutionMode.PAPER, notional_equity_usdt=2000.0, ledger_path=str(ledger_path),
    )
    weights_path = tmp_path / "deployed_target_weights.parquet"
    pd.DataFrame({"BUSDT": [-0.02]}, index=pd.DatetimeIndex([DECISION_TIME])).to_parquet(weights_path, index=True)
    _seed_close_artifact(weights_path)

    report = runner_mod.run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)

    assert report.status == "COMPLETE"
    assert load_ledger(ledger_path).positions["AAAUSDT"] == Decimal("0.5")


def test_run_shadow_cycle_wires_one_order_journal_into_orphan_cleanup_and_executor(tmp_path, monkeypatch) -> None:
    from decimal import Decimal
    import pandas as pd
    import src.live.runner as runner_mod
    from src.live.executor import ExecutionOutcome
    from src.live.order_journal import OrderJournal
    from src.live.runner import run_shadow_cycle
    from src.live.settings import ExecutionMode, LiveSettings

    decision_time = pd.Timestamp("2026-08-24 00:00Z")
    now = decision_time + pd.Timedelta(hours=2)
    artifact = tmp_path / "deployed_target_weights.parquet"
    pd.DataFrame({"AAAUSDT": [0.02]}, index=pd.DatetimeIndex([decision_time])).to_parquet(artifact, index=True)
    _seed_close_artifact(artifact)

    class MarketClient:
        def exchange_info(self):
            return {
                "symbols": [{
                    "symbol": "AAAUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT", "status": "TRADING",
                    "quantityPrecision": 3, "pricePrecision": 2,
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
                return {"totalWalletBalance": "2000", "availableBalance": "1900", "totalInitialMargin": "10",
                        "totalUnrealizedProfit": "0", "dualSidePosition": "false", "multiAssetsMargin": "false"}
            if path == "/fapi/v2/positionRisk":
                return []
            raise AssertionError(path)

        def sync_server_time(self):
            return None

        def open_orders(self):
            return []

    journal_path = tmp_path / "journal" / "live_order_journal.jsonl"
    monkeypatch.setattr(runner_mod, "_market_client", lambda s, dt: MarketClient())
    monkeypatch.setattr(runner_mod, "_order_client", lambda s, dt: OrderClient())
    monkeypatch.setattr(runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl")
    monkeypatch.setattr(runner_mod, "default_order_journal_path", lambda: journal_path)
    captured: dict[str, object] = {}

    def fake_orphans(client, client_order_prefix, audit, *, journal=None):
        captured["orphan_journal"] = journal
        captured["prefix"] = client_order_prefix
        return []

    def fake_execute(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        captured["execute_journal"] = kwargs.get("journal")
        return tuple(
            ExecutionOutcome(symbol=i.symbol, filled_qty=Decimal("0"), unfilled_qty=i.quantity, avg_fill_price=None, chases=0, status="SHADOW")
            for i in intents
        )

    monkeypatch.setattr(runner_mod, "cancel_orphan_orders", fake_orphans)
    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute)
    settings = LiveSettings(
        mode=ExecutionMode.SHADOW, notional_equity_usdt=2000.0, ledger_path=str(tmp_path / "ledger.json"),
        fills_dir=str(tmp_path / "fills"), microstructure_dir=str(tmp_path / "micro"),
        execution_quality_dir=str(tmp_path / "eq"), portfolio_state_dir=str(tmp_path / "port"),
        tax_ledger_dir=str(tmp_path / "tax"), orderbook_capture_enabled=False,
    )

    report = run_shadow_cycle(settings, decision_time, artifact, now=now)

    assert report.status == "COMPLETE"
    assert isinstance(captured["orphan_journal"], OrderJournal)
    assert captured["orphan_journal"] is captured["execute_journal"]
    assert captured["orphan_journal"].path == journal_path
    assert captured["prefix"] == "20260824"
    assert not journal_path.exists()


def test_run_shadow_cycle_paper_halts_when_held_symbol_absent_from_exchange(tmp_path, monkeypatch) -> None:
    from decimal import Decimal

    import pandas as pd

    import src.live.runner as runner_mod
    from src.live.ledger import LedgerState, load_ledger, save_ledger
    from src.live.settings import ExecutionMode, LiveSettings
    from tests.unit.live._runner_stubs import DECISION_TIME, NOW, StubMarketClient, StubOrderClient

    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(runner_mod, "post_alert", lambda url, *, event, detail, decision_time, now: alerts.append((event, detail)) or False)
    monkeypatch.setattr(runner_mod, "send_email_alert", lambda **kwargs: False)
    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient())
    monkeypatch.setattr(runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient())
    monkeypatch.setattr(runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl")
    executed: list[object] = []
    monkeypatch.setattr(runner_mod, "execute_intents", lambda *a, **k: executed.append(a) or ())
    accrued: list[object] = []
    monkeypatch.setattr(runner_mod, "_load_paper_funding", lambda symbols: accrued.append(symbols) or {})
    monkeypatch.setattr(runner_mod, "_load_paper_trade_closes", lambda symbols: {})
    ledger_path = tmp_path / "ledger.json"
    settings = LiveSettings(
        mode=ExecutionMode.PAPER, notional_equity_usdt=2000.0, ledger_path=str(ledger_path),
        fills_dir=str(tmp_path / "fills"), orderbook_capture_enabled=False, microstructure_dir=str(tmp_path / "micro"),
        execution_quality_dir=str(tmp_path / "eq"), portfolio_state_dir=str(tmp_path / "port"), tax_ledger_dir=str(tmp_path / "tax"),
    )
    weights_path = tmp_path / "deployed_target_weights.parquet"
    pd.DataFrame({"BUSDT": [-0.02]}, index=pd.DatetimeIndex([DECISION_TIME])).to_parquet(weights_path, index=True)
    _seed_close_artifact(weights_path)
    initial = LedgerState(positions={"GONEUSDT": Decimal("5"), "AAAUSDT": Decimal("1")}, equity_high_water_mark=Decimal("2000"), cash_usdt=Decimal("1900"))
    save_ledger(ledger_path, initial)

    # When: 보유 심볼 GONEUSDT 가 exchangeInfo 에서 완전히 사라짐(StubMarketClient 는 AAAUSDT/BUSDT 만)
    report = runner_mod.run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)

    # Then: 조용한 MTM 누락/펀딩 지연 HALT 루프 대신 원인 명시 HALT, 원장·집행 불변
    assert report.status == "HALT"
    assert "held symbols absent from exchangeInfo symbols=GONEUSDT" in report.reason
    assert executed == []
    assert accrued == []
    assert load_ledger(ledger_path).positions == initial.positions
    assert "held_symbol_absent" in (tmp_path / "shadow_cycle.jsonl").read_text(encoding="utf-8")



def test_runner_missing_anchor_blocks_orders(artifact, live_env, tmp_path) -> None:
    """Nonzero targets with no decision-close artifact fail before intents."""
    from src.live.deployed_weights import decision_ohlcv_close_path

    decision_ohlcv_close_path(artifact).unlink()
    settings = LiveSettings(notional_equity_usdt=2000.0, ledger_path=str(tmp_path / "ledger_missing.json"))
    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)
    assert report.status == "HALT"
    assert "missing" in (report.reason or "")
    assert report.intent_count == 0
    assert live_env == []


def test_runner_sparse_anchor_blocks_active_target(artifact, live_env, tmp_path) -> None:
    """A sparse roster row must still contain every nonzero target close."""
    from src.live.deployed_weights import decision_ohlcv_close_path

    pd.DataFrame(
        {"AAAUSDT": [100.0]}, index=pd.DatetimeIndex([DECISION_TIME]),
    ).to_parquet(decision_ohlcv_close_path(artifact), index=True)
    settings = LiveSettings(notional_equity_usdt=2000.0, ledger_path=str(tmp_path / "ledger_sparse.json"))
    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)
    assert report.status == "HALT"
    assert "missing for active targets: BUSDT" in (report.reason or "")
    assert report.intent_count == 0
    assert live_env == []


def test_runner_old_mark_artifact_ignored(artifact, live_env, tmp_path) -> None:
    """Only a legacy decision-marks artifact never restores sizing."""
    import pandas as pd

    from src.live.deployed_weights import decision_ohlcv_close_path

    decision_ohlcv_close_path(artifact).unlink()
    pd.DataFrame(
        {"AAAUSDT": [999.0], "BUSDT": [999.0]}, index=pd.DatetimeIndex([DECISION_TIME]),
    ).to_parquet(artifact.parent / "deployed_decision_marks.parquet", index=True)
    settings = LiveSettings(notional_equity_usdt=2000.0, ledger_path=str(tmp_path / "ledger_legacymark.json"))
    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)
    assert report.status == "HALT"
    assert report.intent_count == 0
    assert live_env == []


def test_runner_stale_anchor_row_fails(artifact, live_env, tmp_path) -> None:
    """A prior-day close row never forward-fills the current decision."""
    import pandas as pd

    from src.live.deployed_weights import decision_ohlcv_close_path

    pd.DataFrame(
        {"AAAUSDT": [100.0], "BUSDT": [100.0]},
        index=pd.DatetimeIndex([DECISION_TIME - pd.Timedelta(days=1)]),
    ).to_parquet(decision_ohlcv_close_path(artifact), index=True)
    settings = LiveSettings(notional_equity_usdt=2000.0, ledger_path=str(tmp_path / "ledger_stale.json"))
    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)
    assert report.status == "HALT"
    assert "not present" in (report.reason or "")
    assert report.intent_count == 0
    assert live_env == []


def test_runner_current_ticker_remains_execution_check(artifact, live_env, tmp_path, monkeypatch) -> None:
    """A valid decision close with no current ticker stays NOT_TRADABLE."""
    settings = LiveSettings(notional_equity_usdt=2000.0, ledger_path=str(tmp_path / "ledger_notradable.json"))
    monkeypatch.setattr(runner_mod, "_marks_from_tickers", lambda client, symbols: {})
    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)
    assert report.status == "COMPLETE"
    assert report.intent_count == 0


def test_reader_missing_anchor_fails(tmp_path) -> None:
    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.signal import latest_decision_ohlcv_close

    weights_path = tmp_path / "deployed_target_weights.parquet"
    pd.DataFrame({"AAAUSDT": [0.02]}, index=pd.DatetimeIndex([DECISION_TIME])).to_parquet(weights_path, index=True)
    with pytest.raises(DataIntegrityError, match="missing"):
        latest_decision_ohlcv_close(weights_path, DECISION_TIME)


def test_reader_empty_anchor_fails(tmp_path) -> None:
    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.deployed_weights import decision_ohlcv_close_path
    from src.live.signal import latest_decision_ohlcv_close

    weights_path = tmp_path / "deployed_target_weights.parquet"
    pd.DataFrame({"AAAUSDT": [0.02]}, index=pd.DatetimeIndex([DECISION_TIME])).to_parquet(weights_path, index=True)
    pd.DataFrame(columns=["AAAUSDT"]).to_parquet(decision_ohlcv_close_path(weights_path), index=True)
    with pytest.raises(DataIntegrityError, match="empty"):
        latest_decision_ohlcv_close(weights_path, DECISION_TIME)


def test_reader_corrupt_anchor_fails(tmp_path) -> None:
    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.deployed_weights import decision_ohlcv_close_path
    from src.live.signal import latest_decision_ohlcv_close

    weights_path = tmp_path / "deployed_target_weights.parquet"
    pd.DataFrame({"AAAUSDT": [0.02]}, index=pd.DatetimeIndex([DECISION_TIME])).to_parquet(weights_path, index=True)
    decision_ohlcv_close_path(weights_path).write_bytes(b"not a parquet")
    with pytest.raises(DataIntegrityError, match="unreadable"):
        latest_decision_ohlcv_close(weights_path, DECISION_TIME)


def test_reader_naive_index_fails(tmp_path) -> None:
    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.deployed_weights import decision_ohlcv_close_path
    from src.live.signal import latest_decision_ohlcv_close

    weights_path = tmp_path / "deployed_target_weights.parquet"
    pd.DataFrame({"AAAUSDT": [0.02]}, index=pd.DatetimeIndex([DECISION_TIME])).to_parquet(weights_path, index=True)
    pd.DataFrame({"AAAUSDT": [100.0]}, index=pd.DatetimeIndex([pd.Timestamp("2026-08-24")])).to_parquet(
        decision_ohlcv_close_path(weights_path), index=True,
    )
    with pytest.raises(DataIntegrityError, match="tz-aware"):
        latest_decision_ohlcv_close(weights_path, DECISION_TIME)


def test_reader_missing_row_fails(tmp_path) -> None:
    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.deployed_weights import decision_ohlcv_close_path
    from src.live.signal import latest_decision_ohlcv_close

    weights_path = tmp_path / "deployed_target_weights.parquet"
    pd.DataFrame({"AAAUSDT": [0.02]}, index=pd.DatetimeIndex([DECISION_TIME])).to_parquet(weights_path, index=True)
    pd.DataFrame({"AAAUSDT": [100.0]}, index=pd.DatetimeIndex([DECISION_TIME - pd.Timedelta(days=1)])).to_parquet(
        decision_ohlcv_close_path(weights_path), index=True,
    )
    with pytest.raises(DataIntegrityError, match="not present"):
        latest_decision_ohlcv_close(weights_path, DECISION_TIME)


def test_reader_invalid_values_fail(tmp_path) -> None:
    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.deployed_weights import decision_ohlcv_close_path
    from src.live.signal import latest_decision_ohlcv_close

    weights_path = tmp_path / "deployed_target_weights.parquet"
    pd.DataFrame({"AAAUSDT": [0.02]}, index=pd.DatetimeIndex([DECISION_TIME])).to_parquet(weights_path, index=True)
    for bad in (0.0, float("nan"), -5.0):
        pd.DataFrame({"AAAUSDT": [bad]}, index=pd.DatetimeIndex([DECISION_TIME])).to_parquet(
            decision_ohlcv_close_path(weights_path), index=True,
        )
        with pytest.raises(DataIntegrityError, match="invalid"):
            latest_decision_ohlcv_close(weights_path, DECISION_TIME)


def test_reader_duplicate_rows_fail(tmp_path) -> None:
    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.deployed_weights import decision_ohlcv_close_path
    from src.live.signal import latest_decision_ohlcv_close

    weights_path = tmp_path / "deployed_target_weights.parquet"
    pd.DataFrame({"AAAUSDT": [0.02]}, index=pd.DatetimeIndex([DECISION_TIME])).to_parquet(weights_path, index=True)
    dup = pd.DataFrame(
        {"AAAUSDT": [100.0, 101.0]}, index=pd.DatetimeIndex([DECISION_TIME, DECISION_TIME]),
    )
    dup.to_parquet(decision_ohlcv_close_path(weights_path), index=True)
    with pytest.raises(DataIntegrityError, match="duplicate"):
        latest_decision_ohlcv_close(weights_path, DECISION_TIME)


def test_reader_rejects_non_weights_token(tmp_path) -> None:
    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.signal import latest_decision_ohlcv_close

    other = tmp_path / "other.parquet"
    pd.DataFrame({"AAAUSDT": [100.0]}, index=pd.DatetimeIndex([DECISION_TIME])).to_parquet(other, index=True)
    with pytest.raises(DataIntegrityError, match="missing token"):
        latest_decision_ohlcv_close(other, DECISION_TIME)


def test_runner_sealed_anchor_roundtrip(tmp_path) -> None:
    """A sealed close artifact reads with the valid key and fails closed otherwise."""
    import base64

    import pytest
    from pydantic import SecretStr

    from src.live.deployed_weights import append_weight_row, decision_ohlcv_close_path
    from src.live.errors import ArtifactSealError
    from src.live.signal import latest_decision_ohlcv_close

    weights_path = tmp_path / "deployed_target_weights.parquet"
    pd.DataFrame({"AAAUSDT": [0.02]}, index=pd.DatetimeIndex([DECISION_TIME])).to_parquet(weights_path, index=True)
    key = SecretStr(base64.b64encode(b"0" * 32).decode())
    assert append_weight_row(
        decision_ohlcv_close_path(weights_path), DECISION_TIME,
        pd.Series({"AAAUSDT": 100.0}, dtype="float64"), artifact_key=key,
    ) is True
    got = latest_decision_ohlcv_close(weights_path, DECISION_TIME, artifact_key=key)
    assert got.to_dict() == {"AAAUSDT": 100.0}
    assert got.dtype == "float64"
    with pytest.raises(ArtifactSealError):
        latest_decision_ohlcv_close(weights_path, DECISION_TIME, artifact_key=SecretStr(base64.b64encode(b"1" * 32).decode()))
    with pytest.raises(ArtifactSealError):
        latest_decision_ohlcv_close(weights_path, DECISION_TIME)


def test_run_shadow_cycle_skips_depth_when_disabled(tmp_path, monkeypatch) -> None:
    settings, _ = _depth_cycle_settings(tmp_path, monkeypatch, exec_depth_capture_enabled=False)
    weights_path = _three_symbol_weights(tmp_path)
    report = run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)
    assert report.status == "COMPLETE"
    assert _StubDepthRecorder.instances == []
    lines = (tmp_path / "shadow_cycle.jsonl").read_text(encoding="utf-8").splitlines()
    assert any(
        json.loads(line).get("event") == "exec_depth_skipped"
        and json.loads(line).get("reason") == "disabled"
        for line in lines
    )


def test_run_shadow_cycle_unexpected_error_stops_recorder(tmp_path, monkeypatch) -> None:
    settings, _ = _depth_cycle_settings(tmp_path, monkeypatch)

    def _boom(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(runner_mod, "execute_intents", _boom)
    weights_path = _three_symbol_weights(tmp_path)
    with pytest.raises(RuntimeError, match="unexpected"):
        run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)
    assert _StubDepthRecorder.instances[0].stops == [0.0]


def test_fill_events_record_real_time(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from decimal import Decimal
    from src.live.runner import run_shadow_cycle
    from src.live.settings import LiveSettings
    import src.live.runner as runner_mod
    from src.live.executor import ExecutionOutcome

    decision_time = pd.Timestamp("2026-01-01 00:00Z")
    filled_at = pd.Timestamp("2026-01-01 23:07:12", tz="UTC")
    frame = pd.DataFrame({"AAAUSDT": [0.02]}, index=pd.DatetimeIndex([decision_time]))
    artifact = tmp_path / "deployed_target_weights.parquet"
    frame.to_parquet(artifact, index=True)
    _seed_close_artifact(artifact)

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
        return tuple(
            ExecutionOutcome(symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100"), chases=0, status="FILLED", fills=((i.quantity, Decimal("100"), 2.0, "maker_fill", "maker", filled_at),))
            for i in intents
        )
    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute)

    fills_dir = tmp_path / "fills"
    settings = LiveSettings(mode="paper", notional_equity_usdt=2000.0, ledger_path=str(tmp_path / "ledger.json"), fills_dir=str(fills_dir))
    report = run_shadow_cycle(settings, decision_time, artifact, now=decision_time + pd.Timedelta(hours=2))
    assert report.status == "COMPLETE"
    from src.live.fills import load_fills
    df = load_fills(fills_dir)
    assert len(df) == 1
    assert pd.Timestamp(df["timestamp"].iloc[0]).tz_convert("UTC") == filled_at
    assert pd.Timestamp(df["decision_time"].iloc[0]).tz_convert("UTC") == decision_time
