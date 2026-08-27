"""SCENARIO_LIVE_45/46: 가상 MTM equity 선택 규칙 + 타입 있는 parquet 기록/로테이션."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pandas as pd

from src.live.portfolio_state import (
    PortfolioStateRecord,
    append_portfolio_state,
    resolve_effective_equity,
    summarize_portfolio_state,
    virtual_mtm_equity,
)
from src.live.settings import ExecutionMode

DECISION_TIME = pd.Timestamp("2026-08-24 00:00Z")


def _record(**overrides) -> PortfolioStateRecord:
    defaults = {
        "decision_time": DECISION_TIME,
        "mode": "paper",
        "equity_usdt": 2000.0,
        "equity_source": "virtual_mtm",
        "cash_usdt": 1900.0,
        "wallet_balance_usdt": None,
        "unrealized_pnl_usdt": None,
        "equity_high_water_mark_usdt": 2000.0,
        "gross_notional_usdt": 100.0,
        "n_holdings": 1,
        "intent_count": 1,
        "dropped_notional_fraction": 0.0,
    }
    defaults.update(overrides)
    return PortfolioStateRecord(**defaults)


def test_SCENARIO_LIVE_45_VIRTUAL_EQUITY_ONLY_FOR_SUPPRESSED_MODES() -> None:
    positions = {"AAAUSDT": Decimal("1")}
    marks = {"AAAUSDT": Decimal("100")}

    live_equity, live_source = resolve_effective_equity(
        ExecutionMode.LIVE_TESTNET, Decimal("2000"), Decimal("999"),
        {"AAAUSDT": Decimal("1")}, marks,
    )
    assert (live_equity, live_source) == (Decimal("2000"), "venue")

    paper_equity, paper_source = resolve_effective_equity(
        ExecutionMode.PAPER, Decimal("999999"), Decimal("1500"),
        {"AAAUSDT": Decimal("2")}, marks,
    )
    assert (paper_equity, paper_source) == (Decimal("1700"), "virtual_mtm")

    # marks 없는 심볼은 0으로 취급된다(가격 조작 없음).
    assert virtual_mtm_equity(
        Decimal("500"), {"AAAUSDT": Decimal("1"), "BBBUSDT": Decimal("3")}, {"AAAUSDT": Decimal("100")}
    ) == Decimal("600")
    del positions


def test_SCENARIO_LIVE_46_PORTFOLIO_STATE_APPENDS_TYPED_PARQUET_AND_ROTATES(tmp_path: Path, monkeypatch) -> None:
    history_dir = tmp_path / "portfolio_state"
    append_portfolio_state(_record(), history_dir)

    df = pd.read_parquet(history_dir / "active.parquet")
    assert isinstance(df["decision_time"].dtype, pd.DatetimeTZDtype)
    assert df["equity_usdt"].dtype == "float64"
    assert df["gross_notional_usdt"].dtype == "float64"
    assert df["n_holdings"].dtype == "int64"
    assert df["intent_count"].dtype == "int64"

    import src.live.portfolio_state as ps_mod

    monkeypatch.setattr(ps_mod, "RUN_HISTORY_SHARD_MAX_BYTES", 1)
    append_portfolio_state(_record(decision_time=DECISION_TIME + pd.Timedelta(days=1)), history_dir)
    archives = list(history_dir.glob("portfolio_state_*.parquet"))
    assert len(archives) == 1

    empty_summary = summarize_portfolio_state(tmp_path / "does_not_exist")
    assert empty_summary["n_cycles"] == 0
    assert empty_summary["by_mode"] == {}

    summary = summarize_portfolio_state(history_dir)
    assert summary["n_cycles"] >= 1


#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_45_VIRTUAL_EQUITY_ONLY_FOR_SUPPRESSED_MODES",
    "SCENARIO_LIVE_46_PORTFOLIO_STATE_APPENDS_TYPED_PARQUET_AND_ROTATES",
)
