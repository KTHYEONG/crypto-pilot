"""Universe integrity: OPEN_EDGE/UNSCOPED fail closed, INTERIOR withdraws, LISTING_EDGE silent."""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.core.source_gaps import SourceGapInterval
from src.core.types import ExecutionSpec
from src.strategy.targets import FLOW_MOM_TOP20

import src.engine.strategy_backtest as run_mod
from src.engine.strategy_backtest import (
    LakeCoverageError,
    assert_lake_coverage,
    strategy_blocked_decisions,
    strategy_interior_withdrawals,
)
from src.core.instrument_settlements import EMPTY_SETTLEMENT_REGISTRY

import dataclasses


_DAY1 = pd.Timestamp("2021-06-01", tz="UTC")
_SYMBOLS = ("AAA", "BBB")


def _frames(grid: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    cols = list(_SYMBOLS)
    return {
        "highs": pd.DataFrame(101.0, index=grid, columns=cols, dtype="float64"),
        "lows": pd.DataFrame(99.0, index=grid, columns=cols, dtype="float64"),
        "closes": pd.DataFrame(100.0, index=grid, columns=cols, dtype="float64"),
        "marks": pd.DataFrame(100.0, index=grid, columns=cols, dtype="float64"),
        "bar_funding": pd.DataFrame(0.0, index=grid, columns=cols, dtype="float64"),
        "quote_volumes": pd.DataFrame(1000.0, index=grid, columns=cols, dtype="float64"),
        "funding_known": pd.DataFrame(True, index=grid, columns=cols),
    }


def _windows(candidate, labels: list[pd.Timestamp]) -> list:
    from src.engine.execution import ExecutionReplayWindow

    out = []
    first = pd.date_range(labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=1), freq="3min", tz="UTC")
    second = pd.date_range(labels[1] - pd.Timedelta(hours=2), labels[2] + pd.Timedelta(hours=2), freq="3min", tz="UTC")
    for grid, chosen in ((first, labels[:1]), (second, labels[1:])):
        weights = candidate.target_weights.loc[chosen].copy()
        avail = pd.DatetimeIndex(
            [candidate.signal_available_at[candidate.target_weights.index.get_loc(label)] for label in chosen],
            tz="UTC",
        )
        params: dict[str, object] = {
            "window_start": grid[0], "window_end": grid[-1], "columns": _SYMBOLS, "symbols": _SYMBOLS,
            "minute_grid": grid, "target_weights": weights, "signal_available_at": avail,
            "bar_available_at": grid + pd.Timedelta(minutes=3),
        }
        params.update(_frames(grid))
        out.append(ExecutionReplayWindow(**params))  # type: ignore[arg-type]
    return out


def _specs() -> tuple[ExecutionSpec, ExecutionSpec]:
    base = dataclasses.replace(
        ExecutionSpec(), taker_fee_bps=5.0, taker_slippage_bps=1.0, decision_anchor="submit_bar",
    )
    stress = dataclasses.replace(
        ExecutionSpec(), taker_fee_bps=5.0, taker_slippage_bps=13.0, decision_anchor="submit_bar",
    )
    return base, stress


def _interval(symbol: str, extent: str, start: str, end: str | None = None) -> SourceGapInterval:
    start_dt = pd.Timestamp(start, tz="UTC").to_pydatetime().astimezone(UTC)
    end_dt = pd.Timestamp(end, tz="UTC").to_pydatetime().astimezone(UTC) if end else None
    return SourceGapInterval(
        symbol=symbol, plane="ohlcv_3m", start=start_dt, end=end_dt, reason="SOURCE_ABSENT",
        evidence="universe fixture", verified_at=pd.Timestamp("2026-01-01T00:00:00Z").to_pydatetime(),
        resolved_at=None, extent=extent,  # type: ignore[arg-type]
    )


def _roster(days: pd.DatetimeIndex, symbols: tuple[str, ...], selected: str) -> pd.DataFrame:
    frame = pd.DataFrame(False, index=days, columns=list(symbols), dtype=bool)
    frame[selected] = True
    return frame


def test_open_edge_on_selected_symbol_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A roster seat crossing an unresolved OPEN_EDGE names the symbol and first day."""
    base, _ = _specs()
    monkeypatch.setattr(
        run_mod, "active_intervals",
        lambda **kwargs: (_interval("AAA", "OPEN_EDGE", "2021-04-03T00:00:00Z"),),
    )
    days = pd.date_range("2021-04-01", periods=5, freq="D", tz="UTC")
    roster = _roster(days, ("AAA", "BBB"), "AAA")
    with pytest.raises(LakeCoverageError, match=r"AAA.*2021-04-01|2021-04-01.*AAA"):
        assert_lake_coverage(
            days, roster, strategy=FLOW_MOM_TOP20, base_spec=base,
            settlement_registry=EMPTY_SETTLEMENT_REGISTRY,
            evaluation_start=pd.Timestamp("2021-04-01", tz="UTC"),
            evaluation_end=pd.Timestamp("2021-04-06", tz="UTC"),
        )
    with pytest.raises(LakeCoverageError, match=r"data collect.*data verify-source-gaps"):
        assert_lake_coverage(
            days, roster, strategy=FLOW_MOM_TOP20, base_spec=base,
            settlement_registry=EMPTY_SETTLEMENT_REGISTRY,
            evaluation_start=pd.Timestamp("2021-04-01", tz="UTC"),
            evaluation_end=pd.Timestamp("2021-04-06", tz="UTC"),
        )


def test_open_edge_on_never_selected_symbol_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """An OPEN_EDGE gap on a symbol the roster never grants proceeds without withdrawal."""
    base, _ = _specs()
    monkeypatch.setattr(
        run_mod, "active_intervals",
        lambda **kwargs: (_interval("AAA", "OPEN_EDGE", "2021-04-03T00:00:00Z"),),
    )
    days = pd.date_range("2021-04-01", periods=5, freq="D", tz="UTC")
    roster = _roster(days, ("AAA", "BBB"), "BBB")
    assert_lake_coverage(
        days, roster, strategy=FLOW_MOM_TOP20, base_spec=base,
        settlement_registry=EMPTY_SETTLEMENT_REGISTRY,
        evaluation_start=pd.Timestamp("2021-04-01", tz="UTC"),
        evaluation_end=pd.Timestamp("2021-04-06", tz="UTC"),
    )
    frame = strategy_blocked_decisions(
        days, ("AAA", "BBB"), strategy=FLOW_MOM_TOP20, base_spec=base,
        settlement_registry=EMPTY_SETTLEMENT_REGISTRY,
    )
    assert not bool(frame.to_numpy().any())


def test_interior_gap_withdraws_and_is_disclosed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """INTERIOR gaps withdraw the seat and surface in the disclosure list and limitation."""
    from src.engine.backtest_evidence import StrategyReportPeriod, evaluate_strategy_backtest
    from src.engine.backtest_persist import strategy_backtest_payload
    from src.engine.strategy_backtest import StrategyBacktestRequest
    from src.strategy.targets import StrategyTargets
    from src.core.source_gaps import active_intervals
    from src.market_data.services.source_gap_audit import (
        _record_to_row, audit_source_gap_registry, write_audited_registry,
    )
    import json

    base, stress = _specs()
    original = _interval("AAA", "UNSCOPED", "2021-04-03T12:00:00Z", "2021-04-04T00:00:00Z")
    registry = tmp_path / "reg.jsonl"
    registry.write_text(json.dumps(_record_to_row(original)) + "\n", encoding="utf-8")
    root = tmp_path / "lake"
    directory = root / "ohlcv" / "3m"
    directory.mkdir(parents=True)
    observed = pd.DatetimeIndex(["2021-04-03T11:57:00Z", "2021-04-04T00:00:00Z"])
    pd.DataFrame({"datetime": observed}).to_parquet(directory / "AAA.parquet", index=False)
    report = audit_source_gap_registry(
        plane="ohlcv_3m", start=observed[0], end=observed[-1] + pd.Timedelta(minutes=3),
        registry_path=registry, data_root=root,
    )
    assert len(report.rescoped) == 1
    write_audited_registry(report, registry_path=registry, verified_at=pd.Timestamp("2026-01-01T00:00:00Z"))
    monkeypatch.setattr(
        run_mod, "active_intervals",
        lambda **kwargs: active_intervals(path=registry, **kwargs),
    )
    days = pd.date_range("2021-04-01", periods=5, freq="D", tz="UTC")
    assert_lake_coverage(
        days, _roster(days, _SYMBOLS, "AAA"), strategy=FLOW_MOM_TOP20, base_spec=base,
        settlement_registry=EMPTY_SETTLEMENT_REGISTRY,
        evaluation_start=days[0], evaluation_end=days[-1] + pd.Timedelta(days=1),
    )
    frame = strategy_blocked_decisions(
        days, ("AAA", "BBB"), strategy=FLOW_MOM_TOP20, base_spec=base,
        settlement_registry=EMPTY_SETTLEMENT_REGISTRY,
    )
    assert bool(frame.loc[pd.Timestamp("2021-04-02", tz="UTC"), "AAA"])
    counts, _ = strategy_interior_withdrawals(
        days, ("AAA", "BBB"), strategy=FLOW_MOM_TOP20, base_spec=base,
        settlement_registry=EMPTY_SETTLEMENT_REGISTRY,
    )
    assert counts == {"AAA": 1}
    withdrawals = tuple(
        {"symbol": sym, "extent": "INTERIOR", "days": n} for sym, n in sorted(counts.items())
    )

    labels = [_DAY1 + pd.Timedelta(days=i) for i in (1, 2, 3)]
    weights = pd.DataFrame(
        {"AAA": [0.05] * len(labels), "BBB": [-0.05] * len(labels)},
        index=pd.DatetimeIndex(labels, tz="UTC"), dtype="float64",
    )
    avail = pd.DatetimeIndex([label - pd.Timedelta(hours=1) for label in labels], tz="UTC")
    candidate = StrategyTargets(target_weights=weights, signal_available_at=avail, strategy=FLOW_MOM_TOP20)
    windows = _windows(candidate, labels)
    probe = (
        StrategyReportPeriod(label="probe", start=pd.Timestamp("2022-01-01", tz="UTC"), end=pd.Timestamp("2022-01-02", tz="UTC")),
    )
    evidence_probe = evaluate_strategy_backtest(
        candidate, iter(windows), initial_equity=100000.0,
        base_spec=base, stress_spec=stress, report_periods=probe,
    )
    covered = evidence_probe.base_daily.returns.index
    periods = (StrategyReportPeriod(label="P1", start=covered[0], end=covered[1]),)
    evidence = evaluate_strategy_backtest(
        candidate, iter(_windows(candidate, labels)), initial_equity=100000.0,
        base_spec=base, stress_spec=stress, report_periods=periods,
    )
    request = StrategyBacktestRequest(
        source_start=_DAY1, evaluation_start=labels[0], evaluation_end=labels[-1] + pd.Timedelta(days=1),
        strategy=FLOW_MOM_TOP20, initial_equity=100000.0,
        base_spec=base, stress_spec=stress, report_periods=periods,
    )
    run = run_mod.StrategyBacktestRun(
        request=request, candidate=candidate, evidence=evidence,
        execution_start=labels[0], execution_end=labels[-1] + pd.Timedelta(days=1),
        source_symbols=("AAA", "BBB"), data_availability_withdrawals=withdrawals,
    )
    payload = strategy_backtest_payload(run)
    assert payload["data_availability_withdrawals"] == [
        {"symbol": "AAA", "extent": "INTERIOR", "days": 1}
    ]
    assert "DATA_AVAILABILITY_SELECTION" in payload["limitations"]


def test_listing_edge_is_not_disclosed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-listing absence withdraws the seat but produces no disclosure entry."""
    base, _ = _specs()
    monkeypatch.setattr(
        run_mod, "active_intervals",
        lambda **kwargs: (_interval("AAA", "LISTING_EDGE", "2020-01-01T00:00:00Z", "2021-04-04T00:00:00Z"),),
    )
    days = pd.date_range("2021-04-01", periods=5, freq="D", tz="UTC")
    frame = strategy_blocked_decisions(
        days, ("AAA", "BBB"), strategy=FLOW_MOM_TOP20, base_spec=base,
        settlement_registry=EMPTY_SETTLEMENT_REGISTRY,
    )
    assert bool(frame.loc[pd.Timestamp("2021-04-01", tz="UTC"), "AAA"])
    counts, _ = strategy_interior_withdrawals(
        days, ("AAA", "BBB"), strategy=FLOW_MOM_TOP20, base_spec=base,
        settlement_registry=EMPTY_SETTLEMENT_REGISTRY,
    )
    assert counts == {}


def test_lake_coverage_error_is_data_integrity_error() -> None:
    assert issubclass(LakeCoverageError, DataIntegrityError)


def _delisting_record(symbol: str, announced: pd.Timestamp, last_trade: pd.Timestamp):
    from src.core.instrument_settlements import (
        InstrumentSettlementRecord,
        assemble_instrument_settlement_registry,
    )

    record = InstrumentSettlementRecord(
        symbol=symbol,
        event_id=f"{symbol}:{int(last_trade.value // 1_000_000)}",
        announced_at=announced,
        announcement_source="curated",
        announcement_evidence="Binance delisting notice",
        last_trade_at=last_trade,
        delivery_at=last_trade,
        settlement_price=100.0,
        price_source="curated",
        price_evidence="test",
        fee_bps=5.0,
        evidence_digest="sha256:test",
        verified_at=pd.Timestamp("2026-01-01T00:00:00Z"),
    )
    return assemble_instrument_settlement_registry([record], [])


def _delisting_block(days: pd.DatetimeIndex, registry) -> pd.Series:
    from src.engine.strategy_backtest import _strategy_delisting_block

    values = _strategy_delisting_block(
        days, ["AAA"], {"AAA": 0}, snapshot_hour=0, settlement_registry=registry,
    )
    return pd.Series(values[:, 0], index=days)


def test_short_lead_delisting_withdraws_only_after_announcement() -> None:
    delivery = pd.Timestamp("2021-05-10T00:00:00Z")
    announced = delivery - pd.Timedelta(days=1.9)
    days = pd.date_range("2021-05-01", periods=10, freq="D", tz="UTC")
    blocked = _delisting_block(days, _delisting_record("AAA", announced, delivery))
    first_announced = days[days >= announced][0]
    assert not bool(blocked.loc[days < first_announced].any())
    assert bool(blocked.loc[days >= first_announced].all())
    assert first_announced == pd.Timestamp("2021-05-09", tz="UTC")


def test_long_lead_delisting_matches_proxy_lead_block() -> None:
    delivery = pd.Timestamp("2021-05-10T00:00:00Z")
    days = pd.date_range("2021-05-01", periods=10, freq="D", tz="UTC")
    evidenced = _delisting_block(
        days, _delisting_record("AAA", delivery - pd.Timedelta(days=6), delivery),
    )
    proxy = _delisting_block(
        days, _delisting_record("AAA", delivery - pd.Timedelta(days=7), delivery),
    )
    assert evidenced.equals(proxy)
    assert evidenced.loc[pd.Timestamp("2021-05-07", tz="UTC")]
    assert not evidenced.loc[pd.Timestamp("2021-05-06", tz="UTC")]
