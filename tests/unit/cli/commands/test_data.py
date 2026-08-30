"""Contract coverage for the ``data collect mhs-execution`` CLI defaults."""

from __future__ import annotations

import argparse

import pytest

from src.cli.commands.data import _mhs_execution, _refresh_live_universe, add_data_commands


def _mhs_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_data_commands(parser.add_subparsers(dest="group", required=True).add_parser("data"))
    return parser


def test_data_collect_mhs_execution_timeframe_defaults_to_3m() -> None:
    # SCENARIO_MHS_EXECUTION_PLAN_3M_DEFAULT (CLI side): unqualified
    # ``data collect mhs-execution`` defaults to the native 3m interval.
    parser = _mhs_parser()
    args = parser.parse_args(["data", "collect", "mhs-execution"])
    assert args.timeframe == "3m"


def test_data_collect_mhs_execution_accepts_3m_and_rejects_out_of_contract() -> None:
    parser = _mhs_parser()
    assert parser.parse_args(["data", "collect", "mhs-execution", "--timeframe", "3m"]).timeframe == "3m"
    with pytest.raises(SystemExit):
        parser.parse_args(["data", "collect", "mhs-execution", "--timeframe", "7m"])


def test_data_collect_mhs_execution_threads_timeframe_to_plan(monkeypatch) -> None:
    # SCENARIO_MHS_EXECUTION_PLAN_3M_DEFAULT: the CLI default threads into
    # ``build_mhs_execution_plan`` as the collection interval.
    import src.application.data.mhs_execution_collection as mc

    captured: dict = {}
    plan = mc.MhsExecutionCollectionPlan(
        timeframe="3m", start="2025-01-01", end="2025-03-30",
        execution_universe_size=8, symbols=("S00",), manifest_path="plan.json",
    )

    def _spy_plan(start, end, timeframe, execution_universe_size):
        captured.update(
            start=start, end=end, timeframe=timeframe,
            execution_universe_size=execution_universe_size,
        )
        return plan

    monkeypatch.setattr(mc, "build_mhs_execution_plan", _spy_plan)
    monkeypatch.setattr(
        mc, "collect_mhs_execution_data",
        lambda plan, execute=False, workers=4: {"mode": "dry_run"},
    )

    parser = _mhs_parser()
    args = parser.parse_args(["data", "collect", "mhs-execution"])
    _mhs_execution(args)
    assert captured["timeframe"] == "3m"


def test_data_refresh_live_universe_registered_and_dispatches(monkeypatch) -> None:
    """SCENARIO_SIGNAL_10 (data side): ``data collect refresh-live-universe``
    parses and dispatches to _refresh_live_universe, refreshing 1h/funding for
    every cached symbol before the 3m roster (order matters: the roster ranks
    by trailing 1h volume)."""
    parser = _mhs_parser()
    args = parser.parse_args(["data", "refresh-live-universe"])
    assert args.handler is _refresh_live_universe

    import glob as glob_mod

    monkeypatch.setattr(glob_mod, "glob", lambda pattern: ["AAAUSDT.parquet", "BUSDT.parquet"])

    call_order: list[str] = []

    class FakeCollector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            call_order.append(f"ohlcv:{symbol}")

        def ensure_funding_data(self, symbol, start, end):
            call_order.append(f"funding:{symbol}")

    monkeypatch.setattr(
        "src.market_data.services.futures_collection.DataCollector", FakeCollector
    )

    class FakePlan:
        symbols = ("AAAUSDT", "BUSDT")

    def fake_build_plan(start, end, timeframe, execution_universe_size):
        call_order.append("build_plan")
        return FakePlan()

    def fake_collect(plan, execute, workers):
        call_order.append("collect_roster")
        return {"mode": "completed"}

    monkeypatch.setattr(
        "src.application.data.mhs_execution_collection.build_mhs_execution_plan", fake_build_plan
    )
    monkeypatch.setattr(
        "src.application.data.mhs_execution_collection.collect_mhs_execution_data", fake_collect
    )

    _refresh_live_universe(args)

    assert "ohlcv:AAAUSDT" in call_order
    assert "funding:AAAUSDT" in call_order
    assert call_order.index("ohlcv:AAAUSDT") < call_order.index("build_plan")
    assert "collect_roster" in call_order


def test_data_refresh_live_universe_one_symbol_failure_does_not_abort(monkeypatch) -> None:
    """A single symbol's network failure is logged and skipped, not fatal."""
    import glob as glob_mod

    parser = _mhs_parser()
    args = parser.parse_args(["data", "refresh-live-universe"])

    monkeypatch.setattr(glob_mod, "glob", lambda pattern: ["AAAUSDT.parquet", "BUSDT.parquet"])

    class FlakyCollector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            if symbol == "AAAUSDT":
                raise OSError("network unreachable")

        def ensure_funding_data(self, symbol, start, end):
            pass

    monkeypatch.setattr(
        "src.market_data.services.futures_collection.DataCollector", FlakyCollector
    )

    class FakePlan:
        symbols = ()

    monkeypatch.setattr(
        "src.application.data.mhs_execution_collection.build_mhs_execution_plan",
        lambda start, end, timeframe, execution_universe_size: FakePlan(),
    )
    monkeypatch.setattr(
        "src.application.data.mhs_execution_collection.collect_mhs_execution_data",
        lambda plan, execute, workers: {"mode": "completed"},
    )

    _refresh_live_universe(args)  # must not raise despite AAAUSDT's failure


def test_stream_liquidations_subcommand_wires_asyncio_run(monkeypatch) -> None:
    """``data collect stream-liquidations`` parses --symbols/--flush-interval-s
    and drives run_liquidation_stream via asyncio.run with a shutdown flag."""
    from src.cli.commands.data import _stream_liquidations

    parser = _mhs_parser()
    args = parser.parse_args(
        ["data", "collect", "stream-liquidations", "--symbols", "BTCUSDT,ETHUSDT", "--flush-interval-s", "30"]
    )
    assert args.handler is _stream_liquidations
    assert args.flush_interval_s == 30.0

    captured: dict = {}

    async def _fake_stream(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "src.market_data.streams.liquidations.run_liquidation_stream", _fake_stream
    )
    monkeypatch.setattr("src.live.lifecycle.install_shutdown_handlers", lambda *a, **k: None)

    _stream_liquidations(args)

    assert captured["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert captured["flush_interval_s"] == 30.0
    assert hasattr(captured["shutdown"], "requested")


def test_refresh_live_universe_metrics_tail_is_failsoft(monkeypatch) -> None:
    """A raising ensure_metrics_live_tail for one roster symbol is logged and
    skipped; _refresh_live_universe completes and calls it for every symbol."""
    import glob as glob_mod

    parser = _mhs_parser()
    args = parser.parse_args(["data", "refresh-live-universe"])

    monkeypatch.setattr(glob_mod, "glob", lambda pattern: ["AAAUSDT.parquet"])

    tail_calls: list[str] = []

    class FakeCollector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            pass

        def ensure_funding_data(self, symbol, start, end):
            pass

        def ensure_metrics_live_tail(self, symbol, *, lookback_days=7):
            tail_calls.append(symbol)
            if symbol == "R0USDT":
                raise ConnectionError("metrics endpoint down")

    monkeypatch.setattr(
        "src.market_data.services.futures_collection.DataCollector", FakeCollector
    )

    class FakePlan:
        symbols = ("R0USDT", "R1USDT", "R2USDT")

    monkeypatch.setattr(
        "src.application.data.mhs_execution_collection.build_mhs_execution_plan",
        lambda start, end, timeframe, execution_universe_size: FakePlan(),
    )
    monkeypatch.setattr(
        "src.application.data.mhs_execution_collection.collect_mhs_execution_data",
        lambda plan, execute, workers: {"mode": "completed"},
    )

    _refresh_live_universe(args)  # must not raise

    assert tail_calls == ["R0USDT", "R1USDT", "R2USDT"]


#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_SIGNAL_10_CLI_SUBCOMMANDS_AND_EXIT_CODES",  # data 측 dispatch 부분
)
