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
    """v2: refresh does 1h/funding + markPriceKlines, no 3m roster."""
    monkeypatch.setenv("LIVE_MIN_UNIVERSE_SYMBOLS", "1")
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

        def ensure_mark_price_data(self, symbol, timeframe, start, end):
            call_order.append(f"mark:{symbol}")

        def ensure_metrics_live_tail(self, symbol, *, lookback_days=7):
            call_order.append(f"metrics:{symbol}")

    monkeypatch.setattr(
        "src.market_data.services.futures_collection.DataCollector", FakeCollector
    )

    # ensure 3m not called
    monkeypatch.setattr(
        "src.application.data.mhs_execution_collection.build_mhs_execution_plan",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("3m build_plan should not be called")),
    )
    monkeypatch.setattr(
        "src.application.data.mhs_execution_collection.collect_mhs_execution_data",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("3m collect should not be called")),
    )

    _refresh_live_universe(args)

    assert "ohlcv:AAAUSDT" in call_order
    assert "funding:AAAUSDT" in call_order
    assert "mark:AAAUSDT" in call_order


def test_data_refresh_live_universe_one_symbol_failure_does_not_abort(monkeypatch) -> None:
    """A single symbol's network failure is logged and skipped, not fatal."""
    monkeypatch.setenv("LIVE_MIN_UNIVERSE_SYMBOLS", "1")
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

        def ensure_mark_price_data(self, symbol, timeframe, start, end):
            pass

        def ensure_metrics_live_tail(self, symbol, *, lookback_days=7):
            pass

    monkeypatch.setattr(
        "src.market_data.services.futures_collection.DataCollector", FlakyCollector
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
    """Metrics tail no longer exists; ensure_metrics_live_tail is never called."""
    monkeypatch.setenv("LIVE_MIN_UNIVERSE_SYMBOLS", "1")
    import glob as glob_mod

    parser = _mhs_parser()
    args = parser.parse_args(["data", "refresh-live-universe"])

    monkeypatch.setattr(glob_mod, "glob", lambda pattern: ["R0USDT.parquet", "R1USDT.parquet", "R2USDT.parquet"])

    tail_calls: list[str] = []

    class FakeCollector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            pass

        def ensure_funding_data(self, symbol, start, end):
            pass

        def ensure_mark_price_data(self, symbol, timeframe, start, end):
            pass

        def ensure_metrics_live_tail(self, symbol, *, lookback_days=7):
            tail_calls.append(symbol)
            if symbol == "R0USDT":
                raise ConnectionError("metrics endpoint down")

    monkeypatch.setattr(
        "src.market_data.services.futures_collection.DataCollector", FakeCollector
    )

    _refresh_live_universe(args)  # must not raise

    assert tail_calls == []


#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_SIGNAL_10_CLI_SUBCOMMANDS_AND_EXIT_CODES",  # data 측 dispatch 부분
)


# --- auto appended from contract ---
def test_refresh_live_universe_cold_box_fails_loud(tmp_path, monkeypatch) -> None:
    import argparse
    import pytest
    import src.cli.commands.data as data_mod
    from src.common import config as cfg

    monkeypatch.setattr(cfg, "FUTURES_DATA_DIR", tmp_path, raising=False)
    monkeypatch.setattr(data_mod, "FUTURES_DATA_DIR", tmp_path, raising=False)
    (tmp_path / "ohlcv" / "1h").mkdir(parents=True)
    (tmp_path / "ohlcv" / "1h" / "BTCUSDT.parquet").touch()

    def _no_collector(*_a, **_k):
        raise AssertionError("collector must not run on a cold box")

    monkeypatch.setattr(data_mod, "DataCollector", _no_collector, raising=False)

    with pytest.raises(SystemExit) as ei:
        data_mod._refresh_live_universe(argparse.Namespace())

    assert ei.value.code == 3


def test_refresh_live_universe_filters_dev_partition_and_no_metrics(tmp_path, monkeypatch) -> None:
    import argparse
    import src.cli.commands.data as data_mod
    from src.common import config as cfg
    from src.research.universe.pit_universe import symbol_partition

    root = tmp_path
    (root / "ohlcv" / "1h").mkdir(parents=True)
    # build a mix until we have >= 100 dev symbols and some holdout
    made_dev, made_holdout = [], []
    i = 0
    while len(made_dev) < 120 or len(made_holdout) < 5:
        s = f"SYM{i}USDT"
        (root / "ohlcv" / "1h" / f"{s}.parquet").touch()
        (made_dev if symbol_partition(s) == "dev" else made_holdout).append(s)
        i += 1
    monkeypatch.setattr(cfg, "FUTURES_DATA_DIR", root, raising=False)
    monkeypatch.setattr(data_mod, "FUTURES_DATA_DIR", root, raising=False)

    refreshed: list[str] = []
    metrics_calls: list[str] = []

    class _Collector:
        def ensure_metrics_live_tail(self, sym, **k):
            metrics_calls.append(sym)

    monkeypatch.setattr(data_mod, "DataCollector", lambda *a, **k: _Collector(), raising=False)
    monkeypatch.setattr(
        data_mod, "_refresh_one_symbol_tail",
        lambda collector, sym, start, end: refreshed.append(sym) or True,
        raising=False,
    )

    data_mod._refresh_live_universe(argparse.Namespace())

    assert set(refreshed) == set(made_dev)
    assert not any(s in refreshed for s in made_holdout)
    assert metrics_calls == []


def test_seed_cloud_fetches_dev_usdt_universe_idempotent(monkeypatch) -> None:
    import argparse
    import src.cli.commands.data as data_mod
    from src.research.universe.pit_universe import symbol_partition

    listed = ["BTCUSDT", "ETHUSDT", "AAAUSDT", "BBBUSDT", "CCCUSDT", "SOMECOIN", "XRPUSDT", "BNBBUSD"]
    expected = [s for s in listed if s.endswith("USDT") and symbol_partition(s) == "dev"]

    class _Vision:
        def list_all_symbols(self, **k):
            return listed

    monkeypatch.setattr("src.market_data.binance.vision.BinanceVisionDownloader", lambda *a, **k: _Vision(), raising=False)
    monkeypatch.setattr(data_mod, "DataCollector", lambda *a, **k: object(), raising=False)
    seen: list[str] = []
    monkeypatch.setattr(
        data_mod, "_refresh_one_symbol_tail",
        lambda collector, sym, start, end: seen.append(sym) or True,
        raising=False,
    )

    data_mod._seed_cloud(argparse.Namespace(lookback_days=30))

    assert sorted(seen) == sorted(expected)
    assert expected  # non-empty guard sanity


def test_prune_live_data_cli_dispatches_both_prunes(monkeypatch) -> None:
    import argparse
    import src.cli.commands.data as data_mod

    calls: list[str] = []
    monkeypatch.setattr("src.market_data.retention.prune_market_data", lambda *a, **k: calls.append("market") or {}, raising=False)
    monkeypatch.setattr("src.market_data.retention.prune_orderbook_history", lambda *a, **k: calls.append("orderbook") or 0, raising=False)

    data_mod._prune_live_data(argparse.Namespace())

    assert calls == ["market", "orderbook"]


def test_seed_cloud_and_prune_live_data_subcommands_registered() -> None:
    import argparse
    import src.cli.commands.data as data_mod

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    data_mod.add_data_commands(sub.add_parser("data"))

    a = parser.parse_args(["data", "seed-cloud", "--lookback-days", "200"])
    assert a.handler is data_mod._seed_cloud
    assert a.lookback_days == 200

    b = parser.parse_args(["data", "prune-live-data"])
    assert b.handler is data_mod._prune_live_data


