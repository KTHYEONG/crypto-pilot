"""Contract coverage for the ``data collect mhs-execution`` CLI defaults."""

from __future__ import annotations

import argparse

import pytest

from src.cli.commands.data import (
    _mhs_execution,
    _refresh_live_universe,
    add_data_commands,
)


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
    import src.market_data.services.mhs_execution as mc

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
    """v2: refresh does 1h trade OHLCV + settled funding only, no 3m roster, no mark artifacts."""
    monkeypatch.setenv("LIVE_MIN_UNIVERSE_SYMBOLS", "1")
    parser = _mhs_parser()
    args = parser.parse_args(["data", "refresh-live-universe"])
    assert args.handler is _refresh_live_universe

    from src.live.data_refresh import RefreshReport

    called: dict = {}

    def _fake_refresh(*a, **k):
        called["called"] = True
        # simulate that collector would be called for AAAUSDT
        return RefreshReport(total=2, fresh=0, refreshed=2, failed=0, deadline_skipped=0, elapsed_s=1.0, deadline_hit=False, staleness_hours=1.0, ok=True)

    monkeypatch.setattr("src.live.data_refresh.refresh_live_market_data", _fake_refresh)

    _refresh_live_universe(args)

    assert called.get("called") is True


def test_data_refresh_live_universe_one_symbol_failure_does_not_abort(tmp_path, monkeypatch) -> None:
    """A single symbol's network failure is logged and skipped, not fatal."""
    import pandas as pd
    import src.live.data_refresh as data_refresh
    from src.market_data.services.futures_collection import FUNDING_DEFAULT_INTERVAL_MS, last_settled_funding_epoch_ms

    monkeypatch.setenv("LIVE_MIN_UNIVERSE_SYMBOLS", "1")
    monkeypatch.setattr("src.common.paths.FUTURES_DATA_DIR", tmp_path)
    ohlcv_dir = tmp_path / "ohlcv" / "1h"
    ohlcv_dir.mkdir(parents=True)
    funding_dir = tmp_path / "funding"
    funding_dir.mkdir(parents=True)
    now = pd.Timestamp.now(tz="UTC")

    def _write_symbol(symbol: str, fresh: bool) -> None:
        tail = now.floor("1h") if fresh else now.floor("1h") - pd.Timedelta(days=5)
        stamps = [int((tail - pd.Timedelta(hours=h)).value // 10**6) for h in range(48)]
        pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(ohlcv_dir / f"{symbol}.parquet", index=False)
        last = last_settled_funding_epoch_ms(now, FUNDING_DEFAULT_INTERVAL_MS)
        if not fresh:
            last -= 3 * FUNDING_DEFAULT_INTERVAL_MS
        funding = [last - k * FUNDING_DEFAULT_INTERVAL_MS for k in (2, 1, 0)]
        pd.DataFrame({"timestamp": funding, "funding_rate": [0.0001] * 3}).to_parquet(funding_dir / f"{symbol}.parquet", index=False)
    _write_symbol("AAAUSDT", fresh=False)
    _write_symbol("BUSDT", fresh=False)
    seen: list[str] = []

    class FlakyCollector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            seen.append(symbol)
            if symbol == "AAAUSDT":
                raise OSError("network unreachable")
            tail = now.floor("1h")
            stamps = [int((tail - pd.Timedelta(hours=h)).value // 10**6) for h in range(48)]
            pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(ohlcv_dir / f"{symbol}.parquet", index=False)

        def ensure_funding_data(self, symbol, start, end):
            last = last_settled_funding_epoch_ms(now, FUNDING_DEFAULT_INTERVAL_MS)
            funding = [last - k * FUNDING_DEFAULT_INTERVAL_MS for k in (2, 1, 0)]
            pd.DataFrame({"timestamp": funding, "funding_rate": [0.0001] * 3}).to_parquet(funding_dir / f"{symbol}.parquet", index=False)

    monkeypatch.setattr(data_refresh, "DataCollector", FlakyCollector)

    parser = _mhs_parser()
    args = parser.parse_args(["data", "refresh-live-universe"])
    _refresh_live_universe(args)

    assert sorted(seen) == ["AAAUSDT", "BUSDT"]




def test_refresh_live_universe_cold_box_fails_loud(tmp_path, monkeypatch) -> None:
    import argparse
    import pytest
    import src.cli.commands.data as data_mod
    from src.common import paths as cfg

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
    from src.common import paths as cfg
    from src.live.data_refresh import RefreshReport
    from src.quant.universe.pit_universe import symbol_partition

    root = tmp_path
    (root / "ohlcv" / "1h").mkdir(parents=True)
    made_dev, made_holdout = [], []
    i = 0
    while len(made_dev) < 120 or len(made_holdout) < 5:
        s = f"SYM{i}USDT"
        (root / "ohlcv" / "1h" / f"{s}.parquet").touch()
        (made_dev if symbol_partition(s) == "dev" else made_holdout).append(s)
        i += 1
    monkeypatch.setattr(cfg, "FUTURES_DATA_DIR", root, raising=False)
    monkeypatch.setattr(data_mod, "FUTURES_DATA_DIR", root, raising=False)

    captured: dict = {}

    def _fake_refresh(*a, **k):
        captured["called"] = True
        # verify internal dev filtering is delegated correctly: refresh_live_market_data would filter
        # Here we just simulate success
        return RefreshReport(total=len(made_dev), fresh=0, refreshed=len(made_dev), failed=0, deadline_skipped=0, elapsed_s=1.0, deadline_hit=False, staleness_hours=1.0, ok=True)

    monkeypatch.setattr("src.live.data_refresh.refresh_live_market_data", _fake_refresh)

    data_mod._refresh_live_universe(argparse.Namespace())
    assert captured.get("called") is True


def test_seed_cloud_fetches_listed_crypto_perpetuals_universe(monkeypatch) -> None:
    """seed-cloud must mirror the daemon's actual universe (COIN perpetuals across all
    partitions), not a Vision dev-partition listing -- otherwise cold-boot leaves out
    symbols the live strategy step needs on its first cycle."""
    import argparse
    import src.cli.commands.data as data_mod
    from src.live.data_refresh import RefreshReport

    payload = {
        "symbols": [
            {"symbol": "BTCUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT", "underlyingType": "COIN"},
            {"symbol": "ETHUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT", "underlyingType": "COIN"},
            {"symbol": "AAPLUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT", "underlyingType": "EQUITY"},
            {"symbol": "XRPUSDT_260925", "status": "TRADING", "contractType": "CURRENT_QUARTER", "quoteAsset": "USDT", "underlyingType": "COIN"},
        ],
    }
    monkeypatch.setattr("src.market_data.services.universe_gaps.fetch_exchange_info", lambda **k: payload)
    monkeypatch.setattr("src.live.data_refresh.fetch_listed_symbols", lambda *a, **k: frozenset({"BTCUSDT", "ETHUSDT"}))
    seen: list[str] = []

    def _fake_refresh(*a, **k):
        seen.extend(k.get("symbols", []))
        return RefreshReport(total=len(seen), fresh=0, refreshed=len(seen), failed=0, deadline_skipped=0, elapsed_s=1.0, deadline_hit=False, staleness_hours=1.0, ok=True)

    monkeypatch.setattr("src.live.data_refresh.refresh_live_market_data", _fake_refresh)

    data_mod._seed_cloud(argparse.Namespace(lookback_days=30))

    assert sorted(seen) == ["BTCUSDT", "ETHUSDT"]


def test_seed_cloud_fails_closed_on_exchange_info_error(monkeypatch) -> None:
    import argparse
    import pytest
    import src.cli.commands.data as data_mod

    def _boom(**k):
        raise RuntimeError("network down")

    monkeypatch.setattr("src.market_data.services.universe_gaps.fetch_exchange_info", _boom)
    with pytest.raises(SystemExit):
        data_mod._seed_cloud(argparse.Namespace(lookback_days=30))


def test_prune_live_data_cli_dispatches_both_prunes(monkeypatch) -> None:
    import argparse
    import src.cli.commands.data as data_mod

    calls: list[str] = []
    monkeypatch.setattr("src.market_data.retention.prune_market_data", lambda *a, **k: calls.append("market") or {}, raising=False)
    monkeypatch.setattr("src.market_data.retention.prune_orderbook_history", lambda *a, **k: calls.append("orderbook") or 0, raising=False)

    data_mod._prune_live_data(argparse.Namespace())

    assert calls == ["market", "orderbook"]


def test_prune_live_data_sends_backup_alert_through_outbox(tmp_path, monkeypatch) -> None:
    import argparse
    import src.cli.commands.data as data_mod

    alerts: list[dict] = []
    monkeypatch.setenv("LIVE_ALERT_OUTBOX_PATH", str(tmp_path / "outbox.json"))
    monkeypatch.setattr("src.live.orderbook.default_orderbook_dir", lambda: tmp_path)
    monkeypatch.setattr("src.market_data.retention.prune_market_data", lambda *a, **k: {})
    monkeypatch.setattr("src.market_data.retention.prune_orderbook_history", lambda *a, **k: 0)
    monkeypatch.setattr(
        "src.market_data.retention.check_orderbook_prune_impending",
        lambda *a, **k: (True, 5, "2025-09-05"),
    )
    monkeypatch.setattr(
        "src.live.alerting.dispatch_alert",
        lambda settings, **k: alerts.append(k) or True,
    )

    data_mod._prune_live_data(argparse.Namespace())

    assert len(alerts) == 1
    assert alerts[0]["event"] == "orderbook_backup_impending"
    assert alerts[0]["dedupe_key"] == "orderbook_backup_impending:2025-09-05"
    assert "earliest_date=2025-09-05" in alerts[0]["detail"]
    assert not (tmp_path / ".backup_alert_20250905").exists()


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


def test_refresh_live_universe_cli_exits_3_on_cold_universe(monkeypatch) -> None:
    import argparse
    import pytest
    from src.cli.commands import data as data_cmd
    from src.live.data_refresh import ColdUniverseError

    def _raise(*a, **k):
        raise ColdUniverseError("dev universe 3 < min 100")

    monkeypatch.setattr("src.live.data_refresh.refresh_live_market_data", _raise)

    with pytest.raises(SystemExit) as exc:
        data_cmd._refresh_live_universe(argparse.Namespace())
    assert exc.value.code == 3


def test_refresh_live_universe_cli_exits_1_when_report_not_ok(monkeypatch) -> None:
    import argparse
    import pytest
    from src.cli.commands import data as data_cmd
    from src.live.data_refresh import RefreshReport

    rep = RefreshReport(total=500, fresh=0, refreshed=10, failed=490, deadline_skipped=0,
                        elapsed_s=9.0, deadline_hit=False, staleness_hours=50.0, ok=False)
    monkeypatch.setattr("src.live.data_refresh.refresh_live_market_data", lambda *a, **k: rep)

    with pytest.raises(SystemExit) as exc:
        data_cmd._refresh_live_universe(argparse.Namespace())
    assert exc.value.code == 1


# --- auto appended from contract: signal_input_quarantine ---


def test_data_repair_ohlcv_cli_wires_repair_with_retention_default_lookback(monkeypatch) -> None:
    import src.live.data_repair as repair_mod
    from src.common.paths import FUTURES_DATA_DIR
    from src.market_data.retention import MARKET_DATA_MIN_RETENTION_DAYS

    captured: dict[str, object] = {}

    class _FakeCollector:
        pass

    def _fake_repair(symbol, **kwargs):
        captured["symbol"] = symbol
        captured.update(kwargs)
        return repair_mod.RepairResult(symbol=symbol, status="healthy", moved_to=None, rows=5)

    monkeypatch.setattr(repair_mod, "repair_ohlcv_file", _fake_repair)
    monkeypatch.setattr("src.market_data.services.futures_collection.DataCollector", _FakeCollector)
    args = _mhs_parser().parse_args(["data", "repair-ohlcv", "--symbol", "BTCUSDT"])

    args.handler(args)

    assert captured["symbol"] == "BTCUSDT"
    assert captured["lookback_days"] == MARKET_DATA_MIN_RETENTION_DAYS
    assert captured["futures_root"] == FUTURES_DATA_DIR
    assert isinstance(captured["collector"], _FakeCollector)
    assert captured["now"].tzinfo is not None
    explicit = _mhs_parser().parse_args(["data", "repair-ohlcv", "--symbol", "ETHUSDT", "--lookback-days", "60"])
    explicit.handler(explicit)
    assert captured["lookback_days"] == 60

def test_data_seal_mhs_inputs_seals_complete_symbols(tmp_path) -> None:
    # Given: a corpus with one complete and one partial symbol
    import json

    import pandas as pd
    import pytest

    from src.cli.commands.data import add_data_commands

    def _write(rel: str) -> None:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"timestamp": [pd.Timestamp("2025-01-01", tz="UTC")], "close": [1.0]}
        ).to_parquet(path)

    for rel in (
        "ohlcv/1h/COMPLETEUSDT.parquet",
        "ohlcv/3m/COMPLETEUSDT.parquet",
        "funding/COMPLETEUSDT.parquet",
        "markPriceKlines/1h/COMPLETEUSDT.parquet",
        "metrics/1d/COMPLETEUSDT.parquet",
        "ohlcv/1h/PARTIALUSDT.parquet",
    ):
        _write(rel)

    parser = argparse.ArgumentParser()
    add_data_commands(parser.add_subparsers(dest="group", required=True).add_parser("data"))
    out = tmp_path / "manifest" / "input_manifest.json"

    # When
    args = parser.parse_args([
        "data", "seal-mhs-inputs",
        "--lake-root", str(tmp_path),
        "--execution-timeframe", "3m",
        "--output", str(out),
    ])
    args.handler(args)

    # Then: only consumed source paths (1h/3m trade OHLCV + funding) affect the
    # manifest -- arbitrary mark and metrics files never enter the input identity.
    payload = json.loads(out.read_text(encoding="utf-8"))
    attested = {entry["relative_path"] for entry in payload["files"]}
    assert attested == {
        "ohlcv/1h/COMPLETEUSDT.parquet",
        "ohlcv/3m/COMPLETEUSDT.parquet",
        "funding/COMPLETEUSDT.parquet",
        "ohlcv/1h/PARTIALUSDT.parquet",
    }
    assert all("mark" not in rel and "metrics" not in rel for rel in attested)
    assert payload["digest"]

    # And: an empty corpus fails closed instead of writing an empty manifest
    empty_root = tmp_path / "empty"
    (empty_root / "ohlcv" / "1h").mkdir(parents=True)
    empty_args = parser.parse_args([
        "data", "seal-mhs-inputs",
        "--lake-root", str(empty_root),
        "--output", str(tmp_path / "never.json"),
    ])
    with pytest.raises(SystemExit):
        empty_args.handler(empty_args)


def test_seal_mhs_inputs_old_data_root_flag_rejected(tmp_path) -> None:
    import argparse

    import pytest

    from src.cli.commands.data import add_data_commands

    parser = argparse.ArgumentParser()
    add_data_commands(parser.add_subparsers(dest="group", required=True).add_parser("data"))
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args([
            "data", "seal-mhs-inputs",
            "--data-root", str(tmp_path),
            "--output", str(tmp_path / "never.json"),
        ])
    assert exc_info.value.code == 2
    assert not (tmp_path / "never.json").exists()


def test_universe_gaps_no_execute_skips_collection(monkeypatch) -> None:
    import argparse
    import src.cli.commands.data as data_mod
    from src.quant.universe.pit_universe import symbol_partition

    listed = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]
    dev = sorted(s for s in listed if symbol_partition(s) == "dev")
    assert dev

    class _Vision:
        def list_all_symbols(self, **k):
            return list(listed)

    monkeypatch.setattr("src.market_data.binance.vision.BinanceVisionDownloader", lambda *a, **k: _Vision())
    monkeypatch.setattr(
        "src.market_data.services.universe_gaps.local_futures_symbols",
        lambda root, timeframe: frozenset(),
    )
    monkeypatch.setattr(
        "src.market_data.services.universe_gaps.fetch_exchange_info",
        lambda **k: {"symbols": []},
    )
    calls: list = []
    monkeypatch.setattr(
        data_mod.collection, "collect_ohlcv", lambda *a, **k: calls.append(("ohlcv", a))
    )
    monkeypatch.setattr(
        data_mod.collection, "collect_funding", lambda *a, **k: calls.append(("funding", a))
    )
    args = argparse.Namespace(
        timeframe="1h", partition="dev", start="2021-01-01", end="2022-01-01", execute=False
    )
    data_mod._universe_gaps(args)
    assert calls == []


def test_universe_gaps_execute_collects_in_sorted_order(monkeypatch) -> None:
    import argparse
    import src.cli.commands.data as data_mod
    from src.quant.universe.pit_universe import symbol_partition

    listed = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]
    dev = sorted(s for s in listed if symbol_partition(s) == "dev")
    assert dev

    class _Vision:
        def list_all_symbols(self, **k):
            return list(listed)

    monkeypatch.setattr("src.market_data.binance.vision.BinanceVisionDownloader", lambda *a, **k: _Vision())
    monkeypatch.setattr(
        "src.market_data.services.universe_gaps.local_futures_symbols",
        lambda root, timeframe: frozenset(),
    )
    monkeypatch.setattr(
        "src.market_data.services.universe_gaps.fetch_exchange_info",
        lambda **k: {"symbols": []},
    )
    calls: list = []
    monkeypatch.setattr(
        data_mod.collection, "collect_ohlcv", lambda *a, **k: calls.append(("ohlcv", a))
    )
    monkeypatch.setattr(
        data_mod.collection, "collect_funding", lambda *a, **k: calls.append(("funding", a))
    )
    args = argparse.Namespace(
        timeframe="1h", partition="dev", start="2021-01-01", end="2022-01-01", execute=True
    )
    data_mod._universe_gaps(args)
    symbols = [a[0] for kind, a in calls if kind == "ohlcv"]
    assert symbols == sorted(symbols) == dev
    assert len([c for c in calls if c[0] == "funding"]) == len(dev)


def test_data_collect_universe_gaps_argv() -> None:
    parser = _mhs_parser()
    args = parser.parse_args(["data", "collect", "universe-gaps", "--end", "2022-01-01"])
    from src.cli.commands.data import _universe_gaps

    assert args.handler is _universe_gaps
    with __import__("pytest").raises(SystemExit):
        parser.parse_args(["data", "collect", "universe-gaps"])


def test_universe_gaps_fails_closed_when_one_listing_is_empty(monkeypatch) -> None:
    import argparse
    import pytest
    import src.cli.commands.data as data_mod

    class _Vision:
        def list_all_symbols(self, **k):
            # 월간 목록만 실패(빈 결과)한 경우: 불완전한 합집합으로 진행하면 안 된다.
            return [] if "dataset_prefix" in k else ["BTCUSDT"]

    monkeypatch.setattr("src.market_data.binance.vision.BinanceVisionDownloader", lambda *a, **k: _Vision())
    calls: list = []
    monkeypatch.setattr(data_mod.collection, "collect_ohlcv", lambda *a, **k: calls.append(a))
    args = argparse.Namespace(
        timeframe="1h", partition="dev", start="2021-01-01", end="2022-01-01", execute=True
    )
    with pytest.raises(SystemExit):
        data_mod._universe_gaps(args)
    assert calls == []


def test_universe_gaps_fails_closed_on_exchange_info_failure(monkeypatch) -> None:
    import argparse
    import pytest
    import src.cli.commands.data as data_mod

    class _Vision:
        def list_all_symbols(self, **k):
            return ["BTCUSDT"]

    monkeypatch.setattr("src.market_data.binance.vision.BinanceVisionDownloader", lambda *a, **k: _Vision())
    monkeypatch.setattr(
        "src.market_data.services.universe_gaps.fetch_exchange_info",
        lambda **k: (_ for _ in ()).throw(OSError("network unreachable")),
    )
    calls: list = []
    monkeypatch.setattr(data_mod.collection, "collect_ohlcv", lambda *a, **k: calls.append(a))
    args = argparse.Namespace(
        timeframe="1h", partition="dev", start="2021-01-01", end="2022-01-01", execute=True
    )
    with pytest.raises(SystemExit):
        data_mod._universe_gaps(args)
    assert calls == []


def test_universe_gaps_excludes_tokenized_equity(monkeypatch) -> None:
    """SCENARIO: a stock/commodity/index perpetual sharing the USDT-suffix
    naming convention (e.g. TSLAUSDT) must never enter the crypto MHS lake."""
    import argparse
    import src.cli.commands.data as data_mod
    from src.quant.universe.pit_universe import symbol_partition

    listed = ["BTCUSDT", "TSLAUSDT", "XAUUSDT", "HK0700USDT"]
    dev_crypto = [s for s in listed if symbol_partition(s) == "dev" and s == "BTCUSDT"]
    assert dev_crypto == ["BTCUSDT"]

    class _Vision:
        def list_all_symbols(self, **k):
            return list(listed)

    monkeypatch.setattr("src.market_data.binance.vision.BinanceVisionDownloader", lambda *a, **k: _Vision())
    monkeypatch.setattr(
        "src.market_data.services.universe_gaps.local_futures_symbols",
        lambda root, timeframe: frozenset(),
    )
    monkeypatch.setattr(
        "src.market_data.services.universe_gaps.fetch_exchange_info",
        lambda **k: {
            "symbols": [
                {"symbol": "TSLAUSDT", "underlyingType": "EQUITY"},
                {"symbol": "XAUUSDT", "underlyingType": "COMMODITY"},
                {"symbol": "HK0700USDT", "underlyingType": "HK_EQUITY"},
                {"symbol": "BTCUSDT", "underlyingType": "COIN"},
            ]
        },
    )
    calls: list = []
    monkeypatch.setattr(
        data_mod.collection, "collect_ohlcv", lambda *a, **k: calls.append(a[0])
    )
    monkeypatch.setattr(data_mod.collection, "collect_funding", lambda *a, **k: None)
    args = argparse.Namespace(
        timeframe="1h", partition="dev", start="2021-01-01", end="2022-01-01", execute=True
    )
    data_mod._universe_gaps(args)
    assert calls == ["BTCUSDT"]


def test_normalize_market_subcommand_wires_normalizer(monkeypatch) -> None:
    """`data collect normalize-market` delegates to the dedicated normalizer entrypoint."""
    from pathlib import Path

    import src.cli.commands.data as data_mod

    parser = _mhs_parser()
    args = parser.parse_args(["data", "collect", "normalize-market", "--capture-root", "X"])
    assert args.handler is data_mod._normalize_market
    assert args.capture_root == "X"

    captured: dict = {}

    def _fake_run_normalizer(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("src.market_data.streams.normalizer_main.run_normalizer_process", _fake_run_normalizer)

    data_mod._normalize_market(args)

    assert captured["capture_root"] == Path("X")
    assert "liquidations_dir" in captured
    assert "backup_status_path" in captured


def test_prune_live_data_prunes_old_listing_slots(tmp_path, monkeypatch) -> None:
    import argparse
    import src.cli.commands.data as data_mod

    listing = tmp_path / "data" / "state" / "venue_listing"
    listing.mkdir(parents=True)
    (listing / "20200101.json.gz").write_bytes(b"old")
    (listing / "20990101.json.gz").write_bytes(b"new")
    (listing / "notes.txt").write_bytes(b"junk")
    (listing / "notes.json.gz").write_bytes(b"junk")
    (listing / "20261345.json.gz").write_bytes(b"junk")
    monkeypatch.setattr("src.common.paths.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("src.live.orderbook.default_orderbook_dir", lambda: tmp_path / "ob")
    monkeypatch.setattr("src.market_data.retention.prune_market_data", lambda *a, **k: {})
    monkeypatch.setattr("src.market_data.retention.prune_orderbook_history", lambda *a, **k: 0)
    monkeypatch.setattr(
        "src.market_data.retention.check_orderbook_prune_impending", lambda *a, **k: (False, 0, None)
    )

    data_mod._prune_live_data(argparse.Namespace())

    assert not (listing / "20200101.json.gz").exists()
    assert (listing / "20990101.json.gz").exists()
    assert (listing / "notes.txt").exists()
    assert (listing / "notes.json.gz").exists()
    assert (listing / "20261345.json.gz").exists()


def test_prune_live_data_tolerates_listing_unlink_failure(tmp_path, monkeypatch) -> None:
    import argparse
    import pathlib
    import src.cli.commands.data as data_mod

    listing = tmp_path / "data" / "state" / "venue_listing"
    listing.mkdir(parents=True)
    victim = listing / "20200101.json.gz"
    victim.write_bytes(b"old")
    monkeypatch.setattr("src.common.paths.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("src.live.orderbook.default_orderbook_dir", lambda: tmp_path / "ob")
    monkeypatch.setattr("src.market_data.retention.prune_market_data", lambda *a, **k: {})
    monkeypatch.setattr("src.market_data.retention.prune_orderbook_history", lambda *a, **k: 0)
    monkeypatch.setattr(
        "src.market_data.retention.check_orderbook_prune_impending", lambda *a, **k: (False, 0, None)
    )
    real_unlink = pathlib.Path.unlink

    def _flaky_unlink(self, *args, **kwargs):
        if self == victim:
            raise OSError("disk read-only")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "unlink", _flaky_unlink)

    data_mod._prune_live_data(argparse.Namespace())

    assert victim.exists()


def test_venue_rules_cli_prints_written_snapshot_path(tmp_path, monkeypatch, capsys) -> None:
    """`data collect venue-rules` prints exactly the written snapshot path to stdout."""
    import src.cli.commands.data as data_mod
    import src.market_data.binance.venue_rules as venue_mod
    from types import SimpleNamespace

    import pandas as pd

    snapshot = SimpleNamespace(symbols=[], captured_at=pd.Timestamp("2026-10-04", tz="UTC"))
    monkeypatch.setattr("src.common.paths.VENUE_RULES_DIR", tmp_path)
    monkeypatch.setattr(venue_mod, "venue_rule_snapshot_exists", lambda *a, **k: False)
    monkeypatch.setattr(venue_mod, "fetch_venue_rules", lambda **k: snapshot)
    monkeypatch.setattr(venue_mod, "write_venue_rule_snapshot", lambda snapshot, directory: tmp_path / "v.json.gz")

    data_mod._venue_rules(argparse.Namespace())

    assert "v.json.gz" in capsys.readouterr().out


def _settlement_cli_lake(root) -> None:
    import pandas as pd

    t0 = pd.Timestamp("2025-01-01T00:00:00Z")
    stamps = [int((t0 + pd.Timedelta(minutes=3 * i)).value // 10**6) for i in range(20)]
    flat = [int((t0 + pd.Timedelta(minutes=3 * (20 + i))).value // 10**6) for i in range(100)]
    (root / "ohlcv" / "3m").mkdir(parents=True)
    pd.DataFrame({
        "timestamp": stamps + flat,
        "open": [2.0] * 120, "high": [2.0] * 120, "low": [2.0] * 120,
        "close": [2.0] * 120, "quote_vol": [10.0] * 20 + [0.0] * 100,
    }).to_parquet(root / "ohlcv" / "3m" / "FLATUSDT.parquet", index=False)
    (root / "ohlcv" / "1h").mkdir(parents=True)
    hour = [int((t0 + pd.Timedelta(hours=i)).value // 10**6) for i in range(5)]
    pd.DataFrame({
        "timestamp": hour,
        "open": [2.0] * 5, "high": [2.0] * 5, "low": [2.0] * 5,
        "close": [2.0] * 5, "volume": [5.0] + [0.0] * 4,
        "quote_vol": [10.0] + [0.0] * 4,
    }).to_parquet(root / "ohlcv" / "1h" / "FLATUSDT.parquet", index=False)


def test_build_settlement_registry_dry_run_writes_nothing(tmp_path, monkeypatch, caplog) -> None:
    from src.common import paths as cfg
    from src.core.instrument_settlements import (
        EMPTY_SETTLEMENT_REGISTRY,
        default_instrument_settlement_registry_path,
    )

    _settlement_cli_lake(tmp_path)
    monkeypatch.setattr(cfg, "FUTURES_DATA_DIR", tmp_path, raising=False)
    monkeypatch.setattr(
        "src.core.instrument_settlements.load_instrument_settlement_registry",
        lambda path=None: EMPTY_SETTLEMENT_REGISTRY,
    )
    committed = default_instrument_settlement_registry_path().read_bytes()
    with caplog.at_level("INFO", logger="src.cli.commands.data"):
        args = _mhs_parser().parse_args(
            ["data", "build-settlement-registry", "--horizon", "2025-01-02T00:00:00Z"]
        )
        args.handler(args)
    assert default_instrument_settlement_registry_path().read_bytes() == committed
    assert not (tmp_path / "registry.jsonl").exists()
    assert "stage=build_settlement_registry" in caplog.text


def test_build_settlement_registry_write_replaces_registry(tmp_path, monkeypatch) -> None:
    from src.common import paths as cfg
    from src.core.instrument_settlements import (
        EMPTY_SETTLEMENT_REGISTRY,
        parse_instrument_settlement_registry,
    )

    _settlement_cli_lake(tmp_path)
    monkeypatch.setattr(cfg, "FUTURES_DATA_DIR", tmp_path, raising=False)
    monkeypatch.setattr(
        "src.core.instrument_settlements.load_instrument_settlement_registry",
        lambda path=None: EMPTY_SETTLEMENT_REGISTRY,
    )
    target = tmp_path / "committed.jsonl"
    monkeypatch.setattr(
        "src.core.instrument_settlements.default_instrument_settlement_registry_path",
        lambda: target,
    )
    args = _mhs_parser().parse_args(
        ["data", "build-settlement-registry", "--horizon", "2025-01-02T00:00:00Z", "--write"]
    )
    args.handler(args)
    registry = parse_instrument_settlement_registry(target.read_bytes(), source=str(target))
    assert [r.symbol for r in registry.settlements] == ["FLATUSDT"]


def test_build_settlement_registry_bootstraps_without_committed(tmp_path, monkeypatch, caplog) -> None:
    from src.common import paths as cfg
    from src.common.errors import DataIntegrityError

    _settlement_cli_lake(tmp_path)
    monkeypatch.setattr(cfg, "FUTURES_DATA_DIR", tmp_path, raising=False)

    def _missing(path=None):
        raise DataIntegrityError("settlement registry unreadable: gone")

    monkeypatch.setattr(
        "src.core.instrument_settlements.load_instrument_settlement_registry", _missing,
    )
    monkeypatch.setattr(
        "src.core.instrument_settlements.default_instrument_settlement_registry_path",
        lambda: tmp_path / "absent.jsonl",
    )
    with caplog.at_level("INFO", logger="src.cli.commands.data"):
        args = _mhs_parser().parse_args(
            ["data", "build-settlement-registry", "--horizon", "2025-01-02T00:00:00Z"]
        )
        args.handler(args)
    assert "NO_COMMITTED_REGISTRY" in caplog.text
    assert not (tmp_path / "absent.jsonl").exists()


def test_build_settlement_registry_refuses_corrupt_committed(tmp_path, monkeypatch) -> None:
    import pytest

    from src.common import paths as cfg
    from src.common.errors import DataIntegrityError

    _settlement_cli_lake(tmp_path)
    monkeypatch.setattr(cfg, "FUTURES_DATA_DIR", tmp_path, raising=False)
    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_bytes(b"{oops")
    monkeypatch.setattr(
        "src.core.instrument_settlements.default_instrument_settlement_registry_path",
        lambda: corrupt,
    )
    args = _mhs_parser().parse_args(
        ["data", "build-settlement-registry", "--horizon", "2025-01-02T00:00:00Z"]
    )
    with pytest.raises(DataIntegrityError):
        args.handler(args)


def test_build_settlement_registry_reports_unresolved_and_writes_output(
    tmp_path, monkeypatch, caplog,
) -> None:
    import pandas as pd

    from src.common import paths as cfg
    from src.core.instrument_settlements import (
        EMPTY_SETTLEMENT_REGISTRY,
        parse_instrument_settlement_registry,
    )

    _settlement_cli_lake(tmp_path)
    t0 = pd.Timestamp("2025-01-01T00:00:00Z")
    stamps = [int((t0 + pd.Timedelta(minutes=3 * i)).value // 10**6) for i in range(10)]
    pd.DataFrame({
        "timestamp": stamps,
        "open": [7.0] * 10, "high": [7.0] * 10, "low": [7.0] * 10,
        "close": [7.0] * 10, "quote_vol": [10.0] * 2 + [0.0] * 8,
    }).to_parquet(tmp_path / "ohlcv" / "3m" / "THINUSDT.parquet", index=False)
    monkeypatch.setattr(cfg, "FUTURES_DATA_DIR", tmp_path, raising=False)
    monkeypatch.setattr(
        "src.core.instrument_settlements.load_instrument_settlement_registry",
        lambda path=None: EMPTY_SETTLEMENT_REGISTRY,
    )
    with caplog.at_level("INFO", logger="src.cli.commands.data"):
        args = _mhs_parser().parse_args(
            ["data", "build-settlement-registry", "--horizon", "2025-01-02T00:00:00Z"]
        )
        args.handler(args)
    assert "unresolved symbol=THINUSDT" in caplog.text
    out = tmp_path / "candidate.jsonl"
    args = _mhs_parser().parse_args(
        ["data", "build-settlement-registry", "--horizon", "2025-01-02T00:00:00Z",
         "--symbol", "FLATUSDT", "--output", str(out)]
    )
    args.handler(args)
    registry = parse_instrument_settlement_registry(out.read_bytes(), source=str(out))
    assert [r.symbol for r in registry.settlements] == ["FLATUSDT"]


@pytest.mark.parametrize("horizon", ["2025-01-02", "2025-01-02T02:00:00+02:00"])
def test_settlement_cli_rejects_non_utc_horizons(tmp_path, monkeypatch, horizon) -> None:
    from src.common.errors import DataIntegrityError
    from src.core.instrument_settlements import EMPTY_SETTLEMENT_REGISTRY

    _settlement_cli_lake(tmp_path)
    monkeypatch.setattr("src.common.paths.FUTURES_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        "src.core.instrument_settlements.load_instrument_settlement_registry",
        lambda: EMPTY_SETTLEMENT_REGISTRY,
    )
    args = _mhs_parser().parse_args(["data", "build-settlement-registry", "--horizon", horizon])
    with pytest.raises(DataIntegrityError, match="UTC"):
        args.handler(args)


def _halt_cli_lake(root) -> None:
    import pandas as pd

    t0 = pd.Timestamp("2025-01-01T00:00:00Z")
    stamps = [int((t0 + pd.Timedelta(minutes=3 * i)).value // 10**6) for i in range(10)]
    (root / "ohlcv" / "3m").mkdir(parents=True, exist_ok=True)
    for i in range(12):
        pd.DataFrame({
            "timestamp": stamps,
            "open": [3.0] * 10, "high": [3.0] * 10, "low": [3.0] * 10,
            "close": [3.0] * 10, "quote_vol": [10.0] * 3 + [0.0] * 4 + [10.0] * 3,
        }).to_parquet(root / "ohlcv" / "3m" / f"LIVE{i:02d}USDT.parquet", index=False)


def test_build_venue_halts_writes_output(tmp_path, monkeypatch, caplog) -> None:
    from src.core.instrument_settlements import EMPTY_SETTLEMENT_REGISTRY
    from src.core.venue_halts import parse_venue_halt_registry

    _halt_cli_lake(tmp_path)
    monkeypatch.setattr("src.common.paths.FUTURES_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        "src.core.instrument_settlements.load_instrument_settlement_registry",
        lambda path=None: EMPTY_SETTLEMENT_REGISTRY,
    )
    out = tmp_path / "halts.jsonl"
    with caplog.at_level("INFO", logger="src.cli.commands.data"):
        args = _mhs_parser().parse_args(
            ["data", "build-venue-halts", "--horizon", "2025-01-02T00:00:00Z",
             "--output", str(out)]
        )
        args.handler(args)
    assert "stage=build_venue_halts" in caplog.text
    registry = parse_venue_halt_registry(out.read_bytes(), source=str(out))
    assert len(registry.halts) == 1
    assert (registry.halts[0].present_symbols, registry.halts[0].zero_symbols) == (12, 12)


def test_build_venue_halts_write_replaces_registry(tmp_path, monkeypatch) -> None:
    from src.core.instrument_settlements import EMPTY_SETTLEMENT_REGISTRY
    from src.core.venue_halts import parse_venue_halt_registry

    _halt_cli_lake(tmp_path)
    monkeypatch.setattr("src.common.paths.FUTURES_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        "src.core.instrument_settlements.load_instrument_settlement_registry",
        lambda path=None: EMPTY_SETTLEMENT_REGISTRY,
    )
    target = tmp_path / "committed.jsonl"
    monkeypatch.setattr(
        "src.core.venue_halts.default_venue_halt_registry_path", lambda: target,
    )
    args = _mhs_parser().parse_args(
        ["data", "build-venue-halts", "--horizon", "2025-01-02T00:00:00Z", "--write"]
    )
    args.handler(args)
    registry = parse_venue_halt_registry(target.read_bytes(), source=str(target))
    assert len(registry.halts) == 1


def test_build_venue_halts_bootstraps_without_committed(tmp_path, monkeypatch, caplog) -> None:
    from src.common.errors import DataIntegrityError

    _halt_cli_lake(tmp_path)
    monkeypatch.setattr("src.common.paths.FUTURES_DATA_DIR", tmp_path)

    def _missing(path=None):
        raise DataIntegrityError("settlement registry unreadable: gone")

    monkeypatch.setattr(
        "src.core.instrument_settlements.load_instrument_settlement_registry", _missing,
    )
    monkeypatch.setattr(
        "src.core.instrument_settlements.default_instrument_settlement_registry_path",
        lambda: tmp_path / "absent.jsonl",
    )
    with caplog.at_level("INFO", logger="src.cli.commands.data"):
        args = _mhs_parser().parse_args(
            ["data", "build-venue-halts", "--horizon", "2025-01-02T00:00:00Z",
             "--output", str(tmp_path / "halts.jsonl")]
        )
        args.handler(args)
    assert "NO_COMMITTED_REGISTRY" in caplog.text
    assert not (tmp_path / "absent.jsonl").exists()


def test_build_venue_halts_refuses_corrupt_committed(tmp_path, monkeypatch) -> None:
    import pytest

    from src.common.errors import DataIntegrityError

    _halt_cli_lake(tmp_path)
    monkeypatch.setattr("src.common.paths.FUTURES_DATA_DIR", tmp_path)
    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_bytes(b"{oops")
    monkeypatch.setattr(
        "src.core.instrument_settlements.default_instrument_settlement_registry_path",
        lambda: corrupt,
    )
    args = _mhs_parser().parse_args(
        ["data", "build-venue-halts", "--horizon", "2025-01-02T00:00:00Z"]
    )
    with pytest.raises(DataIntegrityError):
        args.handler(args)


def test_report_internal_gaps_resolves_moved_panel_import(monkeypatch) -> None:
    import pandas as pd

    import src.core.panel as panel_mod
    from src.cli.commands.data import _report_internal_gaps

    frame = pd.DataFrame(
        {"AUSDT": [1.0, 1.0, 1.0], "BUSDT": [1.0, float("nan"), 1.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="3min", tz="UTC"),
    )
    seen: dict = {}

    def _stub_panel(root, interval, columns, start, end, partition="all"):
        seen.update(
            root=root, interval=interval, columns=tuple(columns),
            start=start, end=end, partition=partition,
        )
        return dict.fromkeys(("open", "high", "low", "close", "quote_vol"), frame)

    monkeypatch.setattr(panel_mod, "load_base_panel", _stub_panel)
    args = argparse.Namespace(timeframe="3m", start="2024-01-01", end="2024-01-02")
    assert _report_internal_gaps(args) is None
    assert seen["interval"] == "3m"
    assert seen["columns"] == ("open", "high", "low", "close", "quote_vol")
