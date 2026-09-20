"""Invariant scenarios for execution coverage measurement and backfill planning."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.market_data.services.execution_coverage import (
    ExecutionCoverageDeficit,
    measure_execution_coverage,
    plan_execution_coverage_backfill,
)


def _ms(idx: pd.DatetimeIndex) -> list[int]:
    return [int(t.value // 10**6) for t in idx]


def _write_ohlcv(root: Path, symbol: str, timeframe: str, idx: pd.DatetimeIndex) -> Path:
    directory = root / "ohlcv" / timeframe
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "timestamp": _ms(idx),
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
    })
    path = directory / f"{symbol}.parquet"
    frame.to_parquet(path, index=False)
    return path


def _h1(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="1h", tz="UTC")


def _m3(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="3min", tz="UTC")


def _deficit(
    symbol: str = "AAAUSDT",
    signal_end: str = "2022-03-01T00:00:00Z",
    execution_end: str | None = "2022-02-01T00:00:00Z",
    horizon: str = "2022-03-01T00:00:00Z",
    deficit_days: int = 28,
) -> ExecutionCoverageDeficit:
    return ExecutionCoverageDeficit(
        symbol=symbol,
        signal_end=pd.Timestamp(signal_end, tz="UTC"),
        execution_end=pd.Timestamp(execution_end, tz="UTC") if execution_end is not None else None,
        horizon=pd.Timestamp(horizon, tz="UTC"),
        deficit_days=deficit_days,
    )


def test_measure_truncated_symbol_reports_sixty_day_deficit(tmp_path: Path) -> None:
    root = tmp_path / "lake"
    signal = _h1("2021-12-01T00:00:00Z", "2022-03-01T00:00:00Z")
    _write_ohlcv(root, "AAAUSDT", "1h", signal)
    signal_end = signal[-1]
    execution_end = signal_end - timedelta(days=60)
    execution = _m3("2021-12-01T00:00:00Z", execution_end.isoformat())
    _write_ohlcv(root, "AAAUSDT", "3m", execution)
    deficits = measure_execution_coverage(
        horizon_end=pd.Timestamp("2022-03-01T00:00:00Z"), data_root=root,
    )
    assert len(deficits) == 1
    assert deficits[0].symbol == "AAAUSDT"
    assert deficits[0].deficit_days == 60


def test_measure_missing_execution_file_covers_full_signal_span(tmp_path: Path) -> None:
    root = tmp_path / "lake"
    signal = pd.date_range("2022-01-01T00:00:00Z", "2022-01-11T00:00:00Z", freq="1h", tz="UTC")
    _write_ohlcv(root, "AAAUSDT", "1h", signal)
    deficits = measure_execution_coverage(
        horizon_end=pd.Timestamp("2022-02-01T00:00:00Z"), data_root=root,
    )
    assert len(deficits) == 1
    assert deficits[0].execution_end is None
    assert deficits[0].deficit_days == 10
    assert deficits[0].horizon == pd.Timestamp("2022-01-11T00:00:00Z")


def test_measure_delisted_symbol_ending_together_is_not_deficient(tmp_path: Path) -> None:
    root = tmp_path / "lake"
    signal = pd.date_range("2022-01-01T00:00:00Z", "2022-02-01T00:00:00Z", freq="1h", tz="UTC")
    execution = pd.date_range("2022-01-01T00:00:00Z", "2022-02-01T00:00:00Z", freq="3min", tz="UTC")
    _write_ohlcv(root, "AAAUSDT", "1h", signal)
    _write_ohlcv(root, "AAAUSDT", "3m", execution)
    deficits = measure_execution_coverage(
        horizon_end=pd.Timestamp("2022-06-01T00:00:00Z"), data_root=root,
    )
    assert deficits == ()


def test_measure_applies_horizon_cap_for_extended_signal(tmp_path: Path) -> None:
    root = tmp_path / "lake"
    _write_ohlcv(root, "AAAUSDT", "1h", _h1("2022-01-01T00:00:00Z", "2022-04-01T00:00:00Z"))
    _write_ohlcv(root, "AAAUSDT", "3m", _m3("2022-01-01T00:00:00Z", "2022-02-01T00:00:00Z"))
    horizon_end = pd.Timestamp("2022-03-01T00:00:00Z")
    deficits = measure_execution_coverage(horizon_end=horizon_end, data_root=root)
    assert len(deficits) == 1
    assert deficits[0].horizon == horizon_end
    assert deficits[0].deficit_days == 28


def test_measure_ignores_sub_bar_difference(tmp_path: Path) -> None:
    root = tmp_path / "lake"
    signal = _h1("2022-01-01T00:00:00Z", "2022-01-05T00:00:00Z")
    _write_ohlcv(root, "AAAUSDT", "1h", signal)
    full = _m3("2022-01-01T00:00:00Z", "2022-01-05T00:00:00Z")
    _write_ohlcv(root, "AAAUSDT", "3m", full[:-1])
    deficits = measure_execution_coverage(
        horizon_end=pd.Timestamp("2022-01-05T00:00:00Z"), data_root=root,
    )
    assert deficits == ()


def test_measure_orders_by_descending_deficit_deterministically(tmp_path: Path) -> None:
    root = tmp_path / "lake"
    signal = _h1("2021-12-01T00:00:00Z", "2022-03-01T00:00:00Z")
    signal_end = signal[-1]
    for symbol in ("AAAUSDT", "BBBUSDT", "CCCUSDT"):
        _write_ohlcv(root, symbol, "1h", signal)
    ends = {
        "AAAUSDT": signal_end - timedelta(days=30),
        "BBBUSDT": signal_end - timedelta(days=10),
        "CCCUSDT": signal_end - timedelta(days=20),
    }
    for symbol, end in ends.items():
        _write_ohlcv(root, symbol, "3m", _m3("2021-12-01T00:00:00Z", end.isoformat()))
    first = measure_execution_coverage(
        horizon_end=pd.Timestamp("2022-03-01T00:00:00Z"), data_root=root,
    )
    second = measure_execution_coverage(
        horizon_end=pd.Timestamp("2022-03-01T00:00:00Z"), data_root=root,
    )
    assert [d.symbol for d in first] == ["AAAUSDT", "CCCUSDT", "BBBUSDT"]
    assert [d.deficit_days for d in first] == [30, 20, 10]
    assert first == second


def test_plan_re_reads_overlap_before_resume_point() -> None:
    deficit = _deficit(execution_end="2022-02-01T00:00:00Z", horizon="2022-03-01T00:00:00Z")
    windows = plan_execution_coverage_backfill(
        (deficit,), horizon_end=pd.Timestamp("2022-03-01T00:00:00Z"), lookback_days=3,
    )
    assert len(windows) == 1
    symbol, start, end = windows[0]
    assert symbol == "AAAUSDT"
    assert start == pd.Timestamp("2022-01-29T00:00:00Z")
    assert end == pd.Timestamp("2022-03-01T00:00:00Z")


def test_plan_rejects_already_covered_deficit() -> None:
    covered = _deficit(
        execution_end="2022-03-01T00:00:00Z", horizon="2022-03-01T00:00:00Z", deficit_days=0,
    )
    with pytest.raises(DataIntegrityError):
        plan_execution_coverage_backfill(
            (covered,), horizon_end=pd.Timestamp("2022-03-01T00:00:00Z"),
        )


def test_cli_defaults_to_read_only_without_touching_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse

    import src.market_data.services.execution_coverage as coverage_mod
    from src.cli.commands.data import add_data_commands
    import src.market_data.services.collection as collection_mod

    root = tmp_path / "lake"
    signal = _h1("2022-01-01T00:00:00Z", "2022-02-01T00:00:00Z")
    _write_ohlcv(root, "AAAUSDT", "1h", signal)
    exec_idx = _m3("2022-01-01T00:00:00Z", "2022-01-15T00:00:00Z")
    exec_path = _write_ohlcv(root, "AAAUSDT", "3m", exec_idx)
    before = exec_path.read_bytes()
    monkeypatch.setattr(coverage_mod, "FUTURES_DATA_DIR", root)

    parser = argparse.ArgumentParser(prog="cli")
    sub = parser.add_subparsers(dest="command", required=True)
    data_parser = sub.add_parser("data")
    add_data_commands(data_parser)
    args = parser.parse_args(["data", "sync-execution-coverage", "--end", "2022-02-01T00:00:00Z"])
    assert args.lookback_days == 3
    assert args.execute is False
    assert args.symbol is None

    calls: list[tuple[str, ...]] = []

    def _forbidden(*call_args: object, **call_kwargs: object) -> None:
        calls.append(tuple(str(a) for a in call_args))

    monkeypatch.setattr(collection_mod, "collect_ohlcv", _forbidden)
    args.handler(args)
    assert calls == []
    assert exec_path.read_bytes() == before


def test_measure_ignores_execution_only_symbol(tmp_path: Path) -> None:
    root = tmp_path / "lake"
    _write_ohlcv(root, "GHOSTUSDT", "3m", _m3("2022-01-01T00:00:00Z", "2022-01-05T00:00:00Z"))
    deficits = measure_execution_coverage(
        horizon_end=pd.Timestamp("2022-02-01T00:00:00Z"),
        symbols=["GHOSTUSDT"], data_root=root,
    )
    assert deficits == ()


def test_measure_respects_symbol_filter_and_empty_root(tmp_path: Path) -> None:
    root = tmp_path / "lake"
    signal = _h1("2021-12-01T00:00:00Z", "2022-03-01T00:00:00Z")
    signal_end = signal[-1]
    for symbol in ("AAAUSDT", "BBBUSDT"):
        _write_ohlcv(root, symbol, "1h", signal)
        _write_ohlcv(
            root, symbol, "3m", _m3("2021-12-01T00:00:00Z", (signal_end - timedelta(days=15)).isoformat()),
        )
    deficits = measure_execution_coverage(
        horizon_end=pd.Timestamp("2022-03-01T00:00:00Z"), symbols=["BBBUSDT"], data_root=root,
    )
    assert [d.symbol for d in deficits] == ["BBBUSDT"]
    assert measure_execution_coverage(
        horizon_end=pd.Timestamp("2022-02-01T00:00:00Z"), data_root=tmp_path / "no-lake",
    ) == ()


def test_measure_skips_empty_signal_and_treats_empty_execution_as_absent(tmp_path: Path) -> None:
    root = tmp_path / "lake"
    (root / "ohlcv" / "1h").mkdir(parents=True)
    pd.DataFrame({"timestamp": pd.Series(dtype="int64")}).to_parquet(
        root / "ohlcv" / "1h" / "EMPTYUSDT.parquet", index=False,
    )
    signal = pd.date_range("2022-01-01T00:00:00Z", "2022-01-11T00:00:00Z", freq="1h", tz="UTC")
    _write_ohlcv(root, "ABSENTUSDT", "1h", signal)
    (root / "ohlcv" / "3m").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": pd.Series(dtype="int64")}).to_parquet(
        root / "ohlcv" / "3m" / "ABSENTUSDT.parquet", index=False,
    )
    deficits = measure_execution_coverage(
        horizon_end=pd.Timestamp("2022-02-01T00:00:00Z"), data_root=root,
    )
    assert [d.symbol for d in deficits] == ["ABSENTUSDT"]
    assert deficits[0].execution_end is None
    assert deficits[0].deficit_days == 10


def test_measure_rejects_bad_horizon_and_unreadable_parquet(tmp_path: Path) -> None:
    root = tmp_path / "lake"
    with pytest.raises(DataIntegrityError):
        measure_execution_coverage(horizon_end="2022-01-01", data_root=root)  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError):
        measure_execution_coverage(horizon_end=pd.Timestamp("2022-01-01"), data_root=root)
    with pytest.raises(DataIntegrityError):
        measure_execution_coverage(
            horizon_end=pd.Timestamp("2022-01-01T09:00:00+09:00"), data_root=root,
        )
    with pytest.raises(DataIntegrityError):
        plan_execution_coverage_backfill(
            (), horizon_end=pd.Timestamp("2022-01-01"), lookback_days=3,
        )
    with pytest.raises(DataIntegrityError):
        plan_execution_coverage_backfill(
            (), horizon_end=pd.Timestamp("2022-01-01T00:00:00Z"), lookback_days=-1,
        )
    directory = root / "ohlcv" / "1h"
    directory.mkdir(parents=True)
    (directory / "BROKENUSDT.parquet").write_bytes(b"not a parquet file")
    with pytest.raises(DataIntegrityError):
        measure_execution_coverage(
            horizon_end=pd.Timestamp("2022-02-01T00:00:00Z"),
            symbols=["BROKENUSDT"], data_root=root,
        )


def test_measure_reads_datetime_column_and_rejects_time_less_files(tmp_path: Path) -> None:
    root = tmp_path / "lake"
    one_h = root / "ohlcv" / "1h"
    three_m = root / "ohlcv" / "3m"
    one_h.mkdir(parents=True)
    three_m.mkdir(parents=True)
    idx_1h = _h1("2022-01-01T00:00:00Z", "2022-01-05T00:00:00Z")
    pd.DataFrame({"datetime": idx_1h, "close": 1.0}).to_parquet(
        one_h / "DTUSDT.parquet", index=False,
    )
    pd.DataFrame({"datetime": _m3("2022-01-01T00:00:00Z", "2022-01-05T00:00:00Z"), "close": 1.0}).to_parquet(
        three_m / "DTUSDT.parquet", index=False,
    )
    assert measure_execution_coverage(
        horizon_end=pd.Timestamp("2022-01-05T00:00:00Z"), symbols=["DTUSDT"], data_root=root,
    ) == ()
    pd.DataFrame({"timestamp": [float("nan")] * 3}).to_parquet(
        one_h / "NANUSDT.parquet", index=False,
    )
    assert measure_execution_coverage(
        horizon_end=pd.Timestamp("2022-01-05T00:00:00Z"), symbols=["NANUSDT"], data_root=root,
    ) == ()
    pd.DataFrame({"datetime": [pd.NaT, pd.NaT]}).to_parquet(
        one_h / "NANDTUSDT.parquet", index=False,
    )
    assert measure_execution_coverage(
        horizon_end=pd.Timestamp("2022-01-05T00:00:00Z"), symbols=["NANDTUSDT"], data_root=root,
    ) == ()
    pd.DataFrame({"open": [1.0, 2.0]}).to_parquet(one_h / "NOTIMEUSDT.parquet", index=False)
    with pytest.raises(DataIntegrityError):
        measure_execution_coverage(
            horizon_end=pd.Timestamp("2022-01-05T00:00:00Z"), symbols=["NOTIMEUSDT"], data_root=root,
        )


def test_plan_starts_absent_symbol_at_signal_start() -> None:
    deficit = _deficit(
        symbol="ABSENTUSDT", signal_end="2022-01-11T00:00:00Z", execution_end=None,
        horizon="2022-01-11T00:00:00Z", deficit_days=10,
    )
    windows = plan_execution_coverage_backfill(
        (deficit,), horizon_end=pd.Timestamp("2022-01-11T00:00:00Z"),
    )
    assert windows[0][1] == pd.Timestamp("2022-01-01T00:00:00Z")
    assert windows[0][2] == pd.Timestamp("2022-01-11T00:00:00Z")


def test_cli_execute_collects_and_reports_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from argparse import Namespace

    import src.market_data.services.collection as collection_mod
    import src.market_data.services.execution_coverage as coverage_mod
    from src.cli.commands.data import _sync_execution_coverage

    root = tmp_path / "lake"
    signal = _h1("2022-01-01T00:00:00Z", "2022-02-01T00:00:00Z")
    _write_ohlcv(root, "AAAUSDT", "1h", signal)
    _write_ohlcv(root, "AAAUSDT", "3m", _m3("2022-01-01T00:00:00Z", "2022-01-15T00:00:00Z"))
    monkeypatch.setattr(coverage_mod, "FUTURES_DATA_DIR", root)
    recorded: list[tuple[str, str, str, str]] = []
    monkeypatch.setattr(
        collection_mod, "collect_ohlcv",
        lambda symbol, timeframe, start, end: recorded.append((symbol, timeframe, start, end)),
    )
    args = Namespace(end="2022-02-01T00:00:00Z", symbol=None, lookback_days=3, execute=True)
    _sync_execution_coverage(args)
    assert len(recorded) == 1
    assert recorded[0][0] == "AAAUSDT"
    assert recorded[0][1] == "3m"

    def _boom(symbol: str, timeframe: str, start: str, end: str) -> None:
        raise RuntimeError("network down")

    monkeypatch.setattr(collection_mod, "collect_ohlcv", _boom)
    with pytest.raises(SystemExit) as exc_info:
        _sync_execution_coverage(args)
    assert exc_info.value.code == 1
