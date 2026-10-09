"""Shared fixtures for strategy backtest CLI tests."""

from __future__ import annotations

import argparse
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.cli.commands.backtest as backtest_mod
from src.cli.main import build_root_parser


def _parse(argv: list[str]) -> argparse.Namespace:
    return build_root_parser(argv).parse_args(argv)

def _strategy_argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "backtest", "strategy",
        "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01",
        "--output", str(tmp_path / "strategy.json"), *extra,
    ]

def _install_strategy(monkeypatch: pytest.MonkeyPatch) -> dict:
    import src.engine.strategy_backtest as run_mod

    seen: dict = {}

    def _fake_run(request: object, *args: object, **kwargs: object) -> object:
        seen["request"] = request
        seen["source"] = kwargs.get("source")
        seen["snapshot_cache"] = kwargs.get("snapshot_cache")
        return types.SimpleNamespace(request=request)

    def _fake_persist(run: object, output: Path, **kwargs: object) -> Path:
        seen["output"] = output
        seen["statistics"] = kwargs.get("statistics")
        Path(output).write_text("{}", encoding="utf-8")
        return output

    monkeypatch.setattr(run_mod, "run_strategy_backtest", _fake_run)
    monkeypatch.setattr(run_mod, "load_strategy_source", lambda request: types.SimpleNamespace(request=request))
    monkeypatch.setattr(backtest_mod, "_strategy_run_statistics", lambda run: {})
    import src.engine.backtest_persist as report_mod

    monkeypatch.setattr(report_mod, "persist_strategy_backtest", _fake_persist)
    return seen

def _account_argv(*extra: str) -> list[str]:
    return [
        "backtest", "account",
        "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01",
        *extra,
    ]

def _install_account(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *,
    unit_fail: bool = False, unit_liquidated: bool = False, stub_venue: bool = True,
    unit_intraday: float = 0.05, account_intraday: float = 0.05,
) -> dict:
    from tests.unit.application._strategy_account_helpers import _install_strategy_account_fakes

    seen = _install_strategy_account_fakes(
        monkeypatch, tmp_path, unit_fail=unit_fail, unit_liquidated=unit_liquidated,
        stub_venue=stub_venue, unit_intraday=unit_intraday, account_intraday=account_intraday,
    )
    monkeypatch.setattr(backtest_mod, "STRATEGY_BACKTESTS_DIR", tmp_path / "x" / "runs")
    return seen

def _run_dirs(tmp_path: Path) -> list[Path]:
    root = tmp_path / "x" / "runs"
    return sorted(root.iterdir()) if root.is_dir() else []

def _write_held_parquet(root: Path, symbol: str, start: pd.Timestamp, bars: int, close: float = 100.0) -> pd.DatetimeIndex:
    grid = pd.date_range(start, periods=bars, freq="3min", tz="UTC")
    frame = pd.DataFrame({
        "timestamp": np.array([int(ts.value // 1_000_000) for ts in grid], dtype="int64"),
        "open": close, "high": close + 1.0, "low": close - 1.0, "close": close,
    })
    target = root / "3m"
    target.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target / f"{symbol}.parquet")
    return grid

def _assemble_fixture(tmp_path: Path) -> tuple:
    from src.strategy.targets import FLOW_MOM_TOP20, StrategyTargets
    from src.engine.strategy_backtest import StrategySourceContext
    from src.core.resources import MhsMemoryBudget

    entries = pd.date_range("2021-04-01", periods=2, freq="D", tz="UTC")
    weights = pd.DataFrame({"AAA": [0.05, -0.05], "BBB": [0.0, 0.0]}, index=entries, dtype="float64")
    candidate = StrategyTargets(
        target_weights=weights,
        signal_available_at=pd.DatetimeIndex(entries - pd.Timedelta(hours=1)),
        strategy=FLOW_MOM_TOP20,
    )
    grid = _write_held_parquet(tmp_path, "AAA", entries[0] - pd.Timedelta(hours=1), 1000)
    days = pd.date_range("2021-01-01", periods=92, freq="D", tz="UTC")
    qv = pd.DataFrame({"AAA": 5_000_000.0, "BBB": 1_000_000.0}, index=days, dtype="float64")
    drift = pd.DataFrame(
        {"AAA": 100.0 + 0.01 * pd.Series(range(92), index=days), "BBB": 50.0},
        index=days, dtype="float64",
    )
    funding_ts = pd.DatetimeIndex([entries[0] + pd.Timedelta(hours=8, minutes=1), entries[1] + pd.Timedelta(hours=8)])
    funding = pd.Series([0.0001, 0.0002], index=funding_ts, dtype="float64")
    context = StrategySourceContext(
        census=("AAA", "BBB"), funding_by_symbol={"AAA": funding}, funding_failures={},
        root=str(tmp_path), budget=MhsMemoryBudget(), daily_close=drift, daily_quote_volume=qv,
    )
    return candidate, context, entries, grid

def _patch_releases_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    import src.strategy.release as release_mod
    import shutil

    root = tmp_path / "releases"
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy(release_mod.release_path("flow_mom_top20"), root / "flow_mom_top20.json")
    monkeypatch.setattr(release_mod, "releases_dir", lambda root_arg=None: root)
    return root

def _account_namespace(tmp_path: Path, start: str = "2025-01-01", end: str = "2025-02-01") -> argparse.Namespace:
    return argparse.Namespace(
        source_start="2024-01-01", start=start, end=end,
        policy="growth", execution="taker", fixed_exposure=None, capital=2100.0,
        impact_y=0.6, no_order_filters=False, venue_rules=None, data_root=None,
        export_unit_returns=None, total_tree_pss_bytes=None,
        replay_tree_pss_bytes=None, min_available_bytes=None,
    )
