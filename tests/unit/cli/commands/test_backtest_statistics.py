"""Invariant scenarios for backtest statistics and trial-ledger CLI leaves."""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.cli.commands.backtest as backtest_mod
from tests.unit.cli.commands._backtest_helpers import (
    _account_namespace,
    _patch_releases_root,
)


@pytest.fixture(autouse=True)
def _isolated_release_ledgers(tmp_path, monkeypatch):
    import shutil
    import src.strategy.release as release_mod

    source = release_mod.release_path("flow_mom_top20")
    root = tmp_path / "default_releases"
    root.mkdir()
    shutil.copy(source, root / "flow_mom_top20.json")
    monkeypatch.setattr(release_mod, "releases_dir", lambda root_arg=None: root)

def test_strategy_run_statistics_reports_base_and_stress() -> None:
    """The CLI statistics step maps both ledgers to JSON-safe decision-grade payloads."""
    import dataclasses

    from src.engine.backtest_evidence import StrategyReportPeriod, evaluate_strategy_backtest
    from src.engine.strategy_backtest import StrategyBacktestRequest
    from src.strategy.targets import FLOW_MOM_TOP20, StrategyTargets
    from src.core.types import ExecutionSpec

    base = dataclasses.replace(
        ExecutionSpec(), taker_fee_bps=5.0, taker_slippage_bps=1.0, decision_anchor="submit_bar",
    )
    stress = dataclasses.replace(
        ExecutionSpec(), taker_fee_bps=5.0, taker_slippage_bps=13.0, decision_anchor="submit_bar",
    )
    symbols = ("AAA", "BBB")
    labels = [pd.Timestamp("2021-06-02", tz="UTC") + pd.Timedelta(days=i) for i in range(3)]
    weights = pd.DataFrame(
        {"AAA": [0.05] * 3, "BBB": [-0.05] * 3}, index=pd.DatetimeIndex(labels, tz="UTC"), dtype="float64",
    )
    avail = pd.DatetimeIndex([label - pd.Timedelta(hours=1) for label in labels], tz="UTC")
    candidate = StrategyTargets(target_weights=weights, signal_available_at=avail, strategy=FLOW_MOM_TOP20)

    def _window(grid: pd.DatetimeIndex, chosen: list) -> object:
        from src.engine.execution import ExecutionReplayWindow

        params: dict = {
            "window_start": grid[0], "window_end": grid[-1], "columns": symbols, "symbols": symbols,
            "minute_grid": grid, "target_weights": weights.loc[chosen].copy(),
            "signal_available_at": pd.DatetimeIndex(
                [avail[weights.index.get_loc(label)] for label in chosen], tz="UTC"
            ),
            "bar_available_at": grid + pd.Timedelta(minutes=3),
            "highs": pd.DataFrame(101.0, index=grid, columns=list(symbols), dtype="float64"),
            "lows": pd.DataFrame(99.0, index=grid, columns=list(symbols), dtype="float64"),
            "closes": pd.DataFrame(100.0, index=grid, columns=list(symbols), dtype="float64"),
            "marks": pd.DataFrame(100.0, index=grid, columns=list(symbols), dtype="float64"),
            "bar_funding": pd.DataFrame(0.0, index=grid, columns=list(symbols), dtype="float64"),
            "quote_volumes": pd.DataFrame(1000.0, index=grid, columns=list(symbols), dtype="float64"),
            "funding_known": pd.DataFrame(True, index=grid, columns=list(symbols)),
        }
        return ExecutionReplayWindow(**params)  # type: ignore[arg-type]

    first = pd.date_range(labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=1), freq="3min", tz="UTC")
    second = pd.date_range(labels[1] - pd.Timedelta(hours=2), labels[2] + pd.Timedelta(hours=2), freq="3min", tz="UTC")
    periods = (StrategyReportPeriod(label="P1", start=labels[0], end=labels[1]),)
    evidence = evaluate_strategy_backtest(
        candidate, iter([_window(first, labels[:1]), _window(second, labels[1:])]),
        initial_equity=100000.0, base_spec=base, stress_spec=stress, report_periods=periods,
    )
    request = StrategyBacktestRequest(
        source_start=pd.Timestamp("2021-01-01", tz="UTC"), evaluation_start=labels[0],
        evaluation_end=labels[-1] + pd.Timedelta(days=1), strategy=FLOW_MOM_TOP20,
        initial_equity=100000.0, base_spec=base, stress_spec=stress, report_periods=periods,
    )
    import types as _types

    run = _types.SimpleNamespace(request=request, candidate=candidate, evidence=evidence)
    first_stats = backtest_mod._strategy_run_statistics(run)
    second_stats = backtest_mod._strategy_run_statistics(run)
    assert set(first_stats) == {"base", "stress"}
    assert first_stats == second_stats
    assert first_stats["base"]["in_sample_days"] == 3
    assert first_stats["base"]["bootstrap"]["seed"] == 20261008

    legacy_evidence = dataclasses.replace(
        evidence,
        base=dataclasses.replace(
            evidence.base,
            ledger=dataclasses.replace(
                evidence.base.ledger, funding_by_symbol_daily=None, funding_by_symbol={"AAA": -10.0},
            ),
        ),
        stress=dataclasses.replace(
            evidence.stress,
            ledger=dataclasses.replace(
                evidence.stress.ledger, funding_by_symbol_daily=None, funding_by_symbol={"AAA": -5.0},
            ),
        ),
    )
    legacy_run = _types.SimpleNamespace(request=request, candidate=candidate, evidence=legacy_evidence)
    legacy_stats = backtest_mod._strategy_run_statistics(legacy_run)
    assert set(legacy_stats) == {"base", "stress"}
    assert legacy_stats["base"]["funding"]["total_contribution"] == pytest.approx(-10.0 / 100000.0)

def test_trial_family_maps_flow_mom_lineage() -> None:
    assert backtest_mod._trial_family("flow_mom_top20") == "flow_mom"
    assert backtest_mod._trial_family("other") == "other"

def test_record_trial_for_window_appends_before_cutoff(tmp_path, monkeypatch) -> None:
    import json

    from src.strategy.targets import FLOW_MOM_TOP20

    root = _patch_releases_root(monkeypatch, tmp_path)
    backtest_mod._record_trial_for_window(
        FLOW_MOM_TOP20,
        pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-02-01", tz="UTC"),
        0.05, 31,
    )
    rows = (root / "flow_mom.trials.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["source"] == "cli"
    assert row["daily_sharpe"] == 0.05
    backtest_mod._record_trial_for_window(
        FLOW_MOM_TOP20,
        pd.Timestamp("2026-08-01", tz="UTC"), pd.Timestamp("2026-09-01", tz="UTC"),
        0.05, 31,
    )
    assert len((root / "flow_mom.trials.jsonl").read_text(encoding="utf-8").strip().splitlines()) == 1

def test_record_trial_for_window_falls_back_on_digest_failure(tmp_path, monkeypatch) -> None:
    import json
    import types

    root = _patch_releases_root(monkeypatch, tmp_path)
    stub = types.SimpleNamespace(
        strategy_id="flow_mom_top20",
        design_data_cutoff=pd.Timestamp("2026-07-01", tz="UTC"),
        name_clip=0.05,
        exposure_multiplier=1.0,
    )
    backtest_mod._record_trial_for_window(
        stub, pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-02-01", tz="UTC"), None, 31,
    )
    row = json.loads((root / "flow_mom.trials.jsonl").read_text(encoding="utf-8").strip())
    assert row["spec_digest"] == "flow_mom_top20"
    assert row["daily_sharpe"] is None

def test_account_command_suppresses_trial_ledger_failure(tmp_path, monkeypatch) -> None:
    """A broken trial journal never fails an otherwise successful account replay."""
    import src.application.strategy_account as account_mod
    import src.strategy.release as release_mod

    def _boom(root_arg=None) -> Path:
        raise OSError("journal unavailable")

    monkeypatch.setattr(release_mod, "releases_dir", _boom)
    index = pd.date_range("2025-01-01", periods=31, freq="D", tz="UTC")
    equity = pd.Series(np.linspace(2100.0, 2200.0, 31), index=index, dtype="float64")
    report = types.SimpleNamespace(
        result=types.SimpleNamespace(
            daily_equity=equity, liquidated_at=None, maker_fill_fraction=0.98,
        ),
        payload={"moment_source": "stub", "cagr": 0.5, "mdd": 0.05},
    )
    monkeypatch.setattr(account_mod, "run_account_replay", lambda request: report)
    backtest_mod.run_account_replay_command(_account_namespace(tmp_path))
