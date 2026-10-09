"""Shared fixtures for strategy account workflow tests."""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.application.strategy_account import (
    AccountReplayRequest,
)
from src.core.resources import MhsMemoryBudget


def _account_request(tmp_path: Path, **overrides) -> AccountReplayRequest:
    base: dict = {
        "source_start": pd.Timestamp("2024-01-01", tz="UTC"),
        "evaluation_start": pd.Timestamp("2025-01-01", tz="UTC"),
        "evaluation_end": pd.Timestamp("2025-02-01", tz="UTC"),
        "runs_root": tmp_path / "x" / "runs",
        "venue_rules_root": tmp_path / "venue",
    }
    base.update(overrides)
    return AccountReplayRequest(**base)

def _install_strategy_account_fakes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *,
    unit_fail: bool = False, unit_liquidated: bool = False, stub_venue: bool = True,
    unit_intraday: float = 0.05, account_intraday: float = 0.05,
) -> dict:
    import src.market_data.binance.venue_rules as venue_mod
    import src.engine.account_ledger as ledger_mod
    import src.engine.account_sources as sources_mod
    import src.engine.strategy_backtest as run_mod
    from src.common.errors import DataIntegrityError
    from src.engine.account_ledger import AccountLedgerResult
    from src.strategy.targets import FLOW_MOM_TOP20, StrategyTargets
    from src.engine.strategy_backtest import StrategySourceContext
    from src.market_data.binance.venue_rules import VenueRuleSnapshot

    seen: dict = {}
    dates = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    weights = pd.DataFrame({"AAA": [0.05, -0.05, 0.02]}, index=dates, dtype="float64")
    candidate = StrategyTargets(
        target_weights=weights,
        signal_available_at=pd.DatetimeIndex(dates - pd.Timedelta(hours=1)),
        strategy=FLOW_MOM_TOP20,
    )
    frame = pd.DataFrame({"AAA": 100.0}, index=dates, dtype="float64")
    context = StrategySourceContext(
        census=("AAA",), funding_by_symbol={}, funding_failures={}, root="root",
        budget=MhsMemoryBudget(), daily_close=frame, daily_quote_volume=frame,
    )

    def _fake_build(request: object) -> tuple:
        seen["request"] = request
        return candidate, context

    def _fake_assemble(cand: object, ctx: object) -> tuple:
        seen["candidate"] = cand
        anchors = pd.DatetimeIndex(dates - pd.Timedelta(hours=1))
        bars = pd.date_range(anchors[0], dates[-1] + pd.Timedelta(days=1), freq="3min", inclusive="left")
        plane = pd.DataFrame({"AAA": 100.0}, index=bars, dtype="float64")
        marks = ledger_mod.AccountMarkPanels(close=plane, high=plane, low=plane)
        unit = pd.DataFrame({"AAA": [0.05, -0.05, 0.02]}, index=dates, dtype="float64")
        flat = pd.DataFrame({"AAA": [0.0, 0.0, 0.0]}, index=dates, dtype="float64")
        rich = pd.DataFrame({"AAA": [1e9, 1e9, 1e9]}, index=dates, dtype="float64")
        calm = pd.DataFrame({"AAA": [0.02, 0.02, 0.02]}, index=dates, dtype="float64")
        seen["anchors"] = anchors
        return unit, marks, flat, rich, calm, anchors

    def _fake_replay(unit_w: object, marks: object, funding: object, adv: object, sigma: object, rules: object, policy: object, *, anchor_times: object = None, capital: float, taker_fee_bps: float, apply_order_filters: bool = True, unit_equity: pd.Series | None = None, execution: str = "taker", **execution_kwargs: object) -> AccountLedgerResult:
        seen.setdefault("replays", []).append({"capital": capital, "policy": policy, "filters": apply_order_filters, "fee": taker_fee_bps, "unit_equity": unit_equity, "execution": execution, "anchor_times": anchor_times, **execution_kwargs})
        if unit_fail and len(seen["replays"]) == 1:
            raise DataIntegrityError("unit boom")
        equity = pd.Series([capital, capital * 1.1, capital * 1.05], index=dates)
        exposure = pd.Series([1.0, 2.0, 1.5], index=dates)
        seen.setdefault("equities", []).append(equity)
        intraday = unit_intraday if len(seen["replays"]) == 1 else account_intraday
        return AccountLedgerResult(
            capital=capital, daily_equity=equity, daily_exposure=exposure,
            liquidated_at=dates[0] if unit_liquidated and len(seen["replays"]) == 1 else None,
            skipped_orders=3, untraded_fraction=0.01, initial_margin_breaches=0,
            fee_paid=1.0, impact_paid=2.0, funding_paid=0.5,
            fallback_ladder_symbols=(), missing_filter_symbols=(),
            intraday_max_drawdown=intraday,
            maker_fill_fraction=0.9 if execution == "maker" else 0.0,
        )

    snapshot = VenueRuleSnapshot(captured_at=pd.Timestamp("2026-01-01", tz="UTC"), symbols={})
    monkeypatch.setattr(run_mod, "build_request_targets", _fake_build)
    monkeypatch.setattr(sources_mod, "assemble_account_inputs", _fake_assemble)
    monkeypatch.setattr(ledger_mod, "replay_account", _fake_replay)
    if stub_venue:
        monkeypatch.setattr(venue_mod, "latest_venue_rule_snapshot", lambda root: tmp_path / "venue.json")
        monkeypatch.setattr(venue_mod, "load_venue_rule_snapshot", lambda path: snapshot)
    return seen

def _write_catalog_index(tmp_path: Path, rows: list[dict]) -> Path:
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return index

def _write_same_book_reference(
    tmp_path: Path, index: Path, *, run_dir: str = "runs/ref", execution: str | None = None,
    name_clip: float | None = 0.05, exposure_multiplier: float | None = 1.0,
    base_cagr: float | None = None, base_mdd: float | None = None,
    evaluation_start: str = "2025-01-01T00:00:00+00:00",
    evaluation_end: str = "2025-02-01T00:00:00+00:00",
) -> dict:
    row: dict = {
        "kind": "mhs_frozen", "strategy_id": "frozen_mhs_top20_v2", "run_dir": run_dir,
        "evaluation_start": evaluation_start, "evaluation_end": evaluation_end,
        "base_cagr": base_cagr, "base_max_drawdown": base_mdd,
    }
    if execution is not None:
        row["execution"] = execution
    with index.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    result_dir = tmp_path / run_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "result.json").write_text(
        json.dumps({"name_clip": name_clip, "exposure_multiplier": exposure_multiplier}),
        encoding="utf-8",
    )
    return row

def _fake_run_dir(tmp_path: Path, multiplier: float = 2.5) -> Path:
    run_dir = tmp_path / "strategy_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy_id": "frozen_mhs_top20_growth_v2",
        "breadth": 20,
        "exposure_multiplier": multiplier,
        "execution_bound": "OHLCV_IMMEDIATE_TAKER",
        "source_start": "2024-01-01T00:00:00+00:00",
        "evaluation_start": "2025-01-01T00:00:00+00:00",
        "evaluation_end": "2025-02-01T00:00:00+00:00",
    }
    (run_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    idx = pd.date_range("2025-01-01", periods=40, freq="D", tz="UTC")
    pd.DataFrame(
        {"base_return": np.full(len(idx), 0.001), "max_name_weight": np.full(len(idx), 0.05)},
        index=idx,
    ).to_parquet(run_dir / "daily.parquet")
    return run_dir

def _install_exposure_fakes(monkeypatch: pytest.MonkeyPatch) -> dict:
    import src.strategy.universe as universe_mod
    import src.evaluation.exposure as growth_mod
    import src.core.panel as panel_mod
    import src.core.resources as resources_mod

    seen: dict = {}

    def _fake_panel(root, interval, columns, start, end, partition="all", selection_mode="causal_history", allocation_admission=None):
        seen["selection_mode"] = selection_mode
        if allocation_admission is not None:
            allocation_admission(1024)
        idx = pd.date_range(start, end, freq="h", tz="UTC")
        return {name: pd.DataFrame(100.0, index=idx, columns=["AAA", "BBB"], dtype="float64") for name in columns}

    def _fake_roster(daily_close, daily_quote_volume, census, *, breadth, blocked_decisions=None):
        seen["breadth"] = breadth
        seen["blocked_decisions"] = blocked_decisions
        return pd.DataFrame(False, index=daily_close.index, columns=list(census), dtype=bool)

    def _fake_solve(unit_returns, *, max_name_weight, gaps, mean_haircut, grid, plateau_tolerance, n_paths, horizon_years, mean_block_days, seed):
        seen["unit_returns"] = unit_returns
        seen["max_name_weight"] = max_name_weight
        seen["gaps"] = gaps
        seen["solver_params"] = {
            "mean_haircut": mean_haircut, "grid": grid, "plateau_tolerance": plateau_tolerance,
            "n_paths": n_paths, "horizon_years": horizon_years, "mean_block_days": mean_block_days, "seed": seed,
        }
        grid_tuple = tuple(grid)
        return types.SimpleNamespace(
            grid=grid_tuple, growth=(0.1,) * len(grid_tuple), ruin_probability=(0.0,) * len(grid_tuple),
            argmax=grid_tuple[-1], chosen=grid_tuple[0], gap_events_per_year=1.5, gap_sample_size=3,
        )

    monkeypatch.setattr(panel_mod, "load_base_panel", _fake_panel)
    monkeypatch.setattr(universe_mod, "build_pit_roster", _fake_roster)
    monkeypatch.setattr(growth_mod, "solve_log_growth_exposure", _fake_solve)
    monkeypatch.setattr(growth_mod, "structurally_excluded_symbols", lambda: frozenset())
    monkeypatch.setattr(resources_mod, "resolve_mhs_memory_budget", lambda budget: budget)
    monkeypatch.setattr(resources_mod, "assert_mhs_stage_allocation", lambda **kwargs: None)
    return seen

def _run_dirs(tmp_path: Path) -> list[Path]:
    root = tmp_path / "x" / "runs"
    return sorted(root.iterdir()) if root.is_dir() else []

_PRE_REFACTOR = json.loads(
    (Path(__file__).resolve().parents[2] / "fixtures" / "strategy_account" / "pre_refactor_payloads.json").read_text(encoding="utf-8")
)

_PINNED_NOW = pd.Timestamp("2026-03-04T05:06:07", tz="UTC")

_UNIT_CAGR = (105000.0 / 100000.0) ** (365.0 / 3.0) - 1.0

_GOLDEN_ACCOUNT_CASES = {
    "missing_reference": (None, {}),
    "ok": ((_UNIT_CAGR, 0.05), {}),
    "mismatch": ((_UNIT_CAGR + 1.0, 0.5), {}),
    "maker_missing": (None, {"execution": "maker"}),
    "fixed_ok": ((_UNIT_CAGR, 0.05), {"policy": "fixed", "fixed_exposure": 2.5}),
}
