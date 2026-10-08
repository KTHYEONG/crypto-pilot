"""Invariant scenarios for frozen account and exposure application services."""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.application.mhs_frozen_account import (
    FrozenAccountRequest,
    FrozenExposureRequest,
    derive_frozen_exposure,
    frozen_execution_specs,
    load_frozen_run_artifacts,
    reconcile_unit_reference,
    run_frozen_account,
    run_frozen_exposure,
)
from src.core.resources import MhsMemoryBudget


def _account_request(tmp_path: Path, **overrides) -> FrozenAccountRequest:
    base: dict = {
        "source_start": pd.Timestamp("2024-01-01", tz="UTC"),
        "evaluation_start": pd.Timestamp("2025-01-01", tz="UTC"),
        "evaluation_end": pd.Timestamp("2025-02-01", tz="UTC"),
        "runs_root": tmp_path / "x" / "runs",
        "venue_rules_root": tmp_path / "venue",
    }
    base.update(overrides)
    return FrozenAccountRequest(**base)


def _install_frozen_account_fakes(
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
    from src.strategy.targets import FROZEN_MHS_TOP20_V2, FrozenMhsCandidate
    from src.engine.strategy_backtest import FrozenSourceContext
    from src.market_data.binance.venue_rules import VenueRuleSnapshot

    seen: dict = {}
    dates = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    weights = pd.DataFrame({"AAA": [0.05, -0.05, 0.02]}, index=dates, dtype="float64")
    candidate = FrozenMhsCandidate(
        target_weights=weights,
        signal_available_at=pd.DatetimeIndex(dates - pd.Timedelta(hours=1)),
        strategy=FROZEN_MHS_TOP20_V2,
    )
    frame = pd.DataFrame({"AAA": 100.0}, index=dates, dtype="float64")
    context = FrozenSourceContext(
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
    monkeypatch.setattr(run_mod, "build_frozen_request_candidate", _fake_build)
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
    run_dir = tmp_path / "frozen_run"
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
    monkeypatch.setattr(universe_mod, "build_frozen_pit_roster", _fake_roster)
    monkeypatch.setattr(growth_mod, "solve_log_growth_exposure", _fake_solve)
    monkeypatch.setattr(growth_mod, "structurally_excluded_symbols", lambda: frozenset())
    monkeypatch.setattr(resources_mod, "resolve_mhs_memory_budget", lambda budget: budget)
    monkeypatch.setattr(resources_mod, "assert_mhs_stage_allocation", lambda **kwargs: None)
    return seen


def _run_dirs(tmp_path: Path) -> list[Path]:
    root = tmp_path / "x" / "runs"
    return sorted(root.iterdir()) if root.is_dir() else []


def test_frozen_specs_use_submit_anchor() -> None:
    """Both frozen cost cases cross from the last pre-submission mark at 6 and 18 bps."""
    base, stress = frozen_execution_specs()
    assert base.decision_anchor == "submit_bar"
    assert stress.decision_anchor == "submit_bar"
    assert base.one_way_taker_bps() == 6.0
    assert stress.one_way_taker_bps() == 18.0


def test_account_payload_golden_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Growth/taker run persists the exact account.json key sets and sign conventions."""
    seen = _install_frozen_account_fakes(monkeypatch, tmp_path)
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps({"kind": "mhs", "strategy_id": "x", "base_cagr": 0.1}) + "\n", encoding="utf-8")
    unit_cagr = (105000.0 / 100000.0) ** (365.0 / 3.0) - 1.0
    _write_same_book_reference(tmp_path, index, run_dir="runs/old", base_cagr=unit_cagr, base_mdd=0.05)
    report = run_frozen_account(_account_request(tmp_path))
    assert len(seen["replays"]) == 2
    (run_dir,) = _run_dirs(tmp_path)
    payload = json.loads((run_dir / "account.json").read_text(encoding="utf-8"))
    assert set(payload) == {"strategy_id", "capital", "execution", "policy", "venue_captured_at", "venue_path", "evaluation_start", "evaluation_end", "cagr", "mdd", "daily_mdd", "final_equity", "liquidated_at", "mean_exposure", "min_exposure", "last_exposure", "skipped_orders", "untraded_fraction", "initial_margin_breaches", "fee_paid", "impact_paid", "funding_paid", "fallback_ladder_symbols", "missing_filter_symbols", "moment_source", "entry_anchor", "unit_reference", "venue_rules_applied_retroactively", "reconciliation", "created_at"}
    assert set(payload["execution"]) == {"mode", "maker_fee_bps", "taker_fee_bps", "passive_window_bars", "maker_fill_fraction"}
    assert set(payload["policy"]) == {"kind", "exposure_max", "exposure_step", "mean_haircut", "prior_days", "min_moment_days", "shock_per_unit", "margin_reserve", "initial_margin_cap", "impact_y"}
    assert set(payload["unit_reference"]) == {"capital", "cagr", "mdd", "daily_mdd", "maker_fill_fraction"}
    assert set(payload["reconciliation"]) == {"status", "fixed_exposure", "capital", "order_filters", "impact_y", "cagr", "mdd", "mdd_convention", "mdd_definition", "cagr_tolerance", "mdd_tolerance", "reference_canonical", "cagr_gap", "mdd_gap"}
    assert set(payload["reconciliation"]["reference_canonical"]) == {"strategy_id", "run_dir", "evaluation_start", "evaluation_end", "base_cagr", "base_max_drawdown", "name_clip", "exposure_multiplier"}
    assert payload["mdd"] == pytest.approx(-0.05)
    assert payload["unit_reference"]["mdd"] == pytest.approx(-0.05)
    assert payload["reconciliation"]["mdd"] == pytest.approx(0.05)
    assert payload["reconciliation"]["mdd_convention"] == "magnitude"
    assert "error_type" not in json.dumps(payload)
    daily = pd.read_parquet(run_dir / "account_daily.parquet")
    assert list(daily.columns) == ["equity", "exposure"]
    rows = (tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()
    last = json.loads(rows[-1])
    assert last["kind"] == "mhs_frozen_account"
    assert last["base_max_drawdown"] == pytest.approx(0.05)


def test_account_unit_reference_replays_before_account(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default request replays unit reference first with fixed/1.0 then the growth account."""
    from src.strategy.sizing import account_growth_policy
    from src.core.params import ACCOUNT_DEFAULT_CAPITAL_USDT, ACCOUNT_TAKER_FEE_BPS, ACCOUNT_UNIT_REFERENCE_CAPITAL

    seen = _install_frozen_account_fakes(monkeypatch, tmp_path)
    run_frozen_account(_account_request(tmp_path))
    assert len(seen["replays"]) == 2
    unit_replay, main = seen["replays"]
    assert unit_replay["policy"].kind == "fixed"
    assert unit_replay["policy"].exposure_max == 1.0
    assert unit_replay["policy"].impact_y == 0.0
    assert unit_replay["capital"] == ACCOUNT_UNIT_REFERENCE_CAPITAL
    assert unit_replay["filters"] is False
    assert unit_replay["unit_equity"] is None
    assert main["capital"] == ACCOUNT_DEFAULT_CAPITAL_USDT
    assert main["policy"] == account_growth_policy(impact_y=main["policy"].impact_y)
    assert main["fee"] == ACCOUNT_TAKER_FEE_BPS
    assert main["filters"] is True
    pd.testing.assert_series_equal(main["unit_equity"], seen["equities"][0])


def test_account_candidate_is_unlevered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The account book is the registered unlevered clip unit book; both replays use its anchors."""
    from src.strategy.targets import FROZEN_MHS_TOP20_ACCOUNT_UNIT_V2
    from src.core.params import FROZEN_GROWTH_NAME_CLIP

    seen = _install_frozen_account_fakes(monkeypatch, tmp_path)
    run_frozen_account(_account_request(tmp_path))
    assert seen["request"].strategy is FROZEN_MHS_TOP20_ACCOUNT_UNIT_V2
    assert seen["request"].strategy.exposure_multiplier == 1.0
    assert seen["request"].strategy.name_clip == FROZEN_GROWTH_NAME_CLIP
    assert seen["request"].execution_bound == "OHLCV_IMMEDIATE_TAKER"
    for replay in seen["replays"]:
        pd.testing.assert_index_equal(replay["anchor_times"], seen["anchors"])


def test_account_maker_threads_identical_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Maker execution passes identical maker controls to both ledgers."""
    from src.core.params import ACCOUNT_MAKER_FEE_BPS, ACCOUNT_PASSIVE_WINDOW_BARS

    seen = _install_frozen_account_fakes(monkeypatch, tmp_path)
    run_frozen_account(_account_request(tmp_path, execution="maker"))
    assert len(seen["replays"]) == 2
    for replay in seen["replays"]:
        assert replay["execution"] == "maker"
        assert replay["maker_fee_bps"] == ACCOUNT_MAKER_FEE_BPS
        assert replay["passive_window_bars"] == ACCOUNT_PASSIVE_WINDOW_BARS


def test_account_default_execution_stays_taker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Taker replays carry no maker kwargs and the run dir has no maker infix."""
    seen = _install_frozen_account_fakes(monkeypatch, tmp_path)
    run_frozen_account(_account_request(tmp_path))
    for replay in seen["replays"]:
        assert replay["execution"] == "taker"
        assert "maker_fee_bps" not in replay
        assert "passive_window_bars" not in replay
    (run_dir,) = _run_dirs(tmp_path)
    assert "_maker_" not in run_dir.name


def test_account_fixed_moment_source_is_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixed policy discloses no moment source but still replays the unit reference."""
    seen = _install_frozen_account_fakes(monkeypatch, tmp_path)
    report = run_frozen_account(_account_request(tmp_path, policy="fixed", fixed_exposure=2.5))
    assert seen["replays"][0]["policy"].exposure_max == 1.0
    assert seen["replays"][1]["unit_equity"] is not None
    assert report.payload["moment_source"] == "none"


def test_account_execution_disclosed_in_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Maker mode is disclosed in account.json and the catalog row."""
    _install_frozen_account_fakes(monkeypatch, tmp_path)
    report = run_frozen_account(_account_request(tmp_path, execution="maker"))
    assert report.payload["execution"]["mode"] == "maker"
    assert report.payload["execution"]["maker_fill_fraction"] == pytest.approx(0.9)
    rows = (tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(rows[-1])["execution"] == "maker"


def test_account_headline_drawdown_is_intraday_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Headline drawdown is the 3m close path with opposite signs in payload and catalog."""
    _install_frozen_account_fakes(monkeypatch, tmp_path, account_intraday=0.2)
    report = run_frozen_account(_account_request(tmp_path))
    assert report.payload["mdd"] == pytest.approx(-0.2)
    assert report.payload["unit_reference"]["mdd"] == pytest.approx(-0.05)
    assert report.payload["reconciliation"]["mdd"] == pytest.approx(0.05)
    rows = (tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(rows[-1])["base_max_drawdown"] == pytest.approx(0.2)


def test_account_unit_reference_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed unit reference replay raises with the cause chained and no run dir."""
    from src.application.mhs_frozen_account import FrozenAccountError

    _install_frozen_account_fakes(monkeypatch, tmp_path, unit_fail=True)
    with pytest.raises(FrozenAccountError, match=r"unit reference"):
        run_frozen_account(_account_request(tmp_path))
    assert _run_dirs(tmp_path) == []


def test_account_unit_reference_liquidation_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A liquidated unit reference fails the run with no run dir."""
    from src.application.mhs_frozen_account import FrozenAccountError

    _install_frozen_account_fakes(monkeypatch, tmp_path, unit_liquidated=True)
    with pytest.raises(FrozenAccountError, match=r"unit reference liquidated at"):
        run_frozen_account(_account_request(tmp_path))
    assert _run_dirs(tmp_path) == []


def test_account_replay_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An account replay failure raises without persisting."""
    import src.engine.account_ledger as ledger_mod
    from src.application.mhs_frozen_account import FrozenAccountError
    from src.common.errors import DataIntegrityError

    _install_frozen_account_fakes(monkeypatch, tmp_path)

    def _boom(*args: object, **kwargs: object) -> object:
        raise DataIntegrityError("replay boom")

    monkeypatch.setattr(ledger_mod, "replay_account", _boom)
    with pytest.raises(FrozenAccountError, match=r"frozen account failed"):
        run_frozen_account(_account_request(tmp_path))
    assert _run_dirs(tmp_path) == []


def test_account_source_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A source assembly failure raises before any replay."""
    import src.engine.strategy_backtest as run_mod
    from src.application.mhs_frozen_account import FrozenAccountError
    from src.common.errors import DataIntegrityError

    seen = _install_frozen_account_fakes(monkeypatch, tmp_path)

    def _boom(request: object) -> tuple:
        raise DataIntegrityError("source boom")

    monkeypatch.setattr(run_mod, "build_frozen_request_candidate", _boom)
    with pytest.raises(FrozenAccountError, match=r"frozen account failed"):
        run_frozen_account(_account_request(tmp_path))
    assert "replays" not in seen


def test_account_missing_venue_snapshot_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty venue root fails before the candidate build."""
    import src.engine.strategy_backtest as run_mod
    from src.application.mhs_frozen_account import FrozenAccountError

    seen = _install_frozen_account_fakes(monkeypatch, tmp_path, stub_venue=False)
    called = []
    monkeypatch.setattr(run_mod, "build_frozen_request_candidate", lambda req: (called.append(req), (_ for _ in ()).throw(AssertionError("must not build"))))
    with pytest.raises(FrozenAccountError, match=r"data collect venue-rules"):
        run_frozen_account(_account_request(tmp_path))
    assert called == []


def test_account_invalid_request_rejected_before_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid controls raise ValueError without touching venue or candidate seams."""
    import src.market_data.binance.venue_rules as venue_mod
    import src.engine.strategy_backtest as run_mod

    _install_frozen_account_fakes(monkeypatch, tmp_path)
    monkeypatch.setattr(venue_mod, "latest_venue_rule_snapshot", lambda root: (_ for _ in ()).throw(AssertionError("no io")))
    monkeypatch.setattr(run_mod, "build_frozen_request_candidate", lambda req: (_ for _ in ()).throw(AssertionError("no build")))
    bad_variants = [
        {"source_start": pd.Timestamp("2025-01-01")},
        {"policy": "turbo"},
        {"execution": "limit"},
        {"policy": "fixed"},
        {"fixed_exposure": 0},
        {"capital": float("inf")},
    ]
    for overrides in bad_variants:
        with pytest.raises(ValueError, match=r".+"):
            run_frozen_account(_account_request(tmp_path, **overrides))


def test_account_run_directory_suffix_on_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A repeated timestamp resolves to a fresh suffixed run directory."""
    from src.application.mhs_frozen_account import _resolve_account_destination

    frozen = pd.Timestamp("2025-03-03 12:00:00", tz="UTC")
    monkeypatch.setattr(pd.Timestamp, "now", staticmethod(lambda tz=None: frozen))
    root = tmp_path / "runs"
    kwargs = {"runs_root": root, "start": pd.Timestamp("2025-01-01", tz="UTC"), "end": pd.Timestamp("2025-02-01", tz="UTC"), "policy": "growth", "capital": 2100.0}
    first = _resolve_account_destination(**kwargs)
    second = _resolve_account_destination(**kwargs)
    assert first.name.endswith("Z")
    assert second.name == f"{first.name}-2"


def test_account_maker_run_directory_suffixed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A maker run directory carries the maker infix after the capital."""
    _install_frozen_account_fakes(monkeypatch, tmp_path)
    run_frozen_account(_account_request(tmp_path, execution="maker"))
    (run_dir,) = _run_dirs(tmp_path)
    assert "_account_growth_2100_maker_" in run_dir.name


def test_account_catalog_append_confined_to_runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exactly one account row lands in the catalog derived from runs_root."""
    _install_frozen_account_fakes(monkeypatch, tmp_path)
    run_frozen_account(_account_request(tmp_path))
    rows = (tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["kind"] == "mhs_frozen_account"
    assert list(tmp_path.rglob("index.jsonl")) == [tmp_path / "index.jsonl"]


def test_account_unit_returns_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Export writes stamped parquet and leaves the payload unchanged."""
    import pyarrow.parquet as pq

    seen = _install_frozen_account_fakes(monkeypatch, tmp_path)
    dest = tmp_path / "nested" / "u.parquet"
    report = run_frozen_account(_account_request(tmp_path, export_unit_returns=dest))
    unit_equity = seen["equities"][0]
    table = pq.read_table(dest)
    metadata = table.schema.metadata
    assert metadata[b"strategy_id"] == b"frozen_mhs_top20_v2"
    assert metadata[b"execution"] == b"taker"
    assert metadata[b"evaluation_start"] == b"2025-01-01T00:00:00+00:00"
    assert metadata[b"evaluation_end"] == b"2025-02-01T00:00:00+00:00"
    assert metadata[b"run_dir"] == str(report.run_dir).encode()
    frame = table.to_pandas()
    expected = unit_equity.pct_change().iloc[1:]
    assert list(frame.columns) == ["unit_return"]
    assert frame.index.name == "entry_day"
    np.testing.assert_allclose(frame["unit_return"].to_numpy(), expected.to_numpy())
    assert (report.run_dir / "account.json").is_file()
    assert not dest.with_suffix(dest.suffix + ".tmp").exists()
    plain = run_frozen_account(_account_request(tmp_path, runs_root=tmp_path / "plain" / "runs")).payload
    exported = {key: value for key, value in report.payload.items() if key != "created_at"}
    unexported = {key: value for key, value in plain.items() if key != "created_at"}
    assert exported == unexported


def test_reconcile_missing_reference() -> None:
    """None reference yields a missing_reference record with null gaps."""
    rec = reconcile_unit_reference(None, unit_cagr=0.1, unit_mdd=0.05)
    assert rec["status"] == "missing_reference"
    assert rec["reference_canonical"] is None
    assert rec["cagr_gap"] is None
    assert rec["mdd_gap"] is None
    assert rec["fixed_exposure"] == 1.0


def test_reconcile_boundary_inclusive() -> None:
    """Gaps exactly at tolerance are ok; twice the CAGR tolerance is mismatch."""
    from src.core.params import ACCOUNT_RECON_CAGR_TOLERANCE

    ref = {"strategy_id": "s", "run_dir": "r", "evaluation_start": "a", "evaluation_end": "b", "base_cagr": 0.0, "base_max_drawdown": 0.0, "name_clip": 0.05, "exposure_multiplier": 1.0}
    ok = reconcile_unit_reference(ref, unit_cagr=ACCOUNT_RECON_CAGR_TOLERANCE, unit_mdd=0.0)
    assert ok["status"] == "ok"
    bad = reconcile_unit_reference(ref, unit_cagr=2 * ACCOUNT_RECON_CAGR_TOLERANCE, unit_mdd=0.0)
    assert bad["status"] == "mismatch"
    assert bad["cagr_gap"] == pytest.approx(2 * ACCOUNT_RECON_CAGR_TOLERANCE)


def test_reconcile_missing_base_metric_is_mismatch() -> None:
    """A None base CAGR yields mismatch with a null gap."""
    ref = {"strategy_id": "s", "run_dir": "r", "evaluation_start": "a", "evaluation_end": "b", "base_cagr": None, "base_max_drawdown": 0.05, "name_clip": 0.05, "exposure_multiplier": 1.0}
    rec = reconcile_unit_reference(ref, unit_cagr=0.1, unit_mdd=0.05)
    assert rec["status"] == "mismatch"
    assert rec["cagr_gap"] is None


def test_reconcile_drawdown_sign_agnostic() -> None:
    """Negative and positive base drawdowns give identical gaps."""
    base = {"strategy_id": "s", "run_dir": "r", "evaluation_start": "a", "evaluation_end": "b", "name_clip": 0.05, "exposure_multiplier": 1.0}
    pos = reconcile_unit_reference({**base, "base_cagr": 0.1, "base_max_drawdown": 0.05}, unit_cagr=0.1, unit_mdd=0.05)
    neg = reconcile_unit_reference({**base, "base_cagr": 0.1, "base_max_drawdown": -0.05}, unit_cagr=0.1, unit_mdd=0.05)
    assert pos["mdd_gap"] == neg["mdd_gap"] == pytest.approx(0.0)


def test_reconcile_pure_and_non_mutating() -> None:
    """Repeated calls agree and the input mapping is unchanged."""
    ref = {"strategy_id": "s", "run_dir": "r", "evaluation_start": "a", "evaluation_end": "b", "base_cagr": 0.1, "base_max_drawdown": 0.05, "name_clip": 0.05, "exposure_multiplier": 1.0}
    before = dict(ref)
    first = reconcile_unit_reference(ref, unit_cagr=0.1, unit_mdd=0.05)
    second = reconcile_unit_reference(ref, unit_cagr=0.1, unit_mdd=0.05)
    assert first == second
    assert ref == before


def test_reconcile_non_numeric_base_metric_raises() -> None:
    """Unconvertible base metrics raise typed errors."""
    base = {"strategy_id": "s", "run_dir": "r", "evaluation_start": "a", "evaluation_end": "b", "base_max_drawdown": 0.05, "name_clip": 0.05, "exposure_multiplier": 1.0}
    with pytest.raises(ValueError, match=r".*"):
        reconcile_unit_reference({**base, "base_cagr": "abc"}, unit_cagr=0.1, unit_mdd=0.05)
    with pytest.raises(TypeError):
        reconcile_unit_reference({**base, "base_cagr": [1]}, unit_cagr=0.1, unit_mdd=0.05)
    with pytest.raises(OverflowError):
        reconcile_unit_reference({**base, "base_cagr": 10**400}, unit_cagr=0.1, unit_mdd=0.05)


def test_account_reference_lookup_failure_is_disclosed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing catalog lookup is disclosed with error_type and the run persists."""
    import src.application.mhs_frozen_account as app_mod

    _install_frozen_account_fakes(monkeypatch, tmp_path)

    def _boom(*args: object, **kwargs: object) -> object:
        raise OSError("catalog boom")

    monkeypatch.setattr(app_mod, "_latest_same_book_reference", _boom)
    report = run_frozen_account(_account_request(tmp_path))
    assert report.payload["reconciliation"] == {"status": "failed", "error": "catalog boom", "error_type": "OSError"}
    assert (report.run_dir / "account_daily.parquet").exists()


def test_account_corrupt_catalog_line_disclosed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid JSON in the catalog discloses failed with JSONDecodeError."""
    _install_frozen_account_fakes(monkeypatch, tmp_path)
    (tmp_path / "index.jsonl").write_text("not json\n", encoding="utf-8")
    report = run_frozen_account(_account_request(tmp_path))
    assert report.payload["reconciliation"]["status"] == "failed"
    assert report.payload["reconciliation"]["error_type"] == "JSONDecodeError"
    assert (report.run_dir / "account.json").is_file()


def test_account_non_object_catalog_row_disclosed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-object catalog row discloses failed with ValueError."""
    _install_frozen_account_fakes(monkeypatch, tmp_path)
    (tmp_path / "index.jsonl").write_text("[1, 2]\n", encoding="utf-8")
    report = run_frozen_account(_account_request(tmp_path))
    assert report.payload["reconciliation"]["status"] == "failed"
    assert report.payload["reconciliation"]["error_type"] == "ValueError"


def test_account_non_numeric_reference_metric_disclosed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-numeric base metric discloses failed with ValueError."""
    _install_frozen_account_fakes(monkeypatch, tmp_path)
    index = _write_catalog_index(tmp_path, [])
    _write_same_book_reference(tmp_path, index, run_dir="runs/ref", base_cagr="abc", base_mdd=0.05)
    report = run_frozen_account(_account_request(tmp_path))
    assert report.payload["reconciliation"]["status"] == "failed"
    assert report.payload["reconciliation"]["error_type"] == "ValueError"


def test_account_unexpected_error_not_swallowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RuntimeError from the lookup propagates with no artifacts."""
    import src.application.mhs_frozen_account as app_mod

    _install_frozen_account_fakes(monkeypatch, tmp_path)
    monkeypatch.setattr(app_mod, "_latest_same_book_reference", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        run_frozen_account(_account_request(tmp_path))
    assert _run_dirs(tmp_path) == []
    assert not (tmp_path / "index.jsonl").exists()


def test_account_non_object_result_disclosed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A non-object result.json discloses failed while the run succeeds."""
    _install_frozen_account_fakes(monkeypatch, tmp_path)
    index = _write_catalog_index(tmp_path, [])
    _write_same_book_reference(tmp_path, index, run_dir="runs/good", base_cagr=(105000.0 / 100000.0) ** (365.0 / 3.0) - 1.0, base_mdd=0.05)
    bad_row = {"kind": "mhs_frozen", "strategy_id": "frozen_mhs_top20_v2", "run_dir": "runs/bad", "evaluation_start": "2025-01-01T00:00:00+00:00", "evaluation_end": "2025-02-01T00:00:00+00:00", "base_cagr": 0.1, "base_max_drawdown": 0.05}
    with index.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(bad_row, sort_keys=True) + "\n")
    bad_dir = tmp_path / "runs" / "bad"
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "result.json").write_text("[]", encoding="utf-8")
    with caplog.at_level("WARNING", logger="src.application.mhs_frozen_account"):
        report = run_frozen_account(_account_request(tmp_path))
    assert report.payload["reconciliation"] == {"status": "failed", "error": "reference result is not a JSON object", "error_type": "ValueError"}
    assert (report.run_dir / "account_daily.parquet").exists()
    assert "status=failed" in caplog.text


def test_account_success_schema_has_no_error_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ok/mismatch/missing_reference payloads never carry error_type."""
    _install_frozen_account_fakes(monkeypatch, tmp_path)
    report = run_frozen_account(_account_request(tmp_path))
    assert report.payload["reconciliation"]["status"] == "missing_reference"
    assert "error_type" not in json.dumps(report.payload)


def test_account_same_execution_reference_selected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Maker and taker runs each reconcile against their own execution canonical."""
    _install_frozen_account_fakes(monkeypatch, tmp_path)
    index = _write_catalog_index(tmp_path, [])
    unit_cagr = (105000.0 / 100000.0) ** (365.0 / 3.0) - 1.0
    _write_same_book_reference(tmp_path, index, run_dir="runs/taker", base_cagr=unit_cagr, base_mdd=0.05)
    _write_same_book_reference(tmp_path, index, run_dir="runs/maker", execution="maker", base_cagr=unit_cagr, base_mdd=0.05)
    maker = run_frozen_account(_account_request(tmp_path, execution="maker", runs_root=tmp_path / "m" / "runs"))
    assert maker.payload["reconciliation"]["reference_canonical"]["run_dir"] == "runs/maker"
    taker = run_frozen_account(_account_request(tmp_path))
    assert taker.payload["reconciliation"]["reference_canonical"]["run_dir"] == "runs/taker"


def test_account_missing_same_execution_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A maker run without a maker canonical warns with missing_reference."""
    _install_frozen_account_fakes(monkeypatch, tmp_path)
    index = _write_catalog_index(tmp_path, [])
    _write_same_book_reference(tmp_path, index, run_dir="runs/taker", base_cagr=(105000.0 / 100000.0) ** (365.0 / 3.0) - 1.0, base_mdd=0.05)
    with caplog.at_level("WARNING", logger="src.application.mhs_frozen_account"):
        report = run_frozen_account(_account_request(tmp_path, execution="maker"))
    assert report.payload["reconciliation"]["status"] == "missing_reference"
    assert "status=missing_reference" in caplog.text


def test_account_unclipped_primary_never_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unclipped primary row never reconciles the clipped book."""
    _install_frozen_account_fakes(monkeypatch, tmp_path)
    index = _write_catalog_index(tmp_path, [])
    _write_same_book_reference(tmp_path, index, run_dir="runs/unclipped", execution="maker", name_clip=None, exposure_multiplier=1.0, base_cagr=0.1, base_mdd=0.05)
    report = run_frozen_account(_account_request(tmp_path, execution="maker"))
    assert report.payload["reconciliation"]["status"] == "missing_reference"


def test_account_gap_beyond_tolerance_is_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A same-book gap beyond tolerance is a disclosed mismatch."""
    from src.core.params import ACCOUNT_RECON_CAGR_TOLERANCE

    _install_frozen_account_fakes(monkeypatch, tmp_path)
    index = _write_catalog_index(tmp_path, [])
    unit_cagr = (105000.0 / 100000.0) ** (365.0 / 3.0) - 1.0
    _write_same_book_reference(tmp_path, index, run_dir="runs/ref", base_cagr=unit_cagr + 2.0 * ACCOUNT_RECON_CAGR_TOLERANCE, base_mdd=0.05)
    with caplog.at_level("WARNING", logger="src.application.mhs_frozen_account"):
        report = run_frozen_account(_account_request(tmp_path))
    assert report.payload["reconciliation"]["status"] == "mismatch"
    assert report.payload["reconciliation"]["cagr_gap"] == pytest.approx(-2.0 * ACCOUNT_RECON_CAGR_TOLERANCE)
    assert "status=mismatch" in caplog.text


def test_account_window_or_execution_mismatch_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rows from another window never reconcile this run."""
    _install_frozen_account_fakes(monkeypatch, tmp_path)
    index = _write_catalog_index(tmp_path, [])
    unit_cagr = (105000.0 / 100000.0) ** (365.0 / 3.0) - 1.0
    _write_same_book_reference(tmp_path, index, run_dir="runs/other-window", execution="maker", base_cagr=unit_cagr, base_mdd=0.05, evaluation_start="2024-01-01T00:00:00+00:00", evaluation_end="2024-02-01T00:00:00+00:00")
    _write_same_book_reference(tmp_path, index, run_dir="runs/taker", base_cagr=unit_cagr, base_mdd=0.05)
    report = run_frozen_account(_account_request(tmp_path, execution="maker"))
    assert report.payload["reconciliation"]["status"] == "missing_reference"


def test_account_latest_matching_row_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two same-book rows reconcile against the later catalog entry."""
    _install_frozen_account_fakes(monkeypatch, tmp_path)
    index = _write_catalog_index(tmp_path, [])
    unit_cagr = (105000.0 / 100000.0) ** (365.0 / 3.0) - 1.0
    _write_same_book_reference(tmp_path, index, run_dir="runs/first", base_cagr=unit_cagr, base_mdd=0.05)
    _write_same_book_reference(tmp_path, index, run_dir="runs/second", base_cagr=unit_cagr, base_mdd=0.05)
    report = run_frozen_account(_account_request(tmp_path))
    assert report.payload["reconciliation"]["reference_canonical"]["run_dir"] == "runs/second"


def test_account_same_book_skips_unreadable_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Corrupt and mismatched rows never break reconciliation; the good row wins."""
    _install_frozen_account_fakes(monkeypatch, tmp_path)
    index = tmp_path / "index.jsonl"
    lines = [
        "",
        json.dumps({"kind": "mhs", "strategy_id": "x"}),
        json.dumps({"kind": "mhs_frozen", "strategy_id": "other_book"}),
        json.dumps({"kind": "mhs_frozen", "strategy_id": "frozen_mhs_top20_v2"}),
        json.dumps({"kind": "mhs_frozen", "strategy_id": "frozen_mhs_top20_v2", "run_dir": 123, "evaluation_start": "2025-01-01T00:00:00+00:00", "evaluation_end": "2025-02-01T00:00:00+00:00"}),
        json.dumps({"kind": "mhs_frozen", "strategy_id": "frozen_mhs_top20_v2", "run_dir": "runs/gone", "evaluation_start": "2025-01-01T00:00:00+00:00", "evaluation_end": "2025-02-01T00:00:00+00:00"}),
    ]
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    bad_json = tmp_path / "runs" / "broken"
    bad_json.mkdir(parents=True)
    (bad_json / "result.json").write_text("not json", encoding="utf-8")
    index.write_text(index.read_text(encoding="utf-8") + json.dumps({"kind": "mhs_frozen", "strategy_id": "frozen_mhs_top20_v2", "run_dir": "runs/broken", "evaluation_start": "2025-01-01T00:00:00+00:00", "evaluation_end": "2025-02-01T00:00:00+00:00"}) + "\n", encoding="utf-8")
    unit_cagr = (105000.0 / 100000.0) ** (365.0 / 3.0) - 1.0
    _write_same_book_reference(tmp_path, index, run_dir="runs/levered", base_cagr=unit_cagr, base_mdd=0.05, exposure_multiplier=2.5)
    _write_same_book_reference(tmp_path, index, run_dir="runs/good", base_cagr=unit_cagr, base_mdd=0.05)
    report = run_frozen_account(_account_request(tmp_path))
    assert report.payload["reconciliation"]["status"] == "ok"
    assert report.payload["reconciliation"]["reference_canonical"]["run_dir"] == "runs/good"


def test_frozen_exposure_unlevers_by_run_multiplier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The solver receives base returns and mean name weight divided by the run multiplier."""
    from src.core.params import COMMITTEE_GROWTH_HORIZON_YEARS, COMMITTEE_GROWTH_N_PATHS, FROZEN_EXPOSURE_MEAN_HAIRCUT, NULL_BOOTSTRAP_MEAN_BLOCK_DAYS

    run_dir = _fake_run_dir(tmp_path)
    seen = _install_exposure_fakes(monkeypatch)
    report = run_frozen_exposure(FrozenExposureRequest(run_dir=run_dir))
    np.testing.assert_allclose(seen["unit_returns"].to_numpy(), 0.001 / 2.5)
    assert seen["max_name_weight"] == pytest.approx(0.05 / 2.5)
    assert seen["solver_params"]["mean_haircut"] == FROZEN_EXPOSURE_MEAN_HAIRCUT
    assert seen["solver_params"]["n_paths"] == COMMITTEE_GROWTH_N_PATHS
    assert seen["solver_params"]["horizon_years"] == COMMITTEE_GROWTH_HORIZON_YEARS
    assert seen["solver_params"]["mean_block_days"] == NULL_BOOTSTRAP_MEAN_BLOCK_DAYS
    assert report.payload["execution_bound"] == "OHLCV_IMMEDIATE_TAKER"


def test_frozen_exposure_gap_roster_ignores_trading_exclusions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The gap roster is built with no trading-exclusion filter."""
    run_dir = _fake_run_dir(tmp_path)
    seen = _install_exposure_fakes(monkeypatch)
    run_frozen_exposure(FrozenExposureRequest(run_dir=run_dir))
    assert seen["blocked_decisions"] is None
    assert seen["breadth"] == 20
    assert seen["selection_mode"] == "causal_history"


def test_frozen_exposure_gap_population_restricted_to_delisted_symbols(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only DELISTED symbols feed the gap sampler."""
    import src.evaluation.exposure as growth_mod

    run_dir = _fake_run_dir(tmp_path)
    _install_exposure_fakes(monkeypatch)
    monkeypatch.setattr(growth_mod, "structurally_excluded_symbols", lambda: frozenset({"AAA"}))
    real_sample = growth_mod.roster_gap_sample
    captured: dict = {}

    def _spy(daily_close, roster, *, threshold):
        captured["columns"] = list(daily_close.columns)
        return real_sample(daily_close, roster, threshold=threshold)

    monkeypatch.setattr(growth_mod, "roster_gap_sample", _spy)
    report = run_frozen_exposure(FrozenExposureRequest(run_dir=run_dir))
    assert captured["columns"] == ["AAA"]
    assert report.payload["gap_symbols"] == ["AAA"]


def test_frozen_exposure_empty_exclusion_registry_skips_sampler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No registered exclusion yields a zero-event gap sample without calling the sampler."""
    import src.evaluation.exposure as growth_mod

    run_dir = _fake_run_dir(tmp_path)
    seen = _install_exposure_fakes(monkeypatch)
    monkeypatch.setattr(growth_mod, "roster_gap_sample", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")))
    report = run_frozen_exposure(FrozenExposureRequest(run_dir=run_dir))
    assert seen["gaps"].events_per_year == 0.0
    assert report.payload["gap_symbols"] == []


def test_exclusion_registry_read_at_most_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The exclusion registry is read once for a non-empty census and never for an empty one."""
    import src.evaluation.exposure as growth_mod
    import src.core.panel as panel_mod
    from src.application.mhs_frozen_account import FrozenAccountError

    run_dir = _fake_run_dir(tmp_path)
    _install_exposure_fakes(monkeypatch)
    calls = []
    real = growth_mod.structurally_excluded_symbols
    monkeypatch.setattr(growth_mod, "structurally_excluded_symbols", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    run_frozen_exposure(FrozenExposureRequest(run_dir=run_dir))
    assert len(calls) == 1

    empty_base = tmp_path / "frozen_empty"
    empty_base.mkdir(parents=True, exist_ok=True)
    (tmp_path / "frozen_run" / "result.json").replace(empty_base / "result.json")
    (tmp_path / "frozen_run" / "daily.parquet").replace(empty_base / "daily.parquet")
    calls.clear()
    monkeypatch.setattr(panel_mod, "load_base_panel", lambda *a, **k: {"close": pd.DataFrame(index=pd.date_range("2024-01-01", periods=40, freq="h", tz="UTC")), "quote_vol": pd.DataFrame(index=pd.date_range("2024-01-01", periods=40, freq="h", tz="UTC"))})
    monkeypatch.setattr(growth_mod, "structurally_excluded_symbols", lambda *a, **k: (calls.append(1), frozenset())[1])
    import contextlib

    with contextlib.suppress(FrozenAccountError):
        run_frozen_exposure(FrozenExposureRequest(run_dir=empty_base))
    assert calls == []


def test_derive_frozen_exposure_pure_and_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixed in-memory inputs give equal mappings without a created_at key."""
    _install_exposure_fakes(monkeypatch)
    import src.evaluation.exposure as growth_mod

    monkeypatch.setattr(growth_mod, "structurally_excluded_symbols", lambda: frozenset())
    artifacts = load_frozen_run_artifacts(_fake_run_dir(tmp_path))
    idx = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    close = pd.DataFrame({"AAA": 100.0, "BBB": 50.0}, index=idx, dtype="float64")
    volume = pd.DataFrame({"AAA": 1e6, "BBB": 1e6}, index=idx, dtype="float64")
    first = derive_frozen_exposure(artifacts, run_name="r", daily_close=close, daily_quote_volume=volume, census=("AAA", "BBB"), excluded_symbols=frozenset())
    second = derive_frozen_exposure(artifacts, run_name="r", daily_close=close, daily_quote_volume=volume, census=("AAA", "BBB"), excluded_symbols=frozenset())
    assert first == second
    assert "created_at" not in first


def test_frozen_exposure_fresh_only_before_any_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An existing exposure.json fails before any panel load."""
    import src.core.panel as panel_mod
    from src.application.mhs_frozen_account import FrozenAccountError

    run_dir = _fake_run_dir(tmp_path)
    (run_dir / "exposure.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(panel_mod, "load_base_panel", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no load")))
    with pytest.raises(FrozenAccountError, match=r"fresh"):
        run_frozen_exposure(FrozenExposureRequest(run_dir=run_dir))


def test_frozen_exposure_solver_rejection_leaves_no_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A solver rejection raises without persisting exposure.json."""
    import src.evaluation.exposure as growth_mod
    from src.application.mhs_frozen_account import FrozenAccountError

    run_dir = _fake_run_dir(tmp_path)
    _install_exposure_fakes(monkeypatch)
    monkeypatch.setattr(growth_mod, "solve_log_growth_exposure", lambda *a, **k: (_ for _ in ()).throw(ValueError("no rung")))
    with pytest.raises(FrozenAccountError, match=r"frozen exposure failed"):
        run_frozen_exposure(FrozenExposureRequest(run_dir=run_dir))
    assert not (run_dir / "exposure.json").exists()


def test_frozen_exposure_invalid_artifacts_rejected(tmp_path: Path) -> None:
    """An empty run dir raises an invalid-artifacts error."""
    from src.application.mhs_frozen_account import FrozenAccountError

    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    with pytest.raises(FrozenAccountError, match=r"invalid frozen run artifacts"):
        run_frozen_exposure(FrozenExposureRequest(run_dir=run_dir))


def test_exposure_payload_golden_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exposure output carries the exact key set with no tmp file left behind."""
    run_dir = _fake_run_dir(tmp_path)
    _install_exposure_fakes(monkeypatch)
    report = run_frozen_exposure(FrozenExposureRequest(run_dir=run_dir))
    assert set(report.payload) == {"run_dir", "strategy_id", "execution_bound", "exposure_multiplier", "grid", "growth", "ruin_probability", "argmax", "chosen", "gap_events_per_year", "gap_sample_size", "gap_symbols", "mean_haircut", "plateau_tolerance", "seed", "created_at", "unlever_assumption"}
    assert not (run_dir / "exposure.json.tmp").exists()


_PRE_REFACTOR = json.loads(
    (Path(__file__).resolve().parents[2] / "fixtures" / "frozen_account" / "pre_refactor_payloads.json").read_text(encoding="utf-8")
)
_FROZEN_NOW = pd.Timestamp("2026-03-04T05:06:07", tz="UTC")
_UNIT_CAGR = (105000.0 / 100000.0) ** (365.0 / 3.0) - 1.0
_GOLDEN_ACCOUNT_CASES = {
    "missing_reference": (None, {}),
    "ok": ((_UNIT_CAGR, 0.05), {}),
    "mismatch": ((_UNIT_CAGR + 1.0, 0.5), {}),
    "maker_missing": (None, {"execution": "maker"}),
    "fixed_ok": ((_UNIT_CAGR, 0.05), {"policy": "fixed", "fixed_exposure": 2.5}),
}


@pytest.mark.parametrize("case", sorted(_GOLDEN_ACCOUNT_CASES))
def test_account_payload_matches_pre_refactor_capture(case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Success-path account.json is byte-identical to the pre-refactor CLI output (created_at aside)."""
    reference, overrides = _GOLDEN_ACCOUNT_CASES[case]
    _install_frozen_account_fakes(monkeypatch, tmp_path, account_intraday=0.2)
    monkeypatch.setattr(pd.Timestamp, "now", staticmethod(lambda tz=None: _FROZEN_NOW))
    if reference is not None:
        _write_same_book_reference(tmp_path, tmp_path / "index.jsonl", run_dir="runs/old", base_cagr=reference[0], base_mdd=reference[1])
    report = run_frozen_account(_account_request(tmp_path, **overrides))
    text = (report.run_dir / "account.json").read_text(encoding="utf-8")
    text = text.replace(f'\n  "created_at": "{_FROZEN_NOW.isoformat()}",', "").replace(str(tmp_path), "<TMP>")
    assert text == _PRE_REFACTOR["account"][case]
    assert "error_type" not in text


def test_exposure_payload_matches_pre_refactor_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """exposure.json is byte-identical to the pre-refactor CLI output (created_at aside)."""
    run_dir = _fake_run_dir(tmp_path)
    _install_exposure_fakes(monkeypatch)
    monkeypatch.setattr(pd.Timestamp, "now", staticmethod(lambda tz=None: _FROZEN_NOW))
    report = run_frozen_exposure(FrozenExposureRequest(run_dir=run_dir))
    text = report.path.read_text(encoding="utf-8").replace(f'"created_at": "{_FROZEN_NOW.isoformat()}", ', "")
    assert text == _PRE_REFACTOR["exposure"]


def test_account_headline_failure_leaves_no_run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Headlines are derived before the run directory exists, so a degenerate ledger leaves nothing behind."""
    import src.application.mhs_frozen_account as app_mod

    _install_frozen_account_fakes(monkeypatch, tmp_path)
    real = app_mod._account_headlines

    def _fail_account(equity: pd.Series, capital: float) -> tuple[float, float, float]:
        if capital != 100000.0:
            raise IndexError("empty account ledger")
        return real(equity, capital)

    monkeypatch.setattr(app_mod, "_account_headlines", _fail_account)
    with pytest.raises(IndexError, match=r"empty account ledger"):
        run_frozen_account(_account_request(tmp_path))
    assert _run_dirs(tmp_path) == []
    assert not (tmp_path / "index.jsonl").exists()


def test_account_run_dir_creation_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unusable runs_root surfaces as FrozenAccountError, never a raw OSError traceback."""
    from src.application.mhs_frozen_account import FrozenAccountError

    _install_frozen_account_fakes(monkeypatch, tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    with pytest.raises(FrozenAccountError, match=r"^frozen account failed: ") as raised:
        run_frozen_account(_account_request(tmp_path, runs_root=blocker / "runs"))
    assert isinstance(raised.value.__cause__, OSError)
    assert not (tmp_path / "index.jsonl").exists()


def test_validate_rejects_non_request_and_bad_roots(tmp_path: Path) -> None:
    """Non-request, unordered and non-Path inputs raise before any I/O."""
    from src.application.mhs_frozen_account import validate_frozen_account_request

    with pytest.raises(ValueError, match=r"FrozenAccountRequest"):
        validate_frozen_account_request("nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"source_start < evaluation"):
        run_frozen_account(_account_request(tmp_path, source_start=pd.Timestamp("2025-03-01", tz="UTC")))
    with pytest.raises(ValueError, match=r"runs_root must be a Path"):
        run_frozen_account(_account_request(tmp_path, runs_root="x"))  # type: ignore[arg-type]


def test_invalid_frozen_request_construction_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad cost pair fails the frozen request build without any replay."""
    import src.application.mhs_frozen_account as app_mod
    from src.application.mhs_frozen_account import FrozenAccountError

    seen = _install_frozen_account_fakes(monkeypatch, tmp_path)
    monkeypatch.setattr(app_mod, "frozen_execution_specs", lambda: (None, None))
    with pytest.raises(FrozenAccountError, match=r"invalid frozen account request"):
        run_frozen_account(_account_request(tmp_path))
    assert "replays" not in seen


def test_account_second_replay_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the account replay failing still leaves no run dir."""
    import src.engine.account_ledger as ledger_mod
    from src.application.mhs_frozen_account import FrozenAccountError
    from src.common.errors import DataIntegrityError

    seen = _install_frozen_account_fakes(monkeypatch, tmp_path)
    real = ledger_mod.replay_account

    def _fail_second(*args: object, **kwargs: object) -> object:
        if len(seen.get("replays", [])) == 0:
            return real(*args, **kwargs)
        raise DataIntegrityError("account boom")

    monkeypatch.setattr(ledger_mod, "replay_account", _fail_second)
    with pytest.raises(FrozenAccountError, match=r"frozen account failed: account boom"):
        run_frozen_account(_account_request(tmp_path))
    assert _run_dirs(tmp_path) == []


def test_account_persistence_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An artifact write failure surfaces as a failed run."""
    from src.application.mhs_frozen_account import FrozenAccountError

    _install_frozen_account_fakes(monkeypatch, tmp_path)
    monkeypatch.setattr(Path, "write_text", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(FrozenAccountError, match=r"frozen account failed: disk full"):
        run_frozen_account(_account_request(tmp_path))


def test_derive_roster_integrity_error_is_value_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A roster DataIntegrityError surfaces as ValueError from the derivation."""
    import src.strategy.universe as universe_mod
    from src.common.errors import DataIntegrityError

    _install_exposure_fakes(monkeypatch)
    monkeypatch.setattr(universe_mod, "build_frozen_pit_roster", lambda *a, **k: (_ for _ in ()).throw(DataIntegrityError("roster boom")))
    artifacts = load_frozen_run_artifacts(_fake_run_dir(tmp_path))
    idx = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    close = pd.DataFrame({"AAA": 100.0}, index=idx, dtype="float64")
    volume = pd.DataFrame({"AAA": 1e6}, index=idx, dtype="float64")
    with pytest.raises(ValueError, match=r"roster boom"):
        derive_frozen_exposure(artifacts, run_name="r", daily_close=close, daily_quote_volume=volume, census=("AAA",), excluded_symbols=frozenset())


def test_derive_unexpected_solver_error_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-ValueError solver failure propagates unwrapped."""
    import src.evaluation.exposure as growth_mod

    _install_exposure_fakes(monkeypatch)
    monkeypatch.setattr(growth_mod, "solve_log_growth_exposure", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("solver bug")))
    artifacts = load_frozen_run_artifacts(_fake_run_dir(tmp_path))
    idx = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    close = pd.DataFrame({"AAA": 100.0}, index=idx, dtype="float64")
    volume = pd.DataFrame({"AAA": 1e6}, index=idx, dtype="float64")
    with pytest.raises(RuntimeError, match=r"solver bug"):
        derive_frozen_exposure(artifacts, run_name="r", daily_close=close, daily_quote_volume=volume, census=("AAA",), excluded_symbols=frozenset())
