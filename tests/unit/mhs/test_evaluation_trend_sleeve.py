"""MHS evaluation contract tests (split by behavioral domain; shared builders live in the original module)."""

"""Contract coverage for the MHS application evaluation resource telemetry."""
import numpy as np
import pandas as pd
import pytest
from src.mhs import evaluation as ev
from src.mhs.diagnostic_run import run_mhs_horizon_diagnostic
import src.mhs.evaluation.folds as folds_mod
import src.mhs.statistics as statistics
from src.mhs.evaluation import (
    MhsDiagnosticRequest,
)
from src.quant.universe.pit_universe import symbol_partition

from tests.unit.mhs.test_evaluation_appresearch import (  # noqa: F401
    _FOLD,
    _START,
)

def test_trend_sleeve_request_validation() -> None:
    # SCENARIO_MHS_REQUEST_TREND_SLEEVE_VALIDATION: MhsDiagnosticRequest gains
    # trend_sleeve (bool, default False) and trend_sleeve_gross (float, default
    # 0.0). A positive gross without the opt-in, or a gross outside [0.0, 1.0],
    # raises ValueError (fail closed -- no silent no-op); the default
    # construction leaves both at their off values.
    default = MhsDiagnosticRequest()
    assert default.trend_sleeve is False
    assert default.trend_sleeve_gross == 0.0
    with pytest.raises(ValueError, match="trend_sleeve_gross"):
        MhsDiagnosticRequest(trend_sleeve_gross=0.3)
    with pytest.raises(ValueError, match="trend_sleeve_gross"):
        MhsDiagnosticRequest(trend_sleeve=True, trend_sleeve_gross=-0.1)
    with pytest.raises(ValueError, match="trend_sleeve_gross"):
        MhsDiagnosticRequest(trend_sleeve=True, trend_sleeve_gross=1.5)
    with pytest.raises(ValueError, match="trend_sleeve"):
        MhsDiagnosticRequest(trend_sleeve="yes")
    on = MhsDiagnosticRequest(trend_sleeve=True, trend_sleeve_gross=0.3)
    assert on.trend_sleeve is True
    assert on.trend_sleeve_gross == 0.3

@pytest.mark.slow
def test_trend_sleeve_default_off_bit_identical(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_TREND_SLEEVE_DEFAULT_OFF_BIT_IDENTICAL: with the flags
    # omitted (trend_sleeve=False, trend_sleeve_gross=0.0) the report's
    # trend_sleeve_diagnostic is None and every pre-existing field is
    # bit-identical to the explicit-off baseline -- the sleeve is inert unless
    # explicitly enabled, so a default run cannot change any existing output.
    root, end = mhs_market
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    base = {
        "start": str(_START), "end": str(end), "data_root": str(root),
        "mark_mode": "cache_required", "execution_timeframe": "1m", "log_run": False,
        "execution_universe_size": 8,
    }
    default_report = run_mhs_horizon_diagnostic(MhsDiagnosticRequest(**base))
    explicit_off = run_mhs_horizon_diagnostic(
        MhsDiagnosticRequest(**base, trend_sleeve=False, trend_sleeve_gross=0.0),
    )
    assert default_report.status == "COMPLETE"
    assert default_report.trend_sleeve_diagnostic is None
    assert explicit_off.trend_sleeve_diagnostic is None
    for field in ("books", "blend", "blend_target_gross", "research_go", "folds"):
        assert getattr(default_report, field) == getattr(explicit_off, field)

@pytest.mark.slow
def test_trend_sleeve_diagnostic_populated(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_TREND_SLEEVE_DIAGNOSTIC_POPULATED: with trend_sleeve=True
    # and trend_sleeve_gross=0.3 the report's trend_sleeve_diagnostic is a dict
    # carrying the sleeve's standalone net Sharpe per measured cost tier, its
    # per-calendar-year net_t, its correlation to the slow_momentum book's pnl,
    # and the combined metrics; every value is finite or an explicit None,
    # never NaN silently coerced to 0.0.
    root, end = mhs_market
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, trend_sleeve=True, trend_sleeve_gross=0.3,
    )
    report = run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    diag = report.trend_sleeve_diagnostic
    assert isinstance(diag, dict)
    assert set(diag["net_sharpe_per_tier"]) == set(ev.MEASURED_EXECUTION_COST_TIERS_BPS)
    for value in diag["net_sharpe_per_tier"].values():
        assert value is None or np.isfinite(value)
    yearly = diag["yearly_net_t"]
    assert isinstance(yearly, dict)
    assert set(yearly) == {2021, 2022, 2023, 2024, 2025}
    for value in yearly.values():
        assert value is None or np.isfinite(value)
    corr = diag["slow_momentum_pnl_corr"]
    assert corr is None or np.isfinite(corr)
    combined = diag["combined"]
    assert set(combined["net_sharpe_per_tier"]) == set(ev.MEASURED_EXECUTION_COST_TIERS_BPS)
    for value in combined["net_sharpe_per_tier"].values():
        assert value is None or np.isfinite(value)
    worst = combined["worst_year_net_t"]
    assert worst is None or np.isfinite(worst)

def test_trend_sleeve_position_wraps_frozen_math() -> None:
    # SCENARIO_MHS_TREND_SLEEVE_OVERLAY_ADDITIVE (helper): _trend_sleeve_position
    # is a thin wrapper reusing market_basket_log_price + time_series_trend_position.
    grid = pd.date_range("2021-01-01", periods=144, freq="1h", tz="UTC")
    symbols = ["S1", "S2", "S3"]
    rng = np.random.default_rng(11)
    log_close = pd.DataFrame(
        np.cumsum(rng.normal(0.0, 0.01, (len(grid), len(symbols))), axis=0),
        index=grid, columns=symbols,
    )
    eligible = pd.DataFrame(True, index=grid, columns=symbols)
    decision_grid = pd.date_range(grid[0], grid[-1], freq="24h", tz="UTC")
    expected = ev.time_series_trend_position(
        ev.market_basket_log_price(log_close, eligible),
        ev.TREND_SLEEVE_HORIZONS_HOURS, decision_grid,
    )
    got = ev._trend_sleeve_position(log_close, eligible, decision_grid)
    pd.testing.assert_series_equal(got, expected)

def test_apply_trend_sleeve_is_additive_and_pure() -> None:
    # SCENARIO_MHS_TREND_SLEEVE_OVERLAY_ADDITIVE (helper): _apply_trend_sleeve
    # returns blend + sleeve elementwise without mutating the input frame.
    grid = pd.date_range("2021-01-01", periods=48, freq="1h", tz="UTC")
    symbols = ["A", "B", "C"]
    blend_1h = pd.DataFrame(0.1, index=grid, columns=symbols)
    position = pd.Series(0.5, index=grid)
    mask = pd.DataFrame(True, index=grid, columns=symbols)
    expected = blend_1h.add(
        ev.trend_sleeve_weights(position, mask, 0.3).reindex(blend_1h.index).fillna(0.0),
        fill_value=0.0,
    )
    out = ev._apply_trend_sleeve(blend_1h, position, mask, 0.3)
    pd.testing.assert_frame_equal(out, expected)
    assert out is not blend_1h
    pd.testing.assert_frame_equal(
        blend_1h, pd.DataFrame(0.1, index=grid, columns=symbols),
    )

def test_trend_sleeve_overlay_off_byte_identical(mhs_market_with_taker_buy_quote, monkeypatch) -> None:
    # SCENARIO_MHS_TREND_SLEEVE_OVERLAY_OFF_BYTE_IDENTICAL: with the overlay
    # disabled (trend_sleeve=False, or trend_sleeve_gross=0.0) neither sleeve
    # helper is ever called and the fold targets are byte-identical to the
    # pre-change baseline.
    root, end = mhs_market_with_taker_buy_quote
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    base = {
        "start": str(_START), "end": str(end), "data_root": str(root),
        "mark_mode": "cache_required", "execution_timeframe": "1m", "log_run": False,
        "execution_universe_size": 8, "committee_capital": True,
    }
    request = MhsDiagnosticRequest(**base)

    def _must_not_be_called(*_args, **_kwargs):
        raise AssertionError("sleeve machinery must not run when the overlay is off")

    monkeypatch.setattr(ev, "_trend_sleeve_position", _must_not_be_called)
    monkeypatch.setattr(folds_mod, "_trend_sleeve_position", _must_not_be_called)
    monkeypatch.setattr(ev, "_apply_trend_sleeve", _must_not_be_called)
    monkeypatch.setattr(folds_mod, "_apply_trend_sleeve", _must_not_be_called)
    target_patched, _sig, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request, funding_by_symbol,
    )

    monkeypatch.undo()
    target_baseline, _sig, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request, funding_by_symbol,
    )
    pd.testing.assert_frame_equal(target_patched, target_baseline)

    target_gross0, _sig, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD,
        MhsDiagnosticRequest(**base, trend_sleeve=True, trend_sleeve_gross=0.0),
        funding_by_symbol,
    )
    pd.testing.assert_frame_equal(target_gross0, target_baseline)

def test_trend_sleeve_overlay_additive_fold(mhs_market_with_taker_buy_quote, monkeypatch) -> None:
    # SCENARIO_MHS_TREND_SLEEVE_OVERLAY_ADDITIVE (fold path): the executed fold
    # blend_1h is the pre-change committee blend plus trend_sleeve_weights at the
    # configured gross, elementwise -- and the sleeve breaks dollar-neutrality.
    root, end = mhs_market_with_taker_buy_quote
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, committee_capital=True,
        trend_sleeve=True, trend_sleeve_gross=0.3,
    )
    real = ev._apply_trend_sleeve
    seen = {"called": False}

    def _spy(blend_1h, position, execution_mask, gross_budget):
        seen["called"] = True
        seen["position"] = position
        sleeve = ev.trend_sleeve_weights(position, execution_mask, gross_budget)
        expected = blend_1h.add(sleeve.reindex(blend_1h.index).fillna(0.0), fill_value=0.0)
        out = real(blend_1h, position, execution_mask, gross_budget)
        pd.testing.assert_frame_equal(out, expected)
        # The pre-sleeve committee blend is dollar-neutral; the additive sleeve
        # deliberately is not -- row sums may be nonzero afterwards.
        assert float(blend_1h.sum(axis=1).abs().max()) < 1e-6
        return out

    monkeypatch.setattr(ev, "_apply_trend_sleeve", _spy)
    monkeypatch.setattr(folds_mod, "_apply_trend_sleeve", _spy)
    target_on, _sig, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request, funding_by_symbol,
    )
    assert seen["called"]
    assert seen["position"] is not None
    assert float(seen["position"].abs().max()) > 0.0
    assert np.isfinite(target_on.to_numpy(dtype="float64")).all()

@pytest.mark.slow
def test_trend_sleeve_overlay_additive_toplevel(mhs_market_with_taker_buy_quote, monkeypatch) -> None:
    # SCENARIO_MHS_TREND_SLEEVE_OVERLAY_ADDITIVE (top-level path): the sleeve is
    # applied exactly once, the top-level blend_1h is the pre-change blend plus
    # the gross-budget sleeve, and -- with committee_capital -- the executed
    # committee book passed to the replay carries the same overlay.
    root, end = mhs_market_with_taker_buy_quote
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, committee_capital=True,
        trend_sleeve=True, trend_sleeve_gross=0.3,
    )
    real = ev._apply_trend_sleeve
    spy_out = {}

    def _spy(blend_1h, position, execution_mask, gross_budget):
        sleeve = ev.trend_sleeve_weights(position, execution_mask, gross_budget)
        expected = blend_1h.add(sleeve.reindex(blend_1h.index).fillna(0.0), fill_value=0.0)
        out = real(blend_1h, position, execution_mask, gross_budget)
        pd.testing.assert_frame_equal(out, expected)
        spy_out["out"] = out
        spy_out["position"] = position
        return out

    monkeypatch.setattr(ev, "_apply_trend_sleeve", _spy)
    monkeypatch.setattr(folds_mod, "_apply_trend_sleeve", _spy)
    captured = {}

    def _fake_books(*args, **kwargs):
        captured["blend_1h"] = args[20]
        captured["committee_execution_book"] = kwargs.get("committee_execution_book")
        return (None, None, None, {}, None)

    monkeypatch.setattr(ev, "_run_books_concurrent", _fake_books)
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    report = run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    assert "out" in spy_out
    assert float(spy_out["position"].abs().max()) > 0.0
    assert captured["committee_execution_book"] is not None
    pd.testing.assert_frame_equal(captured["committee_execution_book"], spy_out["out"])
    assert not captured["committee_execution_book"].equals(captured["blend_1h"])

def test_trend_sleeve_overlay_roster_no_starvation(mhs_market_with_taker_buy_quote, monkeypatch) -> None:
    # SCENARIO_MHS_TREND_SLEEVE_ROSTER_NO_STARVATION: every symbol carrying a
    # nonzero post-overlay target is a subset of the execution-mask-eligible
    # roster, so minute_roster/execution_symbols derivation downstream never
    # picks up a symbol the committee book would not already have traded.
    root, end = mhs_market_with_taker_buy_quote
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, committee_capital=True,
        trend_sleeve=True, trend_sleeve_gross=0.3,
    )
    real = ev._apply_trend_sleeve
    seen = {}

    def _spy(blend_1h, position, execution_mask, gross_budget):
        seen["execution_mask"] = execution_mask
        return real(blend_1h, position, execution_mask, gross_budget)

    monkeypatch.setattr(ev, "_apply_trend_sleeve", _spy)
    monkeypatch.setattr(folds_mod, "_apply_trend_sleeve", _spy)
    target_on, _sig, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request, funding_by_symbol,
    )
    assert "execution_mask" in seen
    mask_1h = seen["execution_mask"]
    shared_idx = target_on.index.intersection(mask_1h.index)
    mask_at = mask_1h.reindex(shared_idx).fillna(False)
    leak = target_on.reindex(shared_idx).where(~mask_at)
    assert float(np.nansum(leak.to_numpy())) == 0.0
    assert set(_roster) <= set(target_on.columns)

def test_trend_sleeve_fold_memory_order(mhs_market_with_taker_buy_quote, monkeypatch) -> None:
    # SCENARIO_MHS_TREND_SLEEVE_FOLD_MEMORY_ORDER: trend_position is computed
    # strictly before the existing `del eligible` -- the wrapper must receive the
    # live eligible frame on the 24h slow_grid -- so eligible is released at the
    # same point as before and only the tiny trend Series survives.
    root, end = mhs_market_with_taker_buy_quote
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, committee_capital=True,
        trend_sleeve=True, trend_sleeve_gross=0.3,
    )
    real = ev._trend_sleeve_position
    seen = {}

    def _spy(log_close, eligible, decision_grid):
        seen["eligible_shape"] = eligible.shape
        seen["decision_grid"] = decision_grid
        return real(log_close, eligible, decision_grid)

    monkeypatch.setattr(ev, "_trend_sleeve_position", _spy)
    monkeypatch.setattr(folds_mod, "_trend_sleeve_position", _spy)
    _target, _sig, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request, funding_by_symbol,
    )
    assert "eligible_shape" in seen
    assert seen["eligible_shape"][0] == len(_grid)
    assert seen["eligible_shape"][1] >= 8
    steps = pd.Series(seen["decision_grid"]).diff().dropna()
    assert (steps == pd.Timedelta(hours=24)).all()

def test_trend_sleeve_diagnostic_uses_deployed_book(mhs_market) -> None:
    # SCENARIO_MHS_TREND_SLEEVE_DIAGNOSTIC_USES_DEPLOYED_BOOK: combined metrics
    # and the correlation are measured against the caller's current_book -- pass
    # a synthetic book identical to the sleeve itself, which forces the reported
    # correlation to exactly 1.0, a value the rebuilt frozen slow_momentum book
    # could only produce by coincidence.
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    funded = [
        s for s in symbols
        if s in funding_by_symbol and s not in ev.SOURCE_GAP_EXCLUDED_SYMBOLS
    ]
    panel = ev.load_base_panel(
        str(root), "1h", ("close", "open", "quote_vol"), _START, end,
        partition="dev", min_bars=2000,
    )
    close, opens, quote_vol = panel["close"][funded], panel["open"][funded], panel["quote_vol"][funded]
    grid_1h = close.index
    bar_period = grid_1h[1] - grid_1h[0]
    funding_window = {
        s: funding_by_symbol[s].loc[
            (funding_by_symbol[s].index >= grid_1h[0])
            & (funding_by_symbol[s].index < grid_1h[-1] + bar_period)
        ]
        for s in funded
    }
    bar_funding = ev.bar_funding_panel(funding_window, grid_1h)
    eligible = ev.liquid_half_eligibility(quote_vol, lookback_bars=720, min_history_bars=720)
    log_close = np.log(close)
    execution_mask = ev._pit_execution_mask(quote_vol, eligible, 8)
    request = MhsDiagnosticRequest(trend_sleeve=True, trend_sleeve_gross=0.3)

    decision_grid = pd.date_range(grid_1h[0], grid_1h[-1], freq="24h", tz="UTC")
    basket = ev.market_basket_log_price(log_close, eligible)
    position = ev.time_series_trend_position(
        basket, ev.TREND_SLEEVE_HORIZONS_HOURS, decision_grid,
    )
    sleeve = ev.trend_sleeve_weights(position, execution_mask, request.trend_sleeve_gross)

    diag = ev._trend_sleeve_diagnostic(
        log_close, eligible, opens, bar_funding, execution_mask, sleeve.copy(), request,
    )
    assert diag["slow_momentum_pnl_corr"] == pytest.approx(1.0)
    combined_net, _ = ev.mhs_ledger_pnl(
        sleeve.add(sleeve), opens, bar_funding,
        ev.MEASURED_EXECUTION_COST_TIERS_BPS["base"],
    )
    expected_combined_sharpe = statistics._annualized_1h_sharpe(combined_net)
    assert expected_combined_sharpe is not None
    assert diag["combined"]["net_sharpe_per_tier"]["base"] == pytest.approx(
        expected_combined_sharpe,
    )

def test_trend_sleeve_gross_budget_bounds() -> None:
    # SCENARIO_MHS_TREND_SLEEVE_GROSS_BUDGET_BOUNDS: the existing __post_init__
    # validation is unchanged -- gross in [0.0, 1.0] inclusive is accepted and a
    # positive gross without the opt-in (or out of bounds) fails closed.
    assert MhsDiagnosticRequest(trend_sleeve=True, trend_sleeve_gross=1.0).trend_sleeve_gross == 1.0
    assert MhsDiagnosticRequest(trend_sleeve=True, trend_sleeve_gross=0.0).trend_sleeve_gross == 0.0
    with pytest.raises(ValueError, match="trend_sleeve_gross"):
        MhsDiagnosticRequest(trend_sleeve=True, trend_sleeve_gross=1.0001)
    with pytest.raises(ValueError, match="trend_sleeve_gross"):
        MhsDiagnosticRequest(trend_sleeve=True, trend_sleeve_gross=-1e-9)
    with pytest.raises(ValueError, match="trend_sleeve_gross"):
        MhsDiagnosticRequest(trend_sleeve_gross=0.3)