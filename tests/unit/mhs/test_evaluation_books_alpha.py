"""MHS evaluation pipeline/gate tests (third-level split)."""

"""MHS evaluation pipeline/gate tests (second-level split remainder)."""
"""MHS evaluation core contract tests (everything not in a domain-specific split file)."""
"""Contract coverage for the MHS application evaluation resource telemetry."""
import dataclasses
import numpy as np
import pandas as pd
import pytest
from src.mhs import evaluation as ev
import src.mhs.evaluation.diagnostics as diagnostics_mod
import src.mhs.books as books_mhs
from src.mhs.diagnostic_run import run_mhs_horizon_diagnostic
import src.mhs.scaling as scaling
from src.mhs.evaluation import (
    MhsDiagnosticRequest,
)
from src.mhs.horizons import vol_normalized_horizon_signal
from src.quant.universe.pit_universe import symbol_partition
from tests.unit.mhs.test_evaluation_appresearch import (  # noqa: F401
    _FOLD,
    _START,
    _assert_books_equal,
    _assert_regime_vol_mean_roster_masked,
    _build_book_outcome_args,
    _build_books_concurrent_args,
    _build_compact_report,
    _deployment_readiness,
    _dispatch_spec,
    _gap_mixed_replay,
    _passing_fold_report,
    _perf_opt_placebo_inputs,
    _pre_change_slow_book,
    _reference_bootstrap_ci,
    _reference_participation_warnings,
    _reference_placebo_percentile,
    _reference_resolve_ns_scalar,
    _reference_weights,
    _roster_mask_panel_inputs,
    _sequential_book_reports,
    _signal_disagreement_panel,
    _slow_book_panel_inputs,
    _synthetic_ledger,
    _write_3m_cache,
    _write_quote_volume_market,
)

def test_book_weights_momentum_keeps_raw_signal() -> None:
    """Verify book_weights keeps raw log return for momentum books."""
    log_close, eligible, _, _, idx = _signal_disagreement_panel()
    spec = _dispatch_spec(sign=1)
    weights = ev._book_weights(log_close, eligible, spec, idx)
    expected = _reference_weights(log_close, eligible, idx, spec)
    pd.testing.assert_frame_equal(weights, expected)
    vol_normalized = ev.phase_tranche_book(
        ev.rank_weight_book(
            vol_normalized_horizon_signal(log_close, spec.horizon_hours).reindex(idx),
            eligible.reindex(idx),
            spec.band.sign,
            spec.min_symbols,
        ),
        spec.tranche_count(),
    )
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(weights, vol_normalized)

def test_book_weights_reversal_keeps_raw_signal() -> None:
    """SCENARIO_BOOK_WEIGHTS_REVERSAL_UNCHANGED: ``_book_weights`` for a
    sign=-1 spec stays on raw ``horizon_log_return``."""
    log_close, eligible, _, _, idx = _signal_disagreement_panel()
    spec = _dispatch_spec(sign=-1)
    weights = ev._book_weights(log_close, eligible, spec, idx)
    expected = _reference_weights(log_close, eligible, idx, spec)
    pd.testing.assert_frame_equal(weights, expected)

def test_phase_diagnostics_momentum_keeps_raw_signal(monkeypatch) -> None:
    """SCENARIO_PHASE_DIAGNOSTICS_MOMENTUM_CONSISTENT_WITH_LIVE_SIGNAL:
    ``_phase_diagnostics`` for a sign=+1 spec ranks raw ``horizon_log_return``,
    consistent with ``_book_weights`` after the vol-normalized-signal revert."""
    log_close, eligible, opens, bar_funding, idx = _signal_disagreement_panel()
    spec = _dispatch_spec(sign=1)
    captured: list[pd.DataFrame] = []
    real_rank = ev.rank_weight_book

    def recording(signal, elig, sign, min_symbols):
        captured.append(signal)
        return real_rank(signal, elig, sign, min_symbols)

    monkeypatch.setattr(ev, "rank_weight_book", recording)
    monkeypatch.setattr(diagnostics_mod, "rank_weight_book", recording)
    monkeypatch.setattr(books_mhs, "rank_weight_book", recording)
    ev._phase_diagnostics(log_close, eligible, opens, bar_funding, idx, spec)
    assert captured
    phase_grid = idx[0 :: spec.step_hours]
    expected = ev.horizon_log_return(log_close, spec.horizon_hours).reindex(phase_grid)
    vol_normalized = vol_normalized_horizon_signal(log_close, spec.horizon_hours).reindex(phase_grid)
    pd.testing.assert_frame_equal(captured[0], expected)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(captured[0], vol_normalized)

def test_phase_diagnostics_reversal_keeps_raw_signal(monkeypatch) -> None:
    """SCENARIO_PHASE_DIAGNOSTICS_REVERSAL_UNCHANGED: ``_phase_diagnostics``
    for a sign=-1 spec still ranks raw ``horizon_log_return``."""
    log_close, eligible, opens, bar_funding, idx = _signal_disagreement_panel()
    spec = _dispatch_spec(sign=-1)
    captured: list[pd.DataFrame] = []
    real_rank = ev.rank_weight_book

    def recording(signal, elig, sign, min_symbols):
        captured.append(signal)
        return real_rank(signal, elig, sign, min_symbols)

    monkeypatch.setattr(ev, "rank_weight_book", recording)
    monkeypatch.setattr(diagnostics_mod, "rank_weight_book", recording)
    monkeypatch.setattr(books_mhs, "rank_weight_book", recording)
    ev._phase_diagnostics(log_close, eligible, opens, bar_funding, idx, spec)
    assert captured
    phase_grid = idx[0 :: spec.step_hours]
    expected = ev.horizon_log_return(log_close, spec.horizon_hours).reindex(phase_grid)
    pd.testing.assert_frame_equal(captured[0], expected)

@pytest.mark.slow
def test_mhs_fast_book_mode_default_is_identity(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_FAST_BOOK_MODE_DEFAULT_IS_IDENTITY_03: the default
    # fast_book_mode is single_horizon, an invalid value raises ValueError, and
    # a full run at the default reproduces the pre-change single-horizon fast
    # book byte-identically -- the regression-invariant proof. The default-path
    # run keeps real books; the production w_fast_execution matrix is captured
    # by a spy and must equal the verbatim pre-change chain (vol tilt +
    # renormalize) built on the same panel.
    assert MhsDiagnosticRequest().fast_book_mode == "single_horizon"
    with pytest.raises(ValueError, match="unknown fast_book_mode"):
        MhsDiagnosticRequest(fast_book_mode="bogus")

    root, end = mhs_market
    captured: dict = {}
    real_books = ev._run_books_concurrent

    def _spy_books(*args, **kwargs):
        captured["w_fast_execution"] = args[10]
        return real_books(*args, **kwargs)

    monkeypatch.setattr(ev, "_run_books_concurrent", _spy_books)
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently",
        lambda *a, **k: (None, None, {}, {}, (), _deployment_readiness()),
    )
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8,
    )
    report = run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    book = report.books["fast_reversal"]
    assert book.primary_autocorr_sharpe is not None
    assert np.isfinite(book.primary_autocorr_sharpe)
    assert book.executed_prescreen_net_t is not None
    assert np.isfinite(book.executed_prescreen_net_t)

    log_close, eligible, execution_mask, _req, _grid, _end = _slow_book_panel_inputs(mhs_market)
    fast = ev.BOOK_SPECS["fast_reversal"]
    fast_grid = pd.date_range(_START, end, freq="6h", tz="UTC")
    w_fast = ev._book_weights(log_close, eligible, fast, fast_grid)
    ref_execution = ev.renormalize_within_mask(
        ev.inverse_realized_vol_tilt(
            w_fast, ev.realized_vol(log_close, fast.horizon_hours).reindex(fast_grid),
        ),
        execution_mask.reindex(w_fast.index).fillna(False), fast.min_symbols,
    )
    pd.testing.assert_frame_equal(captured["w_fast_execution"], ref_execution)

@pytest.mark.slow
def test_mhs_fast_book_mode_ensemble_produces_different_executed_book(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_FAST_BOOK_MODE_ENSEMBLE_PRODUCES_DIFFERENT_EXECUTED_BOOK_04:
    # with fast_book_mode='horizon_ensemble' the resulting fast_reversal report
    # carries a DIFFERENT primary_autocorr_sharpe (and different executed
    # prescreen net_t) than the single_horizon default on the same fixture --
    # proving the ensemble branch actually reaches the capital-book construction
    # and RC-1's dual-instrument (executed_prescreen) reflects it automatically.
    # The slow book is untouched by the fast flag. Fails against pre-change
    # code, which has no fast_book_mode branch at all.
    root, end = mhs_market
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently",
        lambda *a, **k: (None, None, {}, {}, (), _deployment_readiness()),
    )
    base = {
        "start": str(_START), "end": str(end), "data_root": str(root),
        "mark_mode": "cache_required", "execution_timeframe": "1m", "log_run": False,
        "execution_universe_size": 8,
    }
    report_default = run_mhs_horizon_diagnostic(MhsDiagnosticRequest(**base))
    report_ensemble = run_mhs_horizon_diagnostic(
        MhsDiagnosticRequest(**base, fast_book_mode="horizon_ensemble"),
    )
    fast_default = report_default.books["fast_reversal"]
    fast_ensemble = report_ensemble.books["fast_reversal"]
    assert fast_default.primary_autocorr_sharpe is not None
    assert fast_ensemble.primary_autocorr_sharpe is not None
    assert fast_default.primary_autocorr_sharpe != fast_ensemble.primary_autocorr_sharpe
    assert fast_default.executed_prescreen_net_t != fast_ensemble.executed_prescreen_net_t
    slow_default = report_default.books["slow_momentum"]
    slow_ensemble = report_ensemble.books["slow_momentum"]
    assert slow_default.primary_autocorr_sharpe == slow_ensemble.primary_autocorr_sharpe
    assert slow_default.executed_prescreen_net_t == slow_ensemble.executed_prescreen_net_t

def test_mhs_alpha_engine_slow_book_single_horizon_is_byte_identical(mhs_market) -> None:
    # SCENARIO_MHS_ALPHA_ENGINE_07: in single_horizon/raw mode
    # ``_horizon_ensemble_execution_weights`` reproduces the pre-change
    # ``_book_weights`` + tilt + renormalize sequence exactly.
    log_close, eligible, execution_mask, _request, _grid, end = _slow_book_panel_inputs(mhs_market)
    slow = ev.BOOK_SPECS["slow_momentum"]
    slow_grid = pd.date_range(_START, end, freq="24h", tz="UTC")
    slow_ema = max(1, round(slow.horizon_hours / slow.step_hours * ev.SIGNAL_EMA_HORIZON_SPAN))
    expected = _pre_change_slow_book(
        log_close, eligible, execution_mask, slow, slow_grid, slow_ema,
    )
    actual = ev._horizon_ensemble_execution_weights(
        log_close, eligible, execution_mask, slow, slow_grid,
        "single_horizon", "raw", slow_ema,
    )
    pd.testing.assert_frame_equal(actual, expected)

def test_mhs_alpha_engine_slow_book_ensemble_is_rowwise_mean_with_consensus_gross(mhs_market) -> None:
    # SCENARIO_MHS_ALPHA_ENGINE_07: in horizon_ensemble mode the output is the
    # row-wise mean of the per-horizon books on the same step grid, dollar-
    # neutral, with strictly smaller mean gross than any single horizon on a
    # panel where the horizons disagree (consensus-scaled exposure).
    log_close, eligible, execution_mask, _request, _grid, end = _slow_book_panel_inputs(mhs_market)
    slow = ev.BOOK_SPECS["slow_momentum"]
    slow_grid = pd.date_range(_START, end, freq="24h", tz="UTC")
    slow_ema = max(1, round(slow.horizon_hours / slow.step_hours * ev.SIGNAL_EMA_HORIZON_SPAN))
    per_horizon: dict[int, pd.DataFrame] = {}
    for h in slow.band.horizons_hours:
        spec = dataclasses.replace(slow, horizon_hours=h)
        per_horizon[h] = _pre_change_slow_book(
            log_close, eligible, execution_mask, spec, slow_grid, slow_ema,
        )
    expected = sum(per_horizon.values()) / len(per_horizon)
    actual = ev._horizon_ensemble_execution_weights(
        log_close, eligible, execution_mask, slow, slow_grid,
        "horizon_ensemble", "raw", slow_ema,
    )
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.sum(axis=1).abs().max() < 1e-9
    per_horizon_gross = [float(w.abs().sum(axis=1).mean()) for w in per_horizon.values()]
    ensemble_gross = float(actual.abs().sum(axis=1).mean())
    assert ensemble_gross <= 1.0 + 1e-9
    assert ensemble_gross < max(per_horizon_gross) - 1e-6

def test_mhs_alpha_engine_slow_book_validates_mode_and_signal_kind(mhs_market) -> None:
    log_close, eligible, execution_mask, _request, _grid, end = _slow_book_panel_inputs(mhs_market)
    slow = ev.BOOK_SPECS["slow_momentum"]
    slow_grid = pd.date_range(_START, end, freq="24h", tz="UTC")
    with pytest.raises(ValueError, match="mode"):
        ev._horizon_ensemble_execution_weights(
            log_close, eligible, execution_mask, slow, slow_grid,
            "bogus", "raw", None,
        )
    with pytest.raises(ValueError, match="signal_kind"):
        ev._horizon_ensemble_execution_weights(
            log_close, eligible, execution_mask, slow, slow_grid,
            "single_horizon", "bogus", None,
        )

def test_mhs_alpha_engine_fold_portfolio_trigger_preserves_invariants(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_ALPHA_ENGINE_08: with rebalance_filter='portfolio_trigger'
    # the fold target weights keep exact dollar neutrality and the realized
    # gross tracks regime_cash_scale (the trigger gates the UNSCALED book and
    # the gross scale multiplies afterwards), whereas the per-symbol deadband
    # branch leaks net exposure and decouples gross from the scale (RC-1).
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    forced_scale: dict[str, pd.Series] = {}

    def _forced_step_scale(vol_mean: pd.Series) -> pd.Series:
        out = pd.Series(
            np.where(np.arange(len(vol_mean)) < len(vol_mean) // 2, 0.5, 1.0),
            index=vol_mean.index,
        )
        forced_scale["series"] = out
        return out

    monkeypatch.setattr(scaling, "_regime_cash_scale", _forced_step_scale)
    request_trig = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        rebalance_filter="portfolio_trigger",
    )
    target_trig, _signal, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request_trig, funding_by_symbol,
    )
    scale = forced_scale["series"].reindex(target_trig.index)
    assert target_trig.sum(axis=1).abs().max() < 1e-9
    assert (target_trig.abs().sum(axis=1) - scale).abs().max() < 1e-9
    assert target_trig.abs().sum(axis=1).max() <= 1.0 + 1e-9

    request_dead = dataclasses.replace(request_trig, rebalance_filter="per_symbol_deadband", committee_target_gross=None)
    target_dead, _signal, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request_dead, funding_by_symbol,
    )
    assert target_dead.sum(axis=1).abs().max() > 1e-3
    assert (target_dead.abs().sum(axis=1) - scale).abs().max() > 1e-3

def test_mhs_alpha_engine_request_field_validation() -> None:
    # SCENARIO_MHS_ALPHA_ENGINE_08 (second half): MhsDiagnosticRequest raises
    # ValueError on unknown slow_book_mode/rebalance_filter/ensemble_signal
    # values and on a non-bool beta_neutralize; the defaults stay frozen.
    req = MhsDiagnosticRequest()
    assert req.slow_book_mode == "single_horizon"
    assert req.rebalance_filter == "per_symbol_deadband"
    assert req.beta_neutralize is False
    assert req.ensemble_signal == "raw"
    with pytest.raises(ValueError, match="slow_book_mode"):
        MhsDiagnosticRequest(slow_book_mode="bogus")
    with pytest.raises(ValueError, match="rebalance_filter"):
        MhsDiagnosticRequest(rebalance_filter="bogus")
    with pytest.raises(ValueError, match="beta_neutralize"):
        MhsDiagnosticRequest(beta_neutralize=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ensemble_signal"):
        MhsDiagnosticRequest(ensemble_signal="bogus")