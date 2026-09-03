"""MHS evaluation core tests (second-level split by domain)."""

"""MHS evaluation core contract tests (everything not in a domain-specific split file)."""
"""Contract coverage for the MHS application evaluation resource telemetry."""
import dataclasses
import math
from concurrent.futures import Future

import numpy as np
import pandas as pd
import pytest
from src.application.research.mhs import evaluation as ev
import src.application.research.mhs.resources as resources
import src.application.research.mhs.scaling as scaling
from src.application.research.mhs.evaluation import (
    MhsDiagnosticRequest,
    _StageRecorder,
)
from src.research.universe.pit_universe import symbol_partition
from tests.unit.application.research.mhs.test_evaluation import (  # noqa: F401
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

@pytest.mark.slow
class TestAnchoredFoldBounded:
    """MHS-MEM-03-ANCHORED-FOLD-BOUNDED: each anchored fold uses bounded
    windowed replay (no dense fold-wide minute panel) and enforces the
    configured RSS budget with stable provenance."""

    def _run_fold(self, mhs_market, max_rss_bytes=None):
        root, end = mhs_market
        symbols = [
            s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                        "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
            if symbol_partition(s) == "dev"
        ][:8]
        funding_by_symbol, _ = ev._load_funding_series(symbols)
        request = MhsDiagnosticRequest(
            start=str(_START), end=str(end), data_root=str(root),
            mark_mode="cache_required", execution_timeframe="1m", log_run=False,
            max_rss_bytes=max_rss_bytes,
        )
        return ev._run_anchored_fold(
            str(root), _FOLD, request, funding_by_symbol, 1.0, 0, None,
        )

    def test_fold_uses_windowed_replay_dense_snapshots_disabled(self, mhs_market) -> None:
        report = self._run_fold(mhs_market)
        assert report.strict is not None
        assert report.strict.event_snapshots_retained is False
        assert report.stress is not None
        assert report.stress.event_snapshots_retained is False

    def test_fold_records_ordered_window_telemetry(self, mhs_market) -> None:
        root, end = mhs_market
        symbols = [
            s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                        "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
            if symbol_partition(s) == "dev"
        ][:8]
        funding_by_symbol, _ = ev._load_funding_series(symbols)
        request = MhsDiagnosticRequest(
            start=str(_START), end=str(end), data_root=str(root),
            mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        )
        recorder = _StageRecorder(log_run=False)
        ev._run_anchored_fold(str(root), _FOLD, request, funding_by_symbol, 1.0, 0, recorder)
        window_stages = [m.stage for m in recorder.records if m.stage.startswith("anchored_fold_0_window_")]
        assert window_stages, "fold paired window telemetry must be recorded"
        # The reference pass records each window under ``_window_``; the
        # rescaled primary/stress pair share one interleaved stream under
        # ``_window_rescaled_``, so no separate stress re-iteration exists.
        reference_stages = [
            s for s in window_stages
            if not s.startswith("anchored_fold_0_window_rescaled")
        ]
        assert reference_stages == sorted(reference_stages)
        rescaled_stages = [
            s for s in window_stages
            if s.startswith("anchored_fold_0_window_rescaled")
        ]
        assert rescaled_stages == sorted(rescaled_stages)
        # The interleaved fan-out records one physical window per stage: the
        # stress bound consumes the same iterator, so no separate stress
        # re-iteration telemetry exists.
        assert not [
            m.stage for m in recorder.records
            if m.stage.startswith("anchored_fold_0_stress_window_")
        ]

    def test_fold_builds_window_iterator_twice_streaming(self, mhs_market, monkeypatch) -> None:
        """SCENARIO_MHS_STREAM_FOLD_TWO_GENERATIONS: the streaming fold
        regenerates the execution windows exactly twice -- once for the
        reference pass and once for the interleaved rescaled primary/stress
        batch -- never once per bound (the bounded-memory successor of the
        materialize-once invariant)."""
        root, end = mhs_market
        symbols = [
            s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                        "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
            if symbol_partition(s) == "dev"
        ][:8]
        funding_by_symbol, _ = ev._load_funding_series(symbols)
        request = MhsDiagnosticRequest(
            start=str(_START), end=str(end), data_root=str(root),
            mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        )
        calls = {"n": 0}
        original = ev._iter_mhs_execution_windows

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(ev, "_iter_mhs_execution_windows", counting)
        report = ev._run_anchored_fold(
            str(root), _FOLD, request, funding_by_symbol, 1.0, 0, None,
        )
        assert report.strict is not None
        assert report.stress is not None
        # Reference pass + one interleaved rescaled batch (bounded memory).
        assert calls["n"] == 2

    def test_rss_budget_enforced_inside_fold_fails_closed(self, mhs_market, monkeypatch) -> None:
        monkeypatch.setattr(resources, "_current_rss_bytes", lambda: 100_000_000_000)
        report = self._run_fold(mhs_market, max_rss_bytes=1_000)
        # The budget DataIntegrityError becomes a typed fold failure (not an
        # uncaught process error) under the fold contract's fail-closed code
        # set. An RSS breach is classified as RESOURCE_BUDGET_BREACH (spec
        # §3.3 ``fold_integrity``), never as an invalid primary ledger.
        assert report.strict is None
        assert report.stress is None
        assert report.failures == (ev.GO_REASON_RESOURCE_BREACH,)

    def test_no_rss_budget_returns_complete_fold(self, mhs_market) -> None:
        report = self._run_fold(mhs_market, max_rss_bytes=None)
        # I-FAMILY: 완결 fold의 failures는 데이터 무결성 코드만 담는다 --
        # fold별 level 코드(Sharpe/stress/연수익)는 pooled 게이트로 이전했다.
        assert report.strict is not None
        assert report.failures == ()

@pytest.mark.slow
def test_anchored_fold_is_two_pass(mhs_market, monkeypatch) -> None:
    # SCENARIO_ANCHORED_FOLD_IS_TWO_PASS: the fold's reported primary
    # (strict/autocorr-sharpe/max-drawdown) reflects the P&L-vol-target
    # rescaled Pass-2 replay, not the unscaled reference. An engineered
    # non-trivial scale must move the fold's reported metrics away from the
    # all-ones (reference-equivalent) run.
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ][:8]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
    )

    def _all_ones_scale(reference_daily_returns: pd.Series) -> pd.Series:
        return pd.Series(1.0, index=reference_daily_returns.index)

    def _forced_step_scale(reference_daily_returns: pd.Series) -> pd.Series:
        idx = reference_daily_returns.index
        mid = idx[0] + (idx[-1] - idx[0]) / 2
        return pd.Series(np.where(idx < mid, 1.0, 0.2), index=idx)

    monkeypatch.setattr(scaling, "_pnl_vol_target_scale", _all_ones_scale)
    reference = ev._run_anchored_fold(
        str(root), _FOLD, request, funding_by_symbol, 1.0, 0, None,
    )
    monkeypatch.setattr(scaling, "_pnl_vol_target_scale", _forced_step_scale)
    rescaled = ev._run_anchored_fold(
        str(root), _FOLD, request, funding_by_symbol, 1.0, 0, None,
    )
    assert reference.strict is not None
    assert reference.stress is not None
    assert rescaled.strict is not None
    assert rescaled.stress is not None
    assert rescaled.primary_autocorr_sharpe != reference.primary_autocorr_sharpe
    # The rescaled Pass-2 replay must have traded a genuinely different book
    # than the all-ones reference; the max drawdown is not a reliable differentiator
    # because the fold book now tracks the alpha roster closely, so the MDD
    # window is dominated by the identical (unscaled) first half.
    assert not rescaled.strict.ledger.equity.equals(reference.strict.ledger.equity)

def test_fold_execution_weights_are_renormalized(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_FOLD_EXECUTION_WEIGHTS_ARE_RENORMALIZED: the fold builder
    # re-normalizes its execution weights onto the roster instead of collapsing
    # them to a partial-gross subset of the full-universe book.
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8,
    )
    real = ev.renormalize_within_mask
    captured: list[tuple[pd.DataFrame, pd.DataFrame, int]] = []

    def spy(weights, mask, min_symbols):
        out = real(weights, mask, min_symbols)
        captured.append((out, mask, min_symbols))
        return out

    monkeypatch.setattr(ev, "renormalize_within_mask", spy)
    target_weights, _signal, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request, funding_by_symbol,
    )
    assert captured, "fold builder must route execution weights through renormalize_within_mask"
    assert not target_weights.empty
    for out, mask, min_symbols in captured:
        live = mask.sum(axis=1) >= min_symbols
        assert live.any(), "fold decision rows must have a live roster"
        # unit-gross and dollar-neutral within the surviving roster cells
        assert out.abs().sum(axis=1).where(live).sub(1.0).abs().max() < 1e-9
        assert out.sum(axis=1).where(live).abs().max() < 1e-9
        # masked-out columns are exactly zero, never the unnormalized input
        assert float(out[~mask].abs().max().max()) == 0.0

def test_fold_weights_are_vol_tilted_before_renormalization(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_FOLD_WEIGHTS_ARE_VOL_TILTED_BEFORE_RENORMALIZATION: the fold
    # builder tilts each book by its own-horizon inverse realized vol before the
    # unchanged renormalize_within_mask, so a higher-vol roster symbol receives
    # a smaller post-tilt, pre-renormalization magnitude than an equal-rank
    # lower-vol symbol.
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8,
    )

    tilt_calls: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    renorm_inputs: list[pd.DataFrame] = []
    real_tilt = ev.inverse_realized_vol_tilt
    real_renorm = ev.renormalize_within_mask

    def tilt_spy(weights, vol):
        tilt_calls.append((weights, vol))
        return real_tilt(weights, vol)

    def renorm_spy(weights, mask, min_symbols):
        renorm_inputs.append(weights)
        return real_renorm(weights, mask, min_symbols)

    monkeypatch.setattr(ev, "inverse_realized_vol_tilt", tilt_spy)
    monkeypatch.setattr(ev, "renormalize_within_mask", renorm_spy)
    ev._build_fold_target_weights(str(root), _FOLD, request, funding_by_symbol)

    assert len(tilt_calls) == 2, "fold builder must tilt both the fast and slow books"
    assert len(renorm_inputs) == 2, "fold builder must renormalize both tilted books"
    for (raw, vol), renorm_in in zip(tilt_calls, renorm_inputs, strict=True):
        # renormalize receives the tilt output -- the raw rank book scaled by
        # 1/vol -- never the untilted book.
        assert renorm_in.equals(real_tilt(raw, vol))
        valid = np.isfinite(vol.to_numpy(dtype="float64")) & (vol.to_numpy(dtype="float64") > 0.0)
        assert valid.any(), "tilt must be a real scaling, not a no-op"

    # The tilt is applied on each book's own horizon and reindexed onto its grid.
    fast = ev.BOOK_SPECS["fast_reversal"]
    slow = ev.BOOK_SPECS["slow_momentum"]
    panel_start = max(
        _FOLD.train_start,
        _FOLD.validation_start - pd.Timedelta(hours=ev.FOLD_PANEL_WARMUP_HOURS),
    )
    fast_grid = pd.date_range(panel_start, _FOLD.validation_end, freq="6h", tz="UTC")
    slow_grid = pd.date_range(panel_start, _FOLD.validation_end, freq="24h", tz="UTC")
    fast_raw, fast_vol = tilt_calls[0]
    slow_raw, slow_vol = tilt_calls[1]
    assert fast_raw.index.equals(fast_grid)
    assert fast_vol.index.equals(fast_grid)
    assert slow_raw.index.equals(slow_grid)
    assert slow_vol.index.equals(slow_grid)

    # Semantic ordering: among roster symbols sharing an equal raw rank-slot
    # magnitude (the book's symmetric extremes), the higher-realized-vol symbol
    # has the strictly smaller pre-renormalization magnitude.
    fast_tilted = real_tilt(fast_raw, fast_vol)
    pairs: list[tuple[int, int, int, float, float]] = []
    for row in range(len(fast_tilted)):
        mags = fast_raw.iloc[row].abs().to_numpy(dtype="float64")
        vols = fast_vol.iloc[row].to_numpy(dtype="float64")
        valid = np.isfinite(vols) & (vols > 0.0) & (mags > 1e-6)
        pairs.extend(
            (row, i, j, float(vols[i]), float(vols[j]))
            for i in range(len(mags))
            for j in range(i + 1, len(mags))
            if valid[i] and valid[j] and np.isclose(mags[i], mags[j]) and vols[i] != vols[j]
        )
    assert pairs, "fixture must contain equal-|rank-weight| pairs with differing realized vol"
    for row, i, j, vi, vj in pairs:
        hi, lo = (i, j) if vi > vj else (j, i)
        assert abs(fast_tilted.iloc[row, hi]) < abs(fast_tilted.iloc[row, lo])

def test_fold_vol_mean_masked_to_execution_roster(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_VOL_MEAN_ROSTER_MASK_01: the fold builder's regime-cash-scale
    # vol_mean is computed from execution_mask-filtered realized vol -- a
    # high-vol symbol outside the traded roster must not pull the regime scale.
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8,
    )
    captured: dict[str, pd.Series] = {}
    real_scale = scaling._regime_cash_scale

    def spy(vol_mean, *args, **kwargs):
        captured["vol_mean"] = vol_mean.copy()
        return real_scale(vol_mean, *args, **kwargs)

    monkeypatch.setattr(scaling, "_regime_cash_scale", spy)
    ev._build_fold_target_weights(str(root), _FOLD, request, funding_by_symbol)
    assert "vol_mean" in captured, "fold builder must feed _regime_cash_scale its vol_mean"

    panel_start = max(
        _FOLD.train_start,
        _FOLD.validation_start - pd.Timedelta(hours=ev.FOLD_PANEL_WARMUP_HOURS),
    )
    log_close, execution_mask, _grid = _roster_mask_panel_inputs(
        root, panel_start, _FOLD.validation_end, funding_by_symbol,
        request.execution_universe_size,
    )
    _assert_regime_vol_mean_roster_masked(
        captured, log_close, execution_mask, captured["vol_mean"].index,
    )

def test_fold_decision_grid_matches_slow_cadence(mhs_market) -> None:
    # SCENARIO_MHS_FOLD_DECISION_GRID_MATCHES_SLOW_CADENCE_05: under the
    # fixture with the current frozen weights, _build_fold_target_weights's
    # target_weights index has a row spacing consistent with the 24h slow_grid
    # (not the 1h native grid_1h) for the validation window -- the fold-level
    # Research-GO gate no longer decides at native-hourly cadence when only
    # slow_momentum is admitted.
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
    )
    target_weights, _signal, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request, funding_by_symbol,
    )
    assert not target_weights.empty
    spacing = target_weights.index.to_series().diff().dropna()
    assert not spacing.empty
    assert (spacing == pd.Timedelta(hours=24)).all()

def test_committee_capital_default_off_bit_identical(mhs_market_with_taker_buy_quote, monkeypatch) -> None:
    # SCENARIO_MHS_COMMITTEE_CAPITAL_DEFAULT_OFF_BIT_IDENTICAL: with the opt-in
    # disabled (committee_capital defaults False) _build_fold_target_weights
    # executes zero committee code -- proved by monkeypatching build_feature_books
    # to raise if ever called -- and returns target weights byte-identical to an
    # unpatched baseline run.
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
        fill_mark_parity_gate=False,
    )
    assert request.committee_capital is False

    def _must_not_be_called(*_args, **_kwargs):
        raise AssertionError("must not be called")

    monkeypatch.setattr(ev, "build_feature_books", _must_not_be_called)
    target_patched, _signal, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request, funding_by_symbol,
    )

    monkeypatch.undo()
    target_baseline, _signal, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request, funding_by_symbol,
    )
    pd.testing.assert_frame_equal(target_patched, target_baseline)

def test_committee_capital_reaches_fold_targets(mhs_market_with_taker_buy_quote) -> None:
    # SCENARIO_MHS_COMMITTEE_CAPITAL_REACHES_FOLD_TARGETS: with committee_capital
    # enabled the fold decision targets become the equal-weight committee blend,
    # Verify committee capital feeds fold weights while preserving neutrality.
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
        rebalance_filter="portfolio_trigger", fill_mark_parity_gate=False,
    )
    target_off, _signal, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request, funding_by_symbol,
    )
    request_on = dataclasses.replace(request, committee_capital=True)
    target_on, _signal, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request_on, funding_by_symbol,
    )
    assert not target_off.equals(target_on)
    assert np.isfinite(target_on.to_numpy(dtype="float64")).all()
    assert float(target_on.sum(axis=1).abs().max()) < 1e-6
    assert float(target_on.abs().max().max()) <= 1.0 + 1e-9

def test_committee_capital_no_member_fails_closed(mhs_market_with_taker_buy_quote, monkeypatch) -> None:
    # SCENARIO_MHS_COMMITTEE_CAPITAL_NO_MEMBER_FAILS_CLOSED: when no committee
    # member is admitted, the fold target builder raises RuntimeError naming
    # committee_capital instead of silently falling back to the momentum blend.
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
        committee_capital=True,
    )
    monkeypatch.setattr(ev, "build_feature_books", lambda *a, **k: {})
    with pytest.raises(RuntimeError, match="committee_capital"):
        ev._build_fold_target_weights(str(root), _FOLD, request, funding_by_symbol)

def _parity_fold_report(
    fold_index: int, book_structure: dict[str, float] | None,
) -> ev.MhsFoldReport:
    return ev.MhsFoldReport(
        fold_index=fold_index,
        validation_start="2022-01-08",
        validation_end="2022-12-31",
        strict=None,
        stress=None,
        primary_valid=False,
        primary_autocorr_sharpe=0.0,
        primary_naive_sharpe=0.0,
        primary_net_ann=0.0,
        primary_geometric_cagr=0.0,
        primary_max_drawdown=0.0,
        stress_naive_sharpe=0.0,
        decision_intents=0,
        termination_counts={},
        failures=(),
        strict_elapsed_seconds=0.0,
        stress_elapsed_seconds=0.0,
        book_structure=book_structure,
    )


def test_fold_blend_parity_measures_deployed_gross() -> None:
    # SCENARIO_MHS_FOLD_BLEND_DEPLOYED_PARITY_05: the parity guard must measure
    # the deployed (post-exposure-scale) gross. Identical pre-scale books
    # (gross_mean 0.84 both sides) whose exposure scales diverge (0.63 vs 1.00)
    # emit FOLD_BLEND_PATH_DIVERGENCE with max_abs_log_deployed_gross_ratio ==
    # |log(0.63)| while the pre-scale ratio stays 0.0 -- proving the pre-scale
    # measurement alone cannot detect the divergence (D2). A trace without
    # exposure_scale_mean lands in unmeasured and emits no code by itself.
    fold_trace = {"n_rows": 100.0, "gross_mean": 0.84, "holdings_mean": 42.0, "exposure_scale_mean": 0.63}
    blend_trace = {"n_rows": 100.0, "gross_mean": 0.84, "holdings_mean": 42.0, "exposure_scale_mean": 1.00}
    fold = _parity_fold_report(0, fold_trace)
    payload, reasons = ev._fold_blend_parity({0: blend_trace}, (fold,))
    assert reasons == (ev.GO_REASON_PATH_DIVERGENCE,)
    assert payload["max_abs_log_deployed_gross_ratio"] == pytest.approx(abs(math.log(0.63)), rel=1e-9)
    assert payload["max_abs_log_gross_ratio"] == pytest.approx(0.0)
    assert payload["folds"][0]["deployed_gross_log_ratio"] == pytest.approx(math.log(0.63), rel=1e-9)

    fold_no_scale = _parity_fold_report(1, {"n_rows": 100.0, "gross_mean": 0.84, "holdings_mean": 42.0})
    payload_no_scale, reasons_no_scale = ev._fold_blend_parity({1: blend_trace}, (fold_no_scale,))
    assert reasons_no_scale == ()
    assert 1 in payload_no_scale["unmeasured"]
    assert payload_no_scale["folds"][1]["deployed_gross_log_ratio"] is None
    # The missing field is never silently treated as scale 1.0: no deployed
    # ratio was folded into the maximum.
    assert payload_no_scale["max_abs_log_deployed_gross_ratio"] == pytest.approx(0.0)


def test_book_outcome_blend_traces_carry_deployed_exposure_scale(mhs_market, monkeypatch) -> None:
    # Regression for the D2 wiring gap: _fold_blend_parity's deployed-gross
    # check (SCENARIO_MHS_FOLD_BLEND_DEPLOYED_PARITY_05) is worthless in
    # production unless the top-level blend_traces built by _book_outcome
    # actually carry exposure_scale_mean -- previously blend_traces was built
    # from _book_structure_trace(target_weights) alone, so the blend side of
    # the ratio was always None and the divergence check could never fire.
    # This drives the REAL two-pass replay (growth_budget mode, which never
    # takes the coupled/streaming path per is_streaming_scale_mode) with a
    # single synthetic fold inside the tiny fixture's own date range, and
    # asserts blend_traces[0] carries a finite, strictly-positive
    # exposure_scale_mean matching the deployed (post pnl_vol_target) scale.
    from src.mhs.evidence import AnchoredPurgedFold

    fold = AnchoredPurgedFold(
        train_start=_START,
        train_end=_START + pd.Timedelta(hours=200),
        validation_start=_START + pd.Timedelta(hours=250),
        validation_end=_START + pd.Timedelta(hours=2600),
        forward_dependency_hours=1,
        purge_hours=50,
    )
    # SCENARIO_MHS_FOLD_RESTRUCTURE_NO_HARDCODED_CONSUMERS: injected stub
    # folds stay count-agnostic; the real schedule is asserted in test_evaluation.py.
    monkeypatch.setattr(ev, "phase_1_anchored_purged_folds", lambda: (fold,))

    args = _build_book_outcome_args(mhs_market)
    args["name"] = "blend"
    args["request"] = dataclasses.replace(
        args["request"], pnl_vol_target_mode="growth_budget", log_run=False,
    )
    report, blend_traces = ev._book_outcome(**args)
    assert report.failure is None
    assert 0 in blend_traces
    scale_mean = blend_traces[0]["exposure_scale_mean"]
    assert isinstance(scale_mean, float)
    assert np.isfinite(scale_mean)
    assert scale_mean > 0.0

    # End-to-end proof the wiring actually feeds the parity guard: a fold-side
    # trace whose exposure differs materially from the real blend-side value
    # now trips FOLD_BLEND_PATH_DIVERGENCE using the genuine blend trace, not
    # a hand-fabricated one.
    diverging_fold_trace = {**blend_traces[0], "exposure_scale_mean": scale_mean * 3.0}
    fold_report = _parity_fold_report(0, diverging_fold_trace)
    _, reasons = ev._fold_blend_parity(blend_traces, (fold_report,))
    assert reasons == (ev.GO_REASON_PATH_DIVERGENCE,)


def test_constant_risk_fold_reuses_blend_exposure_scale(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_RUN_ANCHORED_FOLD_USES_BLEND_SCALE_DIRECTLY: under
    # constant_risk the fold consumes blend_exposure_scale verbatim
    # (I-SCALE-IS-DEPLOYED-OVERLAY) -- book_structure's exposure_scale_mean
    # equals the passed series mean over the validation dates, and the
    # fold-local replay dispatcher is never invoked.
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ][:8]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        pnl_vol_target_mode="constant_risk",
    )
    daily_idx = pd.date_range(
        _FOLD.validation_start.normalize(), _FOLD.validation_end.normalize(),
        freq="D", tz="UTC",
    )
    # 날짜별로 구분 가능하면서 배치 가능한(자본 불변 위반이 없는) 현실적 스케일.
    blend_scale = pd.Series(
        0.8 + 0.4 * (daily_idx.dayofyear.to_numpy(dtype="float64") / 366.0),
        index=daily_idx,
    )

    def _must_not_replay(*_a: object, **_k: object) -> pd.Series:
        raise AssertionError("constant_risk folds must consume the blend scale, never re-replay it")

    monkeypatch.setattr(scaling, "_replay_exposure_scale", _must_not_replay)
    report = ev._run_anchored_fold(
        str(root), _FOLD, request, funding_by_symbol, 1.0, 0, None,
        blend_exposure_scale=blend_scale,
    )
    assert report.strict is not None
    assert report.failures == ()
    assert report.book_structure["exposure_scale_mean"] == pytest.approx(float(blend_scale.mean()))


def test_constant_risk_fold_missing_blend_scale_fails_closed(mhs_market) -> None:
    # SCENARIO_MHS_RUN_ANCHORED_FOLD_USES_BLEND_SCALE_DIRECTLY (case 2): a
    # missing blend_exposure_scale converts into a typed incomplete-fold
    # failure through the existing try/except DataIntegrityError wrapper --
    # never an uncaught error and never a silent unscaled fallback.
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ][:8]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        pnl_vol_target_mode="constant_risk",
    )
    report = ev._run_anchored_fold(str(root), _FOLD, request, funding_by_symbol, 1.0, 0, None)
    assert report.strict is None
    assert report.stress is None
    assert len(report.failures) == 1


class _InlineExecutor:
    """Synchronous stand-in for the fork pool: runs submissions in-parent."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __enter__(self) -> "_InlineExecutor":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def submit(self, fn, /, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future


_CAPTURED_FOLD_SUBMISSIONS: list[dict[str, object]] = []


def _capturing_anchored_fold(
    root, fold, request, funding_by_symbol, initial_equity, fold_index,
    telemetry=None, slow_horizon_override=None, fast_horizon_override=None,
    funding_carry_override=None, committee_member_weights=None,
    growth_budget_target_vol=None, exposure_warmup_returns=None,
    blend_exposure_scale=None,
):
    _CAPTURED_FOLD_SUBMISSIONS.append({
        "fold_index": fold_index,
        "growth_budget_target_vol": growth_budget_target_vol,
        "exposure_warmup_returns": exposure_warmup_returns,
        "blend_exposure_scale": blend_exposure_scale,
    })
    return ev._incomplete_fold_report(fold, fold_index, ())


def test_post_book_concurrently_forwards_boundary_growth_budget_vols(monkeypatch) -> None:
    # SCENARIO_MHS_FOLD_GROWTH_BUDGET_PROPAGATION_06: the boundary-resolved
    # target-vol mapping reaches each fold submission as its trailing keyword;
    # a None mapping forwards None everywhere so every other run stays
    # byte-identical.
    monkeypatch.setattr(ev, "phase_1_anchored_purged_folds", lambda: (_FOLD,) * 4)
    monkeypatch.setattr(ev, "_run_anchored_fold", _capturing_anchored_fold)
    monkeypatch.setattr(ev, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(ev, "plan_worker_count", lambda *a, **k: 1)
    monkeypatch.setattr(ev, "assert_fork_admission", lambda *a, **k: None)
    request = MhsDiagnosticRequest(log_run=False)

    _CAPTURED_FOLD_SUBMISSIONS.clear()
    ev._run_post_book_concurrently(
        None, "root", request, [], None, None, None, None, None, None, None, {}, 1.0, None,
        fold_growth_budget_target_vol={0: 0.30, 1: 0.31, 2: 0.32, 3: 0.33},
    )
    forwarded = {
        int(c["fold_index"]): c["growth_budget_target_vol"]
        for c in _CAPTURED_FOLD_SUBMISSIONS
    }
    assert forwarded == {0: 0.30, 1: 0.31, 2: 0.32, 3: 0.33}

    _CAPTURED_FOLD_SUBMISSIONS.clear()
    ev._run_post_book_concurrently(
        None, "root", request, [], None, None, None, None, None, None, None, {}, 1.0, None,
    )
    forwarded_none = {
        int(c["fold_index"]): c["growth_budget_target_vol"]
        for c in _CAPTURED_FOLD_SUBMISSIONS
    }
    assert forwarded_none == {0: None, 1: None, 2: None, 3: None}

    _CAPTURED_FOLD_SUBMISSIONS.clear()
    blend_slices = {
        0: pd.Series([1.0], index=pd.DatetimeIndex([pd.Timestamp("2021-06-01", tz="UTC")])),
        3: pd.Series([2.0], index=pd.DatetimeIndex([pd.Timestamp("2024-06-01", tz="UTC")])),
    }
    ev._run_post_book_concurrently(
        None, "root", request, [], None, None, None, None, None, None, None, {}, 1.0, None,
        fold_blend_exposure_scale=blend_slices,
    )
    forwarded_blend = {
        int(c["fold_index"]): c["blend_exposure_scale"]
        for c in _CAPTURED_FOLD_SUBMISSIONS
    }
    assert forwarded_blend[0] is blend_slices[0]
    assert forwarded_blend[3] is blend_slices[3]
    assert forwarded_blend[1] is None
    assert forwarded_blend[2] is None


def test_p14_postbook_concurrent_parity() -> None:
    # SCENARIO_P14_POSTBOOK: the deployment tail computed with the placeholder
    # ``research_go_eligible=None`` and then patched with the fold-derived flag
    # is identical to computing it directly with that flag, so the concurrent
    # post-book path cannot change the readiness result.
    idx = pd.date_range("2021-01-01", periods=3000, freq="1h", tz="UTC")
    rng = np.random.default_rng(42)
    equity = pd.Series(np.cumprod(1.0 + rng.normal(0.0002, 0.004, len(idx))), index=idx)
    full = ev.compute_deployment_readiness(
        equity, 365 * 24, research_go_eligible=False, n_bootstrap=20, seed=7,
    )
    placeholder = ev.compute_deployment_readiness(
        equity, 365 * 24, research_go_eligible=None, primary_valid=True,
        n_bootstrap=20, seed=7,
    )

    patched = dataclasses.replace(placeholder, research_go_eligible=False)
    assert patched == full

def test_p14_postbook_no_deadlock(monkeypatch) -> None:
    # SCENARIO_P14_NO_DEADLOCK: with no anchored folds the concurrent
    # orchestration degrades to the sequential diagnostics tail through the
    # same entry point, proving the fold-pool/thread orchestration never
    # deadlocks or hangs.
    class _FakePrimary:
        ledger = None

    class _FakeBlend:
        primary = _FakePrimary()

    monkeypatch.setattr(ev, "phase_1_anchored_purged_folds", lambda: ())
    calls = {"n": 0}

    def _fast_diag(*_args, **_kwargs):
        calls["n"] += 1
        return (None, None, {}, {}, None)

    monkeypatch.setattr(ev, "_run_post_diag_deploy", _fast_diag)
    result = ev._run_post_book_concurrently(
        _FakeBlend(), "root", None, [], None, None, None, None, None, None, None, {}, 1.0, None,
    )
    assert calls["n"] == 1
    assert result[4] == ()
    assert result[5] is None

def test_fold_worker_records_fast_horizon_override(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_FOLD_REPORT_FAST_HORIZON_FIELDS_DEFAULT (fold worker path):
    # a fold run resolved with a fast fold-scoped override records the selected
    # (horizon, source) on the report while the slow fields stay on the frozen
    # default -- mirroring the slow_horizon_* recording path and keeping the
    # fast re-verification diagnostic-only (no BookSpec/weight construction).
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ][:8]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8,
    )
    report = ev._run_anchored_fold(
        str(root), _FOLD, request, funding_by_symbol, 1.0, 0, None,
        fast_horizon_override=(96, "fold_train_only_discovery"),
    )
    assert report.fast_horizon_hours == 96
    assert report.fast_horizon_source == "fold_train_only_discovery"
    assert report.slow_horizon_hours == 168
    assert report.slow_horizon_source == "frozen_default"

@pytest.mark.slow
def test_diagnostics_run_after_folds_and_evict_caches(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_MHS_DIAGNOSTICS_RUN_AFTER_FOLDS_AND_EVICT_CACHES: the opt-in
    # diagnostics run only after the fold pool returned, the minute/mark frame
    # caches are evicted by the time the run completes, and the committee
    # diagnostic is still populated (regression against the re-ordering).
    root, end = mhs_market_long
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    order: list[str] = []
    real_post = ev._run_post_book_concurrently
    real_committee = ev._committee_diagnostic

    def _spy_post(*args, **kwargs):
        order.append("post_folds")
        return real_post(*args, **kwargs)

    def _spy_committee(*args, **kwargs):
        order.append("committee")
        return real_committee(*args, **kwargs)

    monkeypatch.setattr(ev, "_run_post_book_concurrently", _spy_post)
    monkeypatch.setattr(ev, "_committee_diagnostic", _spy_committee)

    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, committee_book=True,
    )
    report = ev.run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    assert order == ["post_folds", "committee"]
    assert isinstance(report.committee_diagnostic, dict)
    assert report.committee_diagnostic["evaluation_protocol"] == "purged_walk_forward_oos"

    # The full-period mark frame cache was evicted during the run (the minute
    # frame caches were removed in the fork-COW refactor).
    assert ev._get_symbol_mark_frame.cache_info().currsize == 0

def test_fold_worker_records_funding_carry_override(mhs_market) -> None:
    # SCENARIO_MHS_FOLD_REPORT_CARRIES_FUNDING_CARRY_DISCOVERY_05 (fold worker
    # path): a fold run resolved with a funding-carry override records all four
    # fields on the report; without the override (flag off / no admission) they
    # fail closed to their dataclass defaults -- existing construction sites
    # are unaffected.
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ][:8]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8,
    )
    report = ev._run_anchored_fold(
        str(root), _FOLD, request, funding_by_symbol, 1.0, 0, None,
        funding_carry_override=(72, 1, "fold_train_only_discovery", 0.15),
    )
    assert report.funding_carry_lookback_hours == 72
    assert report.funding_carry_sign == 1
    assert report.funding_carry_source == "fold_train_only_discovery"
    assert report.funding_carry_vs_slow_momentum_daily_corr == 0.15

    default_report = ev._run_anchored_fold(
        str(root), _FOLD, request, funding_by_symbol, 1.0, 0, None,
    )
    assert default_report.funding_carry_lookback_hours is None
    assert default_report.funding_carry_sign is None
    assert default_report.funding_carry_source == "frozen_default"
    assert default_report.funding_carry_vs_slow_momentum_daily_corr is None

    incomplete = ev._incomplete_fold_report(_FOLD, 0, ())
    assert incomplete.funding_carry_lookback_hours is None
    assert incomplete.funding_carry_source == "frozen_default"
