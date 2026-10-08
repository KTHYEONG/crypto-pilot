"""MHS evaluation core tests (second-level split by domain)."""

"""MHS evaluation core contract tests (everything not in a domain-specific split file)."""
"""Contract coverage for the MHS application evaluation resource telemetry."""
from tests.fixtures.mhs_requests import research_baseline

import numpy as np
import pandas as pd
import pytest
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


def _synthetic_fold_panel():
    import pandas as pd

    from src.mhs.evidence import AnchoredPurgedFold

    idx = pd.date_range("2021-01-01", periods=2000, freq="1h", tz="UTC")
    cols = ["AAAUSDT", "BBBUSDT", "BTCUSDT"]
    rng = np.random.default_rng(0)
    close = pd.DataFrame(100 + np.cumsum(rng.normal(0, 0.1, (2000, 3)), axis=0), index=idx, columns=cols)
    base_panel = {"close": close, "open": close.copy(), "quote_vol": pd.DataFrame(1e6, index=idx, columns=cols)}
    fold = AnchoredPurgedFold(idx[0], idx[100], idx[800], idx[1800], 24, 24)
    return idx, cols, base_panel, fold, {c: pd.Series(0.0, index=idx) for c in cols}


def test_fold_weights_mark_independence(monkeypatch) -> None:
    """Fold weights use the panel constant and ignore Mark loaders."""

    import src.mhs.evaluation.fold_weights as fw
    import src.core.marks as marks_mod
    from src.core.params import PANEL_MIN_HISTORY_BARS

    idx, cols, base_panel, fold, funding = _synthetic_fold_panel()
    real_eligibility = fw.liquid_half_eligibility
    recorded: dict[str, object] = {}
    calls = {"n": 0}
    def _recording(frame, *args, **kwargs):
        calls["n"] += 1
        recorded["frame"] = frame
        recorded.update(kwargs)
        return real_eligibility(frame, *args, **kwargs)

    monkeypatch.setattr(fw, "liquid_half_eligibility", _recording)
    request = research_baseline()
    baseline, _, _, _ = fw._build_fold_target_weights("root", fold, request, funding,
        base_panel=base_panel, require_minute_roster=False, panel_warmup_hours=24)
    assert calls["n"] == 1
    assert recorded["lookback_bars"] == recorded["min_history_bars"] == PANEL_MIN_HISTORY_BARS
    pd.testing.assert_frame_equal(recorded["frame"], base_panel["quote_vol"].loc[recorded["frame"].index])

    def _boom(*args, **kwargs):
        raise AssertionError("mark loaders must not run")

    for name in ("_load_window_minute_frames", "_load_symbol_minute_frame", "_load_funding_series", "load_funding_rates"):
        monkeypatch.setattr(marks_mod, name, _boom)
    rerun, _, _, _ = fw._build_fold_target_weights("root", fold, request, funding,
        base_panel=base_panel, require_minute_roster=False, panel_warmup_hours=24)
    pd.testing.assert_frame_equal(rerun, baseline)


def test_top_level_fold_eligibility_parity(monkeypatch) -> None:
    """Top-level and fold eligibility share kwargs and masks."""

    import src.mhs.evaluation.fold_weights as fw
    import src.mhs.pipeline.stages.selection as sel
    from src.core.params import PANEL_MIN_HISTORY_BARS
    from src.mhs.pipeline.context import PipelineContext
    from src.mhs.telemetry import StageTelemetry

    idx, cols, base_panel, fold, funding = _synthetic_fold_panel()
    quote_vol = base_panel["quote_vol"]
    sel_calls: dict[str, object] = {}
    fold_calls: dict[str, object] = {}
    real_sel, real_fold = sel.liquid_half_eligibility, fw.liquid_half_eligibility
    def _sel_recording(frame, *args, **kwargs):
        sel_calls.update(kwargs)
        return real_sel(frame, *args, **kwargs)

    def _fold_recording(frame, *args, **kwargs):
        fold_calls.update(kwargs)
        return real_fold(frame, *args, **kwargs)

    monkeypatch.setattr(sel, "liquid_half_eligibility", _sel_recording)
    monkeypatch.setattr(fw, "liquid_half_eligibility", _fold_recording)
    request = research_baseline()
    ctx = PipelineContext(config=request, resolved_end=None, start=idx[0], end=idx[-1],
        rss_budget_bytes=None, rss_reserve_bytes=None, root="root", grid_1h=idx,
        close=base_panel["close"], opens=base_panel["open"], quote_vol=quote_vol,
        taker_buy_quote=None, symbols=cols)
    sel.select_horizons(ctx, StageTelemetry(log_run=False))
    fw._build_fold_target_weights("root", fold, request, funding,
        base_panel=base_panel, require_minute_roster=False, panel_warmup_hours=24)
    assert sel_calls["lookback_bars"] == fold_calls["lookback_bars"] == PANEL_MIN_HISTORY_BARS
    assert sel_calls["min_history_bars"] == fold_calls["min_history_bars"] == PANEL_MIN_HISTORY_BARS
    pd.testing.assert_frame_equal(
        real_sel(quote_vol, lookback_bars=PANEL_MIN_HISTORY_BARS, min_history_bars=PANEL_MIN_HISTORY_BARS),
        real_fold(quote_vol, lookback_bars=PANEL_MIN_HISTORY_BARS, min_history_bars=PANEL_MIN_HISTORY_BARS),
    )


def test_fold_weights_funding_gap_preservation() -> None:
    """Unknown funding coverage still blocks fold weights explicitly."""

    import src.mhs.evaluation.fold_weights as fw
    import src.core.data_policy as data_policy_mod

    idx, cols, base_panel, fold, _funding = _synthetic_fold_panel()
    request = research_baseline()
    with pytest.raises(RuntimeError, match="no fold symbol has funding coverage"):
        fw._build_fold_target_weights("root", fold, request, {}, base_panel=base_panel,
            require_minute_roster=False, panel_warmup_hours=24)
    excluded = sorted(data_policy_mod.SOURCE_GAP_EXCLUDED_SYMBOLS)[0]
    with pytest.raises(RuntimeError, match="no fold symbol has funding coverage"):
        fw._build_fold_target_weights("root", fold, request, {excluded: _funding[cols[0]].copy()},
            base_panel=base_panel, require_minute_roster=False, panel_warmup_hours=24)
