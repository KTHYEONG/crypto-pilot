"""MHS evaluation core tests (second-level split by domain)."""

"""MHS evaluation core contract tests (everything not in a domain-specific split file)."""
"""Contract coverage for the MHS application evaluation resource telemetry."""
import time
import types
import dataclasses
import numpy as np
import pandas as pd
import pytest
from src.application.research.mhs import evaluation as ev
import src.application.research.mhs.statistics as statistics
import src.application.research.mhs.research_go as _research_go
from src.application.research.mhs.evaluation import (
    MhsDiagnosticRequest,
)
from src.mhs.types import ExecutionSpec
from src.mhs.execution import strategy_aware_execution_replay
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

def test_xs_rank_ic_causal_forward_window_ignores_invalid_cells() -> None:
    # SCENARIO_MHS_ALPHA_ENGINE_06: the forward return is built internally as
    # opens.pct_change(forward_bars).shift(-(forward_bars + 1)); the measured
    # window starts at open_{t+1} and never overlaps the signal's own lookback.
    # With forward_bars=1, fwd[t] = (open[t+2] - open[t+1]) / open[t+1].
    index = pd.date_range("2021-01-01", periods=4, freq="1h", tz="UTC")
    opens = pd.DataFrame(
        [[100.0, 100.0, 100.0, 100.0, 100.0],
         [100.0, 100.0, 100.0, 100.0, 100.0],
         [110.0, 105.0, 100.0, 95.0, 90.0],
         [110.0, 105.0, np.nan, 95.0, 90.0]],
        index=index,
        columns=list("ABCDE"),
    )
    signal = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0],
         [1.0, 2.0, np.nan, 4.0, 5.0],
         [1.0, 1.0, 1.0, 1.0, 1.0],
         [5.0, 4.0, 3.0, 2.0, 1.0]],
        index=index,
        columns=list("ABCDE"),
    )
    result = statistics._xs_rank_ic(signal, opens, forward_bars=1)
    # Row 0 is the only valid cross section (>= 5 finite cells): ascending
    # signal ranks against the descending forward returns score IC exactly -1.
    # Row 1 has a NaN signal cell (< 5 valid cells, excluded), rows 2-3 have no
    # forward window (NaN, excluded).
    assert result["n_dates"] == 1
    assert result["mean_ic"] == pytest.approx(-1.0)
    assert result["forward_bars"] == 1

def test_xs_rank_ic_causal_window_scores_near_zero_on_unpredictable_returns() -> None:
    # SCENARIO_MHS_ALPHA_ENGINE_06: on IID returns, a signal equal to the
    # TRAILING return scores ~0 under the tradable (non-overlapping) window,
    # instead of the spuriously high overlap IC the old trailing convention
    # reported (+0.0957 vs the tradable -0.0278 in the spec).
    rng = np.random.default_rng(7)
    n_hours, n_syms = 60, 10
    index = pd.date_range("2021-01-01", periods=n_hours, freq="1h", tz="UTC")
    cols = [f"S{i}" for i in range(n_syms)]
    opens = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, (n_hours, n_syms)), axis=0)),
        index=index,
        columns=cols,
    )
    signal = opens.pct_change()
    result = statistics._xs_rank_ic(signal, opens, forward_bars=1)
    assert result["n_dates"] > 20
    assert abs(result["mean_ic"]) < 0.3
    assert result["forward_bars"] == 1
    with pytest.raises(ValueError, match="forward_bars"):
        statistics._xs_rank_ic(signal, opens, forward_bars=0)

def test_date_clustered_ols_causal_forward_window() -> None:
    # SCENARIO_MHS_ALPHA_ENGINE_06: the pooled panel regression builds its
    # dependent variable internally with shift(-(forward_bars + 1)); with
    # forward_bars=1 the forward window is fwd[t] = r[t + 2], and step returns
    # r[t] = 1.5 * past[t - 2] + 0.25 recover the known 1.5 slope exactly.
    index = pd.date_range("2021-01-01", periods=48, freq="1h", tz="UTC")
    past = pd.DataFrame(
        {"A": np.arange(48, dtype=float), "B": np.arange(48, dtype=float) + 2.0},
        index=index,
    )
    step_ret = np.vstack(
        [np.zeros((2, past.shape[1])), (0.25 + 1.5 * past.iloc[:-2]).to_numpy()],
    )
    opens = pd.DataFrame(100.0 * np.cumprod(1.0 + step_ret, axis=0), index=index, columns=past.columns)
    opens.iloc[3, 0] = np.nan

    result = statistics._date_clustered_ols(opens, past, forward_bars=1)
    # The NaN at opens[3, "A"] poisons the pct_change for two forward cells
    # (fwd[1] reads r[3], fwd[2] divides by open[3]); the last two bars have no
    # forward window, so 96 - 4 (terminal) - 2 (poisoned) = 90 finite pairs.
    assert result["n"] == 90
    assert result["n_dates"] == 2
    assert result["past_beta"] == pytest.approx(1.5, rel=1e-3)
    assert result["forward_bars"] == 1
    with pytest.raises(ValueError, match="forward_bars"):
        statistics._date_clustered_ols(opens, past, forward_bars=0)

def test_mhs_perf_opt_001_placebo_vectorized_exact_and_fast() -> None:
    # PERF_OPT_001_PLACEBO_VECTORIZED: the vectorized NumPy placebo must
    # reproduce the baseline percentile exactly and run >= 5x faster.
    signal, eligible, opens, bar_funding, grid, spec = _perf_opt_placebo_inputs(20260807)
    n_placebos = 300
    for observed in (0.7, -1.5, 0.0):
        expected = _reference_placebo_percentile(
            signal, eligible, opens, bar_funding, grid, spec, observed, n_placebos, 7,
        )
        actual = statistics._placebo_sharpe_percentile(
            signal, eligible, opens, bar_funding, grid, spec, observed, n_placebos, 7,
        )
        assert (expected is None and actual is None) or (expected == actual)

    t0 = time.perf_counter()
    _reference_placebo_percentile(
        signal, eligible, opens, bar_funding, grid, spec, 0.7, n_placebos, 7,
    )
    reference_elapsed = time.perf_counter() - t0
    t1 = time.perf_counter()
    statistics._placebo_sharpe_percentile(
        signal, eligible, opens, bar_funding, grid, spec, 0.7, n_placebos, 7,
    )
    vectorized_elapsed = time.perf_counter() - t1
    assert vectorized_elapsed < reference_elapsed / 5.0

def test_mhs_perf_opt_002_participation_cumsum_exact(tmp_path) -> None:
    # PERF_OPT_002_PARTICIPATION_CUMSUM: the cumsum/searchsorted rewrite
    # must return the exact same warnings dict as the iterrows() baseline.
    symbols = ["SYMA", "SYMB", "SYMC"]
    grid, _ = _write_quote_volume_market(tmp_path, symbols)
    rng = np.random.default_rng(42)
    rows = []
    for i, sym in enumerate(symbols):
        # Minute-aligned fills inside the quote-volume window, some off-grid.
        ts = grid[200 + i::(600 + i * 5)].to_list()
        for j, t in enumerate(ts[:40]):
            rows.append(
                {
                    "timestamp": t,
                    "symbol": sym,
                    "quantity_delta": 0.5 if j % 2 == 0 else -0.5,
                    "fill_price": 100.0 + rng.normal(0.0, 1.0),
                }
            )
    fills = pd.DataFrame(rows)
    replay = types.SimpleNamespace(simulated_fills=fills)

    expected = _reference_participation_warnings(
        replay, str(tmp_path), "1m", symbols, grid,
    )
    actual = ev._participation_warnings(replay, str(tmp_path), "1m", symbols, grid)
    assert set(actual) == set(expected)
    for key in expected:
        assert actual[key] == expected[key]

def test_mhs_perf_opt_003_bootstrap_vectorized_equivalent() -> None:
    # PERF_OPT_003_BOOTSTRAP_VECTORIZED: 2D block sampling must produce
    # statistically equivalent CI bounds (the RNG draw order differs by design,
    # so exact reproduction is neither required nor possible).
    rng = np.random.default_rng(5)
    net = pd.Series(np.cumsum(rng.normal(0.0, 0.01, 400)))
    for seed in (20260807, 3, 11):
        lo_ref, hi_ref = _reference_bootstrap_ci(net, 800, 24, seed)
        lo_new, hi_new = statistics._bootstrap_ci(net, 800, 24, seed)
        assert lo_new < hi_new
        assert lo_ref < hi_ref
        assert abs(lo_new - lo_ref) < 0.05
        assert abs(hi_new - hi_ref) < 0.05

def test_mhs_phase2_o10_bootstrap_chunk_adaptive() -> None:
    # SCENARIO_O10_RSS_GATE: chunk is capped so a (chunk, n) sample matrix stays
    # <= 128MB; at production 5m scale (525,600 bars) that means a small chunk.
    from src.mhs.evidence import _bootstrap_chunk_size

    assert _bootstrap_chunk_size(525_600) <= 63
    assert _bootstrap_chunk_size(43_830) >= 100
    assert _bootstrap_chunk_size(0) == 500

@pytest.mark.slow
def test_horizon_diagnostics_exposes_effective_breadth(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_HORIZON_DIAGNOSTICS_EXPOSES_EFFECTIVE_BREADTH_04: with
    # discovery_gate=True run_mhs_horizon_diagnostic reports finite
    # slow_horizon_effective_breadth/fast_horizon_effective_breadth within
    # [1.0, nominal_candidate_count]; with discovery_gate=False (the default)
    # the two keys are absent -- opt-in, no default-path cost.
    root, end = mhs_market
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    request_on = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, discovery_gate=True,
    )
    report_on = ev.run_mhs_horizon_diagnostic(request_on)
    assert report_on.status == "COMPLETE"
    slow_n_eff = report_on.horizon_diagnostics.get("slow_horizon_effective_breadth")
    fast_n_eff = report_on.horizon_diagnostics.get("fast_horizon_effective_breadth")
    assert slow_n_eff is not None
    assert np.isfinite(slow_n_eff)
    assert fast_n_eff is not None
    assert np.isfinite(fast_n_eff)
    assert 1.0 <= slow_n_eff <= 19
    assert 1.0 <= fast_n_eff <= 7

    request_off = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8,
    )
    report_off = ev.run_mhs_horizon_diagnostic(request_off)
    assert report_off.status == "COMPLETE"
    assert "slow_horizon_effective_breadth" not in report_off.horizon_diagnostics
    assert "fast_horizon_effective_breadth" not in report_off.horizon_diagnostics

def test_naive_sharpe_uses_hourly_annualization() -> None:
    # SCENARIO_MHS_ANNUALIZATION_01: on a synthetic 5-minute ledger the
    # hourly-resampled naive Sharpe agrees with a calendar-correct manual
    # computation, while the pre-fix computation (sqrt(_PERIODS_PER_YEAR_1H)
    # applied to the raw 5-minute returns) understates it by ~sqrt(12).
    n_years = 3.0
    n_bars = round(365.25 * 288 * n_years)
    ledger = _synthetic_ledger("5min", n_bars, mean_ret=0.0004, vol_ret=0.001, seed=7)
    net_1h = ledger.equity.resample("1h").last().dropna().pct_change().dropna()
    ref = float(net_1h.mean() / net_1h.std(ddof=1) * np.sqrt(ev._PERIODS_PER_YEAR_1H))
    assert statistics._naive_sharpe(ledger) == pytest.approx(ref)
    net_5m = ledger.net_returns
    pre_fix = float(net_5m.mean() / net_5m.std(ddof=1) * np.sqrt(ev._PERIODS_PER_YEAR_1H))
    assert ref / pre_fix == pytest.approx(np.sqrt(12.0), rel=0.05)

def test_hourly_ledger_series_hourly_input_is_identity() -> None:
    # SCENARIO_MHS_ANNUALIZATION_02: on an already-hourly ledger the helper is
    # byte-identical up to the leading NaN drop on pct_change, so the existing
    # hourly synthetic-fixture tests keep passing untouched.
    idx = pd.date_range("2021-01-01", periods=48, freq="1h", tz="UTC")
    equity = pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx)
    turnover = pd.Series(np.linspace(0.01, 0.02, len(idx)), index=idx)
    eq_1h, net_1h, turn_1h = statistics._hourly_ledger_series(equity, turnover)
    pd.testing.assert_series_equal(eq_1h, equity)
    pd.testing.assert_series_equal(net_1h, equity.pct_change().dropna())
    pd.testing.assert_series_equal(turn_1h, turnover.iloc[1:].rename(None))

def test_hourly_ledger_series_5m_returns_one_row_per_hour() -> None:
    # SCENARIO_MHS_ANNUALIZATION_02 (5-minute leg): one row per calendar hour,
    # turnover is summed (not last-sampled), and the return series drops the
    # leading NaN of the equity pct_change.
    idx = pd.date_range("2021-01-01", periods=72 * 12, freq="5min", tz="UTC")
    equity = pd.Series(np.linspace(100.0, 130.0, len(idx)), index=idx)
    turnover = pd.Series(np.full(len(idx), 0.01), index=idx)
    eq_1h, net_1h, turn_1h = statistics._hourly_ledger_series(equity, turnover)
    assert len(eq_1h) == len(net_1h) + 1
    assert len(eq_1h) == 72
    assert (eq_1h.index.minute == 0).all()
    assert (eq_1h.index.second == 0).all()
    pd.testing.assert_series_equal(
        eq_1h, equity.resample("1h").last().dropna(),
    )
    assert turn_1h.index.equals(net_1h.index)
    np.testing.assert_allclose(turn_1h.to_numpy(), np.full(len(turn_1h), 12 * 0.01))

def test_hourly_ledger_series_empty_input_is_empty() -> None:
    # SCENARIO_MHS_ANNUALIZATION_02 (edge case): an empty equity series must
    # raise no exception and return three empty Series, mirroring the caller's
    # empty-input nan convention.
    eq = pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"))
    turn = pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"))
    eq_1h, net_1h, turn_1h = statistics._hourly_ledger_series(eq, turn)
    assert eq_1h.empty
    assert net_1h.empty
    assert turn_1h.empty

def test_geometric_cagr_uses_hourly_annualization() -> None:
    # SCENARIO_MHS_ANNUALIZATION_03: on a synthetic 5-minute equity with known
    # total-return ratio R over T years, _geometric_cagr on the C2 hourly series
    # equals R**(1/T)-1, while the raw 5-minute call (the pre-fix path) is off by
    # exactly the 12x (5m) bar-count multiple in the exponent.
    n_years = 3.0
    n_bars = round(365.25 * 288 * n_years)
    idx = pd.date_range("2021-01-01", periods=n_bars, freq="5min", tz="UTC")
    rng = np.random.default_rng(11)
    rets = rng.normal(1.5e-6, 0.001, n_bars)
    equity = pd.Series(np.cumprod(1.0 + rets), index=idx)
    # Flatten the first hour so the hourly resample's opening close equals the
    # raw series' opening close, making the pre/post exponent ratio exactly 12.
    equity.iloc[1:12] = equity.iloc[0]
    turnover = pd.Series(np.zeros(n_bars), index=idx)
    eq_1h, _net_1h, _turn_1h = statistics._hourly_ledger_series(equity, turnover)
    # True CAGR over the hourly span: ratio from the first to the last hourly
    # close, spanning (n_hours - 1) hourly intervals (the code annualizes with
    # n_hours bars, an O(1/n) approximation).
    ratio_h = float(eq_1h.iloc[-1] / eq_1h.iloc[0])
    span_years = (len(eq_1h) - 1) / ev._PERIODS_PER_YEAR_1H
    assert statistics._geometric_cagr(eq_1h) == pytest.approx(
        ratio_h ** (1.0 / span_years) - 1.0, rel=1e-3,
    )
    post = statistics._geometric_cagr(eq_1h)
    pre = statistics._geometric_cagr(equity)
    assert np.log1p(post) / np.log1p(pre) == pytest.approx(12.0)

def test_research_go_eligible_is_reachable(mhs_market, monkeypatch) -> None:
    """SCENARIO_MHS_RESEARCH_GO_ELIGIBLE_IS_REACHABLE_07: with every fold
    passing and every policy threshold registered in
    ``REGISTERED_POLICY_THRESHOLDS``, ``_mhs_research_go`` returns
    eligible=True with no reason codes -- a result the pre-change code (which
    unconditionally appended UNSPECIFIED_POLICY) could never produce. With a
    threshold missing it still fails closed to UNSPECIFIED_POLICY."""
    idx = pd.date_range("2021-01-01 12:01", periods=31, freq="1min", tz="UTC")
    target = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 11:00", tz="UTC")])
    signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 12:00", tz="UTC")])
    px = pd.DataFrame({"A": [100.0] * 31}, index=idx)
    replay = strategy_aware_execution_replay(
        target, signal_at, px, px, px, px,
        pd.DataFrame(0.0, index=idx, columns=["A"]), 1.0,
        "OHLCV_STRICT_PROXY", ExecutionSpec(),
    )
    passing = _passing_fold_report(replay)

    monkeypatch.setattr(
        _research_go, "REGISTERED_POLICY_THRESHOLDS",
        {"cap_30_roster": 30.0, "primary_annual_return": 0.05},
    )
    registered = _research_go._mhs_research_go((passing,))
    assert registered.eligible is True
    assert registered.reason_codes == ()
    assert registered.folds_passed == 1

    monkeypatch.setattr(
        _research_go, "REGISTERED_POLICY_THRESHOLDS",
        {"cap_30_roster": None, "primary_annual_return": 0.05},
    )
    missing = _research_go._mhs_research_go((passing,))
    assert missing.eligible is False
    assert ev.GO_REASON_UNSPECIFIED_POLICY in missing.reason_codes

def test_research_go_data_integrity_reason_split(monkeypatch) -> None:
    """SCENARIO_MHS_RESEARCH_GO_DATA_INTEGRITY_REASON_SPLIT: a fold failing on
    both a relevant execution data gap and a pure alpha-quality Sharpe failure
    carries only the data-integrity code in ``data_integrity_reason_codes``
    while ``reason_codes`` keeps both axes separate."""
    passing = _passing_fold_report(_gap_mixed_replay())
    mixed = dataclasses.replace(
        passing,
        failures=(ev.GO_REASON_EXECUTION_GAP, ev.GO_REASON_PRIMARY_SHARPE),
        termination_counts={"MISSING_DATA": 3, "UNKNOWN_TERMINATION": 0},
        primary_autocorr_sharpe=0.3,
    )
    monkeypatch.setattr(
        _research_go, "REGISTERED_POLICY_THRESHOLDS",
        {"cap_30_roster": 30.0, "primary_annual_return": 0.05},
    )
    go = _research_go._mhs_research_go((mixed,))
    assert go.data_integrity_reason_codes == (ev.GO_REASON_EXECUTION_GAP,)
    assert ev.GO_REASON_PRIMARY_SHARPE not in go.data_integrity_reason_codes
    assert set(go.reason_codes) == {
        ev.GO_REASON_EXECUTION_GAP, ev.GO_REASON_PRIMARY_SHARPE,
    }
    assert go.eligible is False

def test_research_go_data_integrity_reason_empty_when_clean(monkeypatch) -> None:
    """SCENARIO_MHS_RESEARCH_GO_DATA_INTEGRITY_REASON_EMPTY_WHEN_CLEAN: a pure
    alpha-quality failure (primary Sharpe below floor, no data gap) yields
    ``data_integrity_reason_codes == ()`` even though eligible is False -- the
    consumer distinguishes "data intact, alpha underperformed" from "data was
    deficient" by that empty field."""
    passing = _passing_fold_report(_gap_mixed_replay())
    alpha_only = dataclasses.replace(
        passing,
        failures=(ev.GO_REASON_PRIMARY_SHARPE,),
        primary_autocorr_sharpe=0.3,
    )
    monkeypatch.setattr(
        _research_go, "REGISTERED_POLICY_THRESHOLDS",
        {"cap_30_roster": 30.0, "primary_annual_return": 0.05},
    )
    go = _research_go._mhs_research_go((alpha_only,))
    assert go.eligible is False
    assert go.data_integrity_reason_codes == ()
    assert go.reason_codes == (ev.GO_REASON_PRIMARY_SHARPE,)

def test_registered_policy_thresholds_contract() -> None:
    """SCENARIO_MHS_POLICY_THRESHOLDS_REGISTERED_VALUES: the two named policy
    gates exist in source contracts and are registered at their reviewed
    2026-08-17 values (docs/specs/mhs_research_go_policy_registration.md) --
    cap_30_roster mirrors the frozen execution_universe_size design cap
    (attestation only), primary_annual_return is enforced per anchored fold."""
    from src.mhs.types import REGISTERED_POLICY_THRESHOLDS, SEARCH_TRIALS_ATTEMPTED

    assert REGISTERED_POLICY_THRESHOLDS == {
        "cap_30_roster": 30.0, "primary_annual_return": 0.05,
    }
    assert isinstance(SEARCH_TRIALS_ATTEMPTED, int)
    assert SEARCH_TRIALS_ATTEMPTED >= 1
