"""MHS evaluation contract tests (split by behavioral domain; shared builders live in the original module)."""

"""Contract coverage for the MHS application evaluation resource telemetry."""
import numpy as np
import pytest
from src.mhs import evaluation as ev
from src.mhs.diagnostic_run import run_mhs_horizon_diagnostic
from src.mhs.evaluation import (
    MhsDiagnosticRequest,
)
from src.quant.universe.pit_universe import symbol_partition

from tests.unit.mhs.test_evaluation_appresearch import (  # noqa: F401
    _FOLD,
    _START,
    _admitted_selection,
)

def test_fold_safe_slow_book_spec_admitted_vs_fallback() -> None:
    # SCENARIO_MHS_FOLD_SAFE_HORIZON_05_BOOK_SPEC_HELPER_ADMITTED_VS_FALLBACK:
    # the fold-spec resolver returns the unchanged frozen default (168,
    # "frozen_default") unless the fold-scoped gate both admitted AND selected
    # a candidate; only then does it build a BookSpec whose horizon is the
    # selected candidate with band/step_hours/min_symbols identical to the
    # default.
    default = ev.BOOK_SPECS["slow_momentum"]
    fallback = ev.DiscoveryQualificationResult(
        selected_horizon=None, admitted=False, discovery_scores=(),
        discovery_aggregate_net_t=None, qualification_net_t=None,
        qualification_sign_consistent=None,
    )
    spec, horizon, source = ev._fold_safe_slow_book_spec(fallback, default)
    assert spec is default
    assert horizon == 168
    assert source == "frozen_default"

    admitted_none = _admitted_selection(selected_horizon=None)
    spec, horizon, source = ev._fold_safe_slow_book_spec(admitted_none, default)
    assert spec is default
    assert source == "frozen_default"

    admitted = _admitted_selection(360)
    spec, horizon, source = ev._fold_safe_slow_book_spec(admitted, default)
    assert source == "fold_train_only_discovery"
    assert horizon == 360
    assert spec is not default
    assert spec.horizon_hours == 360
    assert spec.band is default.band
    assert spec.step_hours == default.step_hours
    assert spec.min_symbols == default.min_symbols

@pytest.mark.slow
def test_fold_safe_horizon_flag_off_is_byte_identical(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_FOLD_SAFE_HORIZON_06_FLAG_OFF_IS_BYTE_IDENTICAL: with
    # fold_safe_horizon_selection=False (the default) neither the fold worker
    # nor the parent diagnostic touches fold_train_only_discovery_qualification
    # (call-count 0) and the fold report records the frozen 168h default -- a
    # no-op regression guard matching the project's flag-gated ADR pattern.
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
    assert request.fold_safe_horizon_selection is False

    calls = {"n": 0}
    real_fn = ev.fold_train_only_discovery_qualification

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(ev, "fold_train_only_discovery_qualification", counting)
    report = ev._run_anchored_fold(str(root), _FOLD, request, funding_by_symbol, 1.0, 0, None)
    assert calls["n"] == 0
    assert report.slow_horizon_hours == 168
    assert report.slow_horizon_source == "frozen_default"

    # Parent path: the default request passes an empty fold_slow_horizons dict
    # into the fold pool and never runs the fold-scoped selection.
    captured: dict = {}

    def _spy_books(*args, **kwargs):
        return (None, None, None, {}, None)

    def _spy_post(*args, **kwargs):
        captured["fold_slow_horizons"] = args[14] if len(args) > 14 else None
        return (None, None, {}, {}, (), None)

    monkeypatch.setattr(ev, "_run_books_concurrent", _spy_books)
    monkeypatch.setattr(ev, "_run_post_book_concurrently", _spy_post)
    top_report = run_mhs_horizon_diagnostic(request)
    assert top_report.status == "COMPLETE"
    assert calls["n"] == 0
    assert captured["fold_slow_horizons"] == {}

@pytest.mark.slow
def test_fold_safe_horizon_records_source(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_FOLD_SAFE_HORIZON_07_FOLD_REPORT_RECORDS_SOURCE: MhsFoldReport
    # constructed without the new fields defaults to (168, "frozen_default"),
    # _incomplete_fold_report keeps that default, and a fold run resolved with a
    # 360h fold-scoped override records slow_horizon_hours==360 with source
    # "fold_train_only_discovery".
    default_report = ev.MhsFoldReport(
        fold_index=0, validation_start="2021-02-10", validation_end="2021-04-19",
        strict=None, stress=None, primary_valid=False, primary_autocorr_sharpe=0.0,
        primary_naive_sharpe=0.0, primary_net_ann=0.0, primary_geometric_cagr=0.0,
        primary_max_drawdown=0.0, stress_naive_sharpe=0.0, decision_intents=0,
        termination_counts={}, failures=(), strict_elapsed_seconds=0.0,
        stress_elapsed_seconds=0.0,
    )
    assert default_report.slow_horizon_hours == 168
    assert default_report.slow_horizon_source == "frozen_default"

    incomplete = ev._incomplete_fold_report(_FOLD, 0, ())
    assert incomplete.slow_horizon_source == "frozen_default"
    assert incomplete.slow_horizon_hours == 168

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
        str(root), _FOLD, request, funding_by_symbol, 1.0, 0, None, slow_horizon_override=360,
    )
    assert report.slow_horizon_hours == 360
    assert report.slow_horizon_source == "fold_train_only_discovery"

    # Parent wiring: with the flag on, the parent runs the fold-scoped selection
    # once per fold and threads only the resolved plain int down to the fold
    # pool; the top-level slow spec adopts fold 2's selection (360h here).
    captured: dict = {}

    def _admit_by_family(*args, **kwargs):
        # The funding-carry family's selected lookback must come from its own
        # measured grid; the slow/fast families keep the 360h selection.
        if kwargs.get("horizon_candidates") == ev.FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS:
            return _admitted_selection(72)
        return _admitted_selection(360)

    monkeypatch.setattr(ev, "fold_train_only_discovery_qualification", _admit_by_family)

    def _spy_books(*args, **kwargs):
        captured["top_level_slow"] = args[5]
        return (None, None, None, {})

    def _spy_post(*args, **kwargs):
        captured["fold_slow_horizons"] = args[14] if len(args) > 14 else None
        captured["fold_fast_horizons"] = args[15] if len(args) > 15 else None
        return (None, None, {}, {}, (), None)

    monkeypatch.setattr(ev, "_run_books_concurrent", _spy_books)
    monkeypatch.setattr(ev, "_run_post_book_concurrently", _spy_post)
    request_on = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, fold_safe_horizon_selection=True,
    )
    top_report = run_mhs_horizon_diagnostic(request_on)
    assert top_report.status == "COMPLETE"
    assert captured["fold_slow_horizons"] == {0: 360, 1: 360, 2: 360, 3: 360}
    # The fast re-verification is diagnostic-only: the parent threads the
    # resolved (horizon, source) pairs to the fold pool but never alters the
    # top-level fast spec (still the frozen 48h default, and blend weights
    # stay 0.0).
    assert captured["fold_fast_horizons"] == {
        0: (360, "fold_train_only_discovery"),
        1: (360, "fold_train_only_discovery"),
        2: (360, "fold_train_only_discovery"),
        3: (360, "fold_train_only_discovery"),
    }
    assert captured["top_level_slow"].horizon_hours == 360
    assert captured["top_level_slow"].band is ev.BOOK_SPECS["slow_momentum"].band

@pytest.mark.slow
def test_fold_safe_horizon_builds_candidate_weights_once_and_shares_across_folds(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_HORIZON_SEARCH_EFF_05_TOP_LEVEL_WIRING_SHARES_ONE_CACHE_ACROSS_FOLDS:
    # with fold_safe_horizon_selection=True the parent precomputes every
    # discovery weight book exactly once (``_candidate_weight_books``) and every
    # fold's gate reuses that single precompute (fork-inherited copy-on-write in
    # the parallel fold-safe path) -- the measured 3x-redundant weight
    # construction is eliminated without changing any value. The
    # parallel/sequential value-equivalence is pinned by
    # ``test_mhs_perf_opt_fold_discovery_parallel_equivalence``.
    root, end = mhs_market
    calls = {"n": 0}
    real_builder = ev._candidate_weight_books

    def counting_builder(*args, **kwargs):
        calls["n"] += 1
        return real_builder(*args, **kwargs)

    monkeypatch.setattr(ev, "_candidate_weight_books", counting_builder)

    def _spy_books(*args, **kwargs):
        return (None, None, None, {}, None)

    def _spy_post(*args, **kwargs):
        return (None, None, {}, {}, (), None)

    monkeypatch.setattr(ev, "_run_books_concurrent", _spy_books)
    monkeypatch.setattr(ev, "_run_post_book_concurrently", _spy_post)
    request_on = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, fold_safe_horizon_selection=True,
    )
    top_report = run_mhs_horizon_diagnostic(request_on)
    assert top_report.status == "COMPLETE"
    assert calls["n"] == 1

@pytest.mark.slow
def test_fold_safe_funding_carry_parent_wiring(mhs_market_funding_vary, monkeypatch) -> None:
    # SCENARIO_MHS_FOLD_REPORT_CARRIES_FUNDING_CARRY_DISCOVERY_05 (parent path):
    # with fold_safe_horizon_selection=True and a funding-carry admission the
    # parent threads (lookback, sign, source, corr) per fold with a finite
    # train-window orthogonality correlation against slow_momentum; when no
    # candidate admits, all four fields fail closed to frozen_default/None.
    root, end = mhs_market_funding_vary
    captured: dict = {}

    def _run(captured):
        monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))

        def _spy_post(*args, **kwargs):
            captured["fold_funding_carry"] = args[16] if len(args) > 16 else None
            return (None, None, {}, {}, (), None)

        monkeypatch.setattr(ev, "_run_post_book_concurrently", _spy_post)
        request_on = MhsDiagnosticRequest(
            start=str(_START), end=str(end), data_root=str(root),
            mark_mode="cache_required", execution_timeframe="1m", log_run=False,
            execution_universe_size=8, fold_safe_horizon_selection=True,
        )
        top_report = run_mhs_horizon_diagnostic(request_on)
        assert top_report.status == "COMPLETE"
        return captured["fold_funding_carry"]

    def _admit_funding_only(*args, **kwargs):
        if kwargs.get("horizon_candidates") == ev.FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS:
            return _admitted_selection(72)
        return _admitted_selection(None)

    monkeypatch.setattr(ev, "fold_train_only_discovery_qualification", _admit_funding_only)
    admitted = _run(captured)
    assert set(admitted) == {0, 1, 2, 3}
    for lookback, sign, source, corr in admitted.values():
        assert lookback == 72
        assert sign == 1
        assert source == "fold_train_only_discovery"
        assert np.isfinite(corr)

    captured.clear()
    monkeypatch.setattr(
        ev, "fold_train_only_discovery_qualification",
        lambda *a, **k: _admitted_selection(None),
    )
    fail_closed = _run(captured)
    assert set(fail_closed) == {0, 1, 2, 3}
    for lookback, sign, source, corr in fail_closed.values():
        assert lookback is None
        assert sign is None
        assert source == "frozen_default"
        assert corr is None
