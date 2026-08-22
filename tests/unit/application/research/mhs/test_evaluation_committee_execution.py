"""MHS evaluation contract tests (split by behavioral domain; shared builders live in the original module)."""

"""Contract coverage for the MHS application evaluation resource telemetry."""
import dataclasses
import numpy as np
import pandas as pd
import pytest
from src.application.research.mhs import evaluation as ev
from src.application.research.mhs.evaluation import (
    MhsDiagnosticRequest,
)
from src.research.universe.pit_universe import symbol_partition

from tests.unit.application.research.mhs.test_evaluation import (  # noqa: F401
    _FOLD,
    _START,
    _committee_synthetic_panels,
)

def test_committee_execution_book_tranche_1_is_identity() -> None:
    # SCENARIO_COMMITTEE_EXECUTION_BOOK_TRANCHE_1_IS_IDENTITY: the default
    # tranche_count (1) returns exactly the plain mean of the committee member
    # books -- byte-identical to the pre-change implementation and to an
    # explicit tranche_count=1 call.
    from src.mhs.features import FEATURE_REGISTRY, build_feature_books

    close, quote_vol, taker_buy_quote, mask, decision_grid = _committee_synthetic_panels()
    panels = {"close": close, "quote_vol": quote_vol, "taker_buy_quote": taker_buy_quote}
    kwargs = {
        "close": close, "quote_vol": quote_vol, "taker_buy_quote": taker_buy_quote,
        "execution_mask": mask, "decision_grid": decision_grid, "min_symbols": 8,
    }
    default = ev._committee_execution_book(**kwargs)
    explicit = ev._committee_execution_book(**kwargs, tranche_count=1)
    member_specs = [s for s in FEATURE_REGISTRY if s.name in set(ev.COMMITTEE_MEMBERS)]
    books = build_feature_books(member_specs, panels, mask, decision_grid, min_symbols=8)
    assert len(books) >= 1
    reference = sum(books.values()) / float(len(books))
    pd.testing.assert_frame_equal(default, explicit)
    pd.testing.assert_frame_equal(default, reference)

def test_committee_execution_book_tranche_smooths_and_cuts_turnover() -> None:
    # SCENARIO_COMMITTEE_EXECUTION_BOOK_TRANCHE_SMOOTHS_AND_CUTS_TURNOVER: the
    # trailing decision-row mean removes repositioning -- the summed absolute
    # row-to-row change over the decision grid is strictly smaller than the
    # tranche=1 book's.
    close, quote_vol, taker_buy_quote, mask, decision_grid = _committee_synthetic_panels()
    kwargs = {
        "close": close, "quote_vol": quote_vol, "taker_buy_quote": taker_buy_quote,
        "execution_mask": mask, "decision_grid": decision_grid, "min_symbols": 8,
    }
    base = ev._committee_execution_book(**kwargs)
    smoothed = ev._committee_execution_book(**kwargs, tranche_count=3)
    raw_change = float(base.loc[decision_grid].diff().abs().sum().sum())
    smooth_change = float(smoothed.loc[decision_grid].diff().abs().sum().sum())
    assert smooth_change < raw_change

def test_committee_execution_book_tranche_preserves_dollar_neutrality() -> None:
    # SCENARIO_COMMITTEE_EXECUTION_BOOK_TRANCHE_PRESERVES_DOLLAR_NEUTRALITY:
    # every non-zero row stays dollar-neutral and the smoothing never levers up
    # -- max and mean gross of the smoothed book stay <= the raw book's. (The
    # per-row gross claim is not implied by a trailing mean: a mean's gross can
    # exceed one constituent row's gross, so the lever invariant is asserted in
    # aggregate, which is the property the fold replay measures.)
    close, quote_vol, taker_buy_quote, mask, decision_grid = _committee_synthetic_panels()
    kwargs = {
        "close": close, "quote_vol": quote_vol, "taker_buy_quote": taker_buy_quote,
        "execution_mask": mask, "decision_grid": decision_grid, "min_symbols": 8,
    }
    base = ev._committee_execution_book(**kwargs)
    smoothed = ev._committee_execution_book(**kwargs, tranche_count=3)
    non_zero = smoothed.abs().sum(axis=1) > 1e-9
    assert float(smoothed.loc[non_zero].sum(axis=1).abs().max()) < 1e-9
    raw_gross = base.abs().sum(axis=1)
    sm_gross = smoothed.abs().sum(axis=1)
    assert float(sm_gross.max()) <= float(raw_gross.max()) + 1e-9
    assert float(sm_gross.mean()) <= float(raw_gross.mean()) + 1e-9

def test_committee_execution_book_invalid_tranche_raises() -> None:
    # SCENARIO_COMMITTEE_EXECUTION_BOOK_INVALID_TRANCHE_RAISES
    close, quote_vol, taker_buy_quote, mask, decision_grid = _committee_synthetic_panels()
    with pytest.raises(ValueError, match="tranche_count"):
        ev._committee_execution_book(
            close, quote_vol, taker_buy_quote, mask, decision_grid,
            min_symbols=8, tranche_count=0,
        )

def test_committee_execution_book_no_member_still_fails_closed(monkeypatch) -> None:
    # SCENARIO_COMMITTEE_EXECUTION_BOOK_NO_MEMBER_STILL_FAILS_CLOSED: the
    # fail-closed path fires before any smoothing is applied.
    close, quote_vol, taker_buy_quote, mask, decision_grid = _committee_synthetic_panels()
    monkeypatch.setattr(ev, "build_feature_books", lambda *a, **k: {})
    with pytest.raises(RuntimeError, match="no committee member admitted"):
        ev._committee_execution_book(
            close, quote_vol, taker_buy_quote, mask, decision_grid,
            min_symbols=8, tranche_count=3,
        )

def test_committee_tranche_smoothing_requires_committee_capital() -> None:
    # SCENARIO_MHS_DIAGNOSTIC_TRANCHE_SMOOTHING_REQUIRES_COMMITTEE_CAPITAL: the
    # opt-in flag fails closed unless committee_capital is enabled and is
    # strictly bool.
    assert MhsDiagnosticRequest().committee_tranche_smoothing is False
    with pytest.raises(ValueError, match="committee_tranche_smoothing requires committee_capital"):
        MhsDiagnosticRequest(committee_tranche_smoothing=True, committee_capital=False)
    with pytest.raises(ValueError, match="committee_tranche_smoothing must be a bool"):
        MhsDiagnosticRequest(committee_tranche_smoothing="yes")
    assert (
        MhsDiagnosticRequest(committee_capital=True, committee_tranche_smoothing=True).committee_tranche_smoothing
        is True
    )

@pytest.mark.slow
def test_committee_tranche_smoothing_default_off_byte_identical(mhs_market_with_taker_buy_quote, monkeypatch) -> None:
    # SCENARIO_MHS_DIAGNOSTIC_TRANCHE_SMOOTHING_DEFAULT_OFF_BYTE_IDENTICAL:
    # with committee_capital=True and committee_tranche_smoothing omitted
    # (default False) both the fold target path and the top-level report are
    # byte-identical to an explicit committee_tranche_smoothing=False run.
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
    )
    target_default, _signal, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request, funding_by_symbol,
    )
    target_off, _signal, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, dataclasses.replace(request, committee_tranche_smoothing=False),
        funding_by_symbol,
    )
    pd.testing.assert_frame_equal(target_default, target_off)

    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    default_report = ev.run_mhs_horizon_diagnostic(request)
    explicit_off = ev.run_mhs_horizon_diagnostic(
        dataclasses.replace(request, committee_tranche_smoothing=False),
    )
    assert default_report.status == "COMPLETE"
    for field in ("books", "blend", "blend_target_gross", "research_go", "folds"):
        assert getattr(default_report, field) == getattr(explicit_off, field)

@pytest.mark.slow
def test_committee_tranche_smoothing_threads_both_call_sites(mhs_market_with_taker_buy_quote, monkeypatch) -> None:
    # SCENARIO_MHS_DIAGNOSTIC_TRANCHE_SMOOTHING_THREADS_BOTH_CALL_SITES: with
    # committee_capital=True and committee_tranche_smoothing=True the fold
    # target builder AND the top-level blend both thread tranche_count ==
    # COMMITTEE_TRANCHE_COUNT (never 1 at one site and 3 at the other).
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
        execution_universe_size=8, committee_capital=True, committee_tranche_smoothing=True,
    )
    seen: dict[str, int] = {}
    real = ev._committee_execution_book

    def _spy(*args, **kwargs):
        tranche_count = kwargs.get("tranche_count", 1)
        if len(args) > 6:
            tranche_count = args[6]
        seen["tranche_count"] = tranche_count
        return real(*args, **kwargs)

    monkeypatch.setattr(ev, "_committee_execution_book", _spy)
    ev._build_fold_target_weights(str(root), _FOLD, request, funding_by_symbol)
    assert seen["tranche_count"] == ev.COMMITTEE_TRANCHE_COUNT

    seen.clear()
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    ev.run_mhs_horizon_diagnostic(request)
    assert seen["tranche_count"] == ev.COMMITTEE_TRANCHE_COUNT

def test_committee_execution_book_regime_adaptive_differs_from_fixed_variants() -> None:
    # SCENARIO_COMMITTEE_EXECUTION_BOOK_REGIME_ADAPTIVE_DIFFERS_FROM_FIXED:
    # regime_adaptive_window selects per-row between the raw (tranche=1) book
    # and its tranche_count-row smooth, so on a fixture spanning a real
    # decision history it differs from both fixed variants at some rows.
    close, quote_vol, taker_buy_quote, mask, decision_grid = _committee_synthetic_panels()
    kwargs = {
        "close": close, "quote_vol": quote_vol, "taker_buy_quote": taker_buy_quote,
        "execution_mask": mask, "decision_grid": decision_grid, "min_symbols": 8,
    }
    fixed1 = ev._committee_execution_book(**kwargs, tranche_count=1)
    fixed3 = ev._committee_execution_book(**kwargs, tranche_count=3)
    adaptive = ev._committee_execution_book(
        **kwargs, tranche_count=3, regime_adaptive_window=ev.COMMITTEE_REGIME_ADAPTIVE_WINDOW,
    )
    assert not adaptive.equals(fixed1)
    assert not adaptive.equals(fixed3)

def test_committee_execution_book_regime_adaptive_preserves_dollar_neutrality() -> None:
    # SCENARIO_COMMITTEE_EXECUTION_BOOK_REGIME_ADAPTIVE_PRESERVES_DOLLAR_NEUTRALITY:
    # every non-zero adaptive row stays dollar-neutral (it is always exactly
    # one of the two dollar-neutral fixed variants), and aggregate gross never
    # exceeds the larger of the two fixed variants' gross.
    close, quote_vol, taker_buy_quote, mask, decision_grid = _committee_synthetic_panels()
    kwargs = {
        "close": close, "quote_vol": quote_vol, "taker_buy_quote": taker_buy_quote,
        "execution_mask": mask, "decision_grid": decision_grid, "min_symbols": 8,
    }
    fixed1 = ev._committee_execution_book(**kwargs, tranche_count=1)
    fixed3 = ev._committee_execution_book(**kwargs, tranche_count=3)
    adaptive = ev._committee_execution_book(
        **kwargs, tranche_count=3, regime_adaptive_window=ev.COMMITTEE_REGIME_ADAPTIVE_WINDOW,
    )
    non_zero = adaptive.abs().sum(axis=1) > 1e-9
    assert float(adaptive.loc[non_zero].sum(axis=1).abs().max()) < 1e-9
    max_gross = max(float(fixed1.abs().sum(axis=1).max()), float(fixed3.abs().sum(axis=1).max()))
    assert float(adaptive.abs().sum(axis=1).max()) <= max_gross + 1e-9

def test_committee_execution_book_regime_adaptive_invalid_window_raises() -> None:
    # SCENARIO_COMMITTEE_EXECUTION_BOOK_REGIME_ADAPTIVE_INVALID_WINDOW_RAISES
    close, quote_vol, taker_buy_quote, mask, decision_grid = _committee_synthetic_panels()
    with pytest.raises(ValueError, match="regime_adaptive_window"):
        ev._committee_execution_book(
            close, quote_vol, taker_buy_quote, mask, decision_grid,
            min_symbols=8, tranche_count=3, regime_adaptive_window=2,
        )

def test_committee_execution_book_regime_adaptive_no_member_still_fails_closed(
    monkeypatch,
) -> None:
    # SCENARIO_COMMITTEE_EXECUTION_BOOK_REGIME_ADAPTIVE_NO_MEMBER_STILL_FAILS_CLOSED:
    # the fail-closed path fires before any regime-adaptive selection.
    close, quote_vol, taker_buy_quote, mask, decision_grid = _committee_synthetic_panels()
    monkeypatch.setattr(ev, "build_feature_books", lambda *a, **k: {})
    with pytest.raises(RuntimeError, match="no committee member admitted"):
        ev._committee_execution_book(
            close, quote_vol, taker_buy_quote, mask, decision_grid,
            min_symbols=8, tranche_count=3, regime_adaptive_window=15,
        )

def test_committee_regime_adaptive_tranche_requires_committee_capital() -> None:
    # SCENARIO_MHS_DIAGNOSTIC_REGIME_ADAPTIVE_TRANCHE_REQUIRES_COMMITTEE_CAPITAL
    assert MhsDiagnosticRequest().committee_regime_adaptive_tranche is False
    with pytest.raises(
        ValueError, match="committee_regime_adaptive_tranche requires committee_capital",
    ):
        MhsDiagnosticRequest(committee_regime_adaptive_tranche=True, committee_capital=False)
    with pytest.raises(ValueError, match="committee_regime_adaptive_tranche must be a bool"):
        MhsDiagnosticRequest(committee_regime_adaptive_tranche="yes")
    assert (
        MhsDiagnosticRequest(
            committee_capital=True, committee_regime_adaptive_tranche=True,
        ).committee_regime_adaptive_tranche
    )

def test_committee_regime_adaptive_tranche_mutually_exclusive_with_tranche_smoothing() -> None:
    # SCENARIO_MHS_DIAGNOSTIC_REGIME_ADAPTIVE_TRANCHE_MUTUALLY_EXCLUSIVE
    with pytest.raises(
        ValueError,
        match="committee_regime_adaptive_tranche is mutually exclusive with "
        "committee_tranche_smoothing",
    ):
        MhsDiagnosticRequest(
            committee_capital=True,
            committee_regime_adaptive_tranche=True,
            committee_tranche_smoothing=True,
        )

@pytest.mark.slow
def test_committee_regime_adaptive_tranche_default_off_byte_identical(
    mhs_market_with_taker_buy_quote, monkeypatch,
) -> None:
    # SCENARIO_MHS_DIAGNOSTIC_REGIME_ADAPTIVE_TRANCHE_DEFAULT_OFF_BYTE_IDENTICAL:
    # with committee_capital=True and committee_regime_adaptive_tranche omitted
    # (default False) both the fold target path and the top-level report are
    # byte-identical to an explicit committee_regime_adaptive_tranche=False run.
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
    )
    target_default, _signal, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request, funding_by_symbol,
    )
    target_off, _signal, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, dataclasses.replace(request, committee_regime_adaptive_tranche=False),
        funding_by_symbol,
    )
    pd.testing.assert_frame_equal(target_default, target_off)

    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    default_report = ev.run_mhs_horizon_diagnostic(request)
    explicit_off = ev.run_mhs_horizon_diagnostic(
        dataclasses.replace(request, committee_regime_adaptive_tranche=False),
    )
    assert default_report.status == "COMPLETE"
    for field in ("books", "blend", "blend_target_gross", "research_go", "folds"):
        assert getattr(default_report, field) == getattr(explicit_off, field)

@pytest.mark.slow
def test_committee_regime_adaptive_tranche_threads_both_call_sites(
    mhs_market_with_taker_buy_quote, monkeypatch,
) -> None:
    # SCENARIO_MHS_DIAGNOSTIC_REGIME_ADAPTIVE_TRANCHE_THREADS_BOTH_CALL_SITES:
    # with committee_capital=True and committee_regime_adaptive_tranche=True
    # the fold target builder AND the top-level blend both thread
    # regime_adaptive_window == COMMITTEE_REGIME_ADAPTIVE_WINDOW (never
    # None at one site and set at the other).
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
        committee_regime_adaptive_tranche=True,
    )
    seen: dict[str, int | None] = {}
    real = ev._committee_execution_book

    def _spy(*args, **kwargs):
        seen["regime_adaptive_window"] = kwargs.get("regime_adaptive_window")
        return real(*args, **kwargs)

    monkeypatch.setattr(ev, "_committee_execution_book", _spy)
    ev._build_fold_target_weights(str(root), _FOLD, request, funding_by_symbol)
    assert seen["regime_adaptive_window"] == ev.COMMITTEE_REGIME_ADAPTIVE_WINDOW

    seen.clear()
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    ev.run_mhs_horizon_diagnostic(request)
    assert seen["regime_adaptive_window"] == ev.COMMITTEE_REGIME_ADAPTIVE_WINDOW

def test_committee_kelly_sizing_requires_committee_book() -> None:
    # SCENARIO_MHS_DIAGNOSTIC_COMMITTEE_KELLY_SIZING_REQUIRES_COMMITTEE_BOOK:
    # committee_kelly_sizing=True without committee_book=True fails closed in
    # __post_init__ (mirrors discovery_gate_adjusted_net_t-requires-discovery_gate).
    assert MhsDiagnosticRequest().committee_kelly_sizing is False
    with pytest.raises(ValueError, match="committee_kelly_sizing requires committee_book"):
        MhsDiagnosticRequest(committee_kelly_sizing=True, committee_book=False)
    with pytest.raises(ValueError, match="committee_kelly_sizing must be a bool"):
        MhsDiagnosticRequest(committee_kelly_sizing="yes")
    assert MhsDiagnosticRequest(committee_book=True, committee_kelly_sizing=True).committee_kelly_sizing is True

@pytest.mark.slow
def test_committee_kelly_sizing_default_off_byte_identical(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_MHS_DIAGNOSTIC_COMMITTEE_KELLY_SIZING_DEFAULT_OFF_BYTE_IDENTICAL:
    # with committee_book=True and committee_kelly_sizing omitted (default False)
    # the committee walk-forward reports sizing_mode='vol_target' -- the pure
    # pre-change vol-target path.
    root, end = mhs_market_long
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, committee_book=True,
    )
    report = ev.run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    wf = report.committee_diagnostic["walk_forward"]
    assert wf["sizing_mode"] == "vol_target"
    assert set(wf["per_tier"]) == set(ev.MEASURED_EXECUTION_COST_TIERS_BPS)

@pytest.mark.slow
def test_committee_kelly_sizing_on_changes_report(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_MHS_DIAGNOSTIC_COMMITTEE_KELLY_SIZING_ON_CHANGES_REPORT: with
    # committee_kelly_sizing=True the committee walk-forward reports
    # sizing_mode='kelly_blend' -- the opt-in 50/50 quarter-Kelly LCB overlay.
    root, end = mhs_market_long
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    base = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, committee_book=True,
    )
    report = ev.run_mhs_horizon_diagnostic(
        dataclasses.replace(base, committee_kelly_sizing=True),
    )
    assert report.status == "COMPLETE"
    assert report.committee_diagnostic["walk_forward"]["sizing_mode"] == "kelly_blend"

def test_evidence_weighting_request_validation() -> None:
    default = MhsDiagnosticRequest()
    assert default.committee_evidence_weighting is False

    valid = MhsDiagnosticRequest(committee_evidence_weighting=True, committee_capital=True)
    assert valid.committee_evidence_weighting is True

    with pytest.raises(ValueError, match="committee_capital"):
        MhsDiagnosticRequest(committee_evidence_weighting=True, committee_capital=False)

    with pytest.raises(ValueError, match="committee_evidence_weighting"):
        MhsDiagnosticRequest(committee_evidence_weighting="yes")  # type: ignore[arg-type]

def test_evidence_weights_by_boundary_builds_once(monkeypatch) -> None:
    close, quote_vol, taker_buy_quote, mask, decision_grid = _committee_synthetic_panels()
    train_ends = {
        "fold_0": pd.Timestamp("2022-01-01", tz="UTC"),
        "fold_1": pd.Timestamp("2023-01-01", tz="UTC"),
        "fold_2": pd.Timestamp("2024-01-01", tz="UTC"),
    }
    call_count = {"n": 0}
    real_build = ev.build_feature_books

    def counting_build(*args, **kwargs):
        call_count["n"] += 1
        return real_build(*args, **kwargs)

    monkeypatch.setattr(ev, "build_feature_books", counting_build)
    result = ev._committee_evidence_weights_by_boundary(
        close, quote_vol, taker_buy_quote, mask, decision_grid,
        min_symbols=8, train_ends=train_ends,
    )
    assert call_count["n"] == 1
    assert set(result.keys()) == {"fold_0", "fold_1", "fold_2"}
    for label in train_ends:
        assert isinstance(result[label], dict)
        assert len(result[label]) > 0

def test_evidence_weights_by_boundary_differentiates(monkeypatch) -> None:
    grid = pd.date_range("2021-01-01", periods=12000, freq="1h", tz="UTC")
    n_symbols = 8
    rng = np.random.default_rng(42)
    symbols = [f"S{i:02d}" for i in range(n_symbols)]
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 1e-4, (len(grid), n_symbols)), axis=0)),
        index=grid, columns=symbols,
    )
    quote_vol = pd.DataFrame(rng.uniform(900.0, 1100.0, (len(grid), n_symbols)), index=grid, columns=symbols)
    taker_buy_quote = quote_vol * rng.uniform(0.4, 0.6, (len(grid), n_symbols))
    mask = pd.DataFrame(True, index=grid, columns=symbols)
    decision_grid = pd.date_range(grid[0], grid[-1], freq="24h", tz="UTC")

    # Make two members clearly stronger by giving them positive drift
    strong_col = [c for c in close.columns if c.startswith("S0")]
    for c in strong_col:
        close[c] = 100.0 * np.exp(np.cumsum(rng.normal(2e-4, 1e-5, len(grid))))
    train_ends = {
        "early": pd.Timestamp("2021-03-01", tz="UTC"),
        "late": pd.Timestamp("2021-10-01", tz="UTC"),
    }
    result = ev._committee_evidence_weights_by_boundary(
        close, quote_vol, taker_buy_quote, mask, decision_grid,
        min_symbols=8, train_ends=train_ends,
    )
    for label in ("early", "late"):
        assert isinstance(result[label], dict)
        assert len(result[label]) > 0
    # At least one member's weight differs between the two boundaries
    common_keys = set(result["early"]) & set(result["late"])
    assert any(
        abs(result["early"][k] - result["late"][k]) > 1e-6
        for k in common_keys
    ), f"weights identical across boundaries: {result}"

def test_committee_execution_book_member_weights_none_identical() -> None:
    close, quote_vol, taker_buy_quote, mask, decision_grid = _committee_synthetic_panels()
    book_no_arg = ev._committee_execution_book(
        close, quote_vol, taker_buy_quote, mask, decision_grid,
        min_symbols=8, tranche_count=1,
    )
    book_none = ev._committee_execution_book(
        close, quote_vol, taker_buy_quote, mask, decision_grid,
        min_symbols=8, tranche_count=1, member_weights=None,
    )
    pd.testing.assert_frame_equal(book_no_arg, book_none)

def test_committee_execution_book_applies_member_weights() -> None:
    close, quote_vol, taker_buy_quote, mask, decision_grid = _committee_synthetic_panels()
    book_equal = ev._committee_execution_book(
        close, quote_vol, taker_buy_quote, mask, decision_grid,
        min_symbols=8, tranche_count=1,
    )
    # Build a weight dict that puts 0.8 on the first admitted member
    member_specs = [s for s in ev.FEATURE_REGISTRY if s.name in set(ev.COMMITTEE_MEMBERS)]
    books = ev.build_feature_books(
        member_specs,
        {"close": close, "quote_vol": quote_vol, "taker_buy_quote": taker_buy_quote},
        mask, decision_grid, min_symbols=8,
    )
    first_member = next(iter(books.keys()))
    member_weights = {first_member: 0.8}
    book_weighted = ev._committee_execution_book(
        close, quote_vol, taker_buy_quote, mask, decision_grid,
        min_symbols=8, tranche_count=1, member_weights=member_weights,
    )
    # The weighted book should be more correlated with the dominant member
    first_book_grid = books[first_member].reindex(book_equal.index).fillna(0.0)
    corr_equal = book_equal.corrwith(first_book_grid, axis=1).mean()
    corr_weighted = book_weighted.corrwith(first_book_grid, axis=1).mean()
    assert corr_weighted > corr_equal

def test_committee_execution_book_member_weights_fail_closed() -> None:
    close, quote_vol, taker_buy_quote, mask, decision_grid = _committee_synthetic_panels()
    book_equal = ev._committee_execution_book(
        close, quote_vol, taker_buy_quote, mask, decision_grid,
        min_symbols=8, tranche_count=1,
    )
    # member_weights with only keys not in admitted members
    book_mismatch = ev._committee_execution_book(
        close, quote_vol, taker_buy_quote, mask, decision_grid,
        min_symbols=8, tranche_count=1,
        member_weights={"nonexistent_member": 1.0},
    )
    pd.testing.assert_frame_equal(book_equal, book_mismatch)
    # All-zero weights also falls back to equal
    member_specs = [s for s in ev.FEATURE_REGISTRY if s.name in set(ev.COMMITTEE_MEMBERS)]
    books = ev.build_feature_books(
        member_specs,
        {"close": close, "quote_vol": quote_vol, "taker_buy_quote": taker_buy_quote},
        mask, decision_grid, min_symbols=8,
    )
    zero_weights = dict.fromkeys(books, 0.0)
    book_zero = ev._committee_execution_book(
        close, quote_vol, taker_buy_quote, mask, decision_grid,
        min_symbols=8, tranche_count=1, member_weights=zero_weights,
    )
    pd.testing.assert_frame_equal(book_equal, book_zero)

def test_fold_target_weights_threads_committee_member_weights(monkeypatch, mhs_market_with_taker_buy_quote) -> None:
    root, end = mhs_market_with_taker_buy_quote
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ][:8]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, committee_capital=True,
    )
    spy: dict = {}

    def spy_books(*args, **kwargs):
        spy["member_weights"] = kwargs.get("member_weights")
        return (None, None, None, {})

    import contextlib
    monkeypatch.setattr(ev, "_committee_execution_book", spy_books)
    # Should not raise; the spy captures the call
    with contextlib.suppress(Exception):
        ev._build_fold_target_weights(
            str(root), _FOLD, request, funding_by_symbol, None,
            committee_member_weights={"some_member": 1.0},
        )
    # The spy was called and received member_weights
    assert "member_weights" in spy
    assert spy["member_weights"] == {"some_member": 1.0}
