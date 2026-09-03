"""MHS evaluation contract tests (split by behavioral domain; shared builders live in the original module)."""

"""Contract coverage for the MHS application evaluation resource telemetry."""
import numpy as np
import pandas as pd
import pytest
from src.mhs import evaluation as ev
from src.mhs.diagnostic_run import run_mhs_horizon_diagnostic
import src.mhs.statistics as statistics
from src.mhs.evaluation import (
    MhsDiagnosticRequest,
)

from tests.unit.mhs.test_evaluation_appresearch import (  # noqa: F401
    _START,
)

def test_multi_feature_request_validation() -> None:
    # SCENARIO_MHS_REQUEST_MULTI_FEATURE_VALIDATION: MhsDiagnosticRequest gains
    # multi_feature_book (bool, default False). A non-bool value raises
    # ValueError (fail closed -- no silent no-op); the default construction
    # leaves it False and the report's multi_feature_diagnostic is None.
    assert MhsDiagnosticRequest().multi_feature_book is False
    with pytest.raises(ValueError, match="multi_feature_book"):
        MhsDiagnosticRequest(multi_feature_book="yes")
    on = MhsDiagnosticRequest(multi_feature_book=True)
    assert on.multi_feature_book is True

@pytest.mark.slow
def test_multi_feature_default_off_bit_identical(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_MULTI_FEATURE_DEFAULT_OFF_BIT_IDENTICAL: with the flag
    # omitted (multi_feature_book=False) the report's multi_feature_diagnostic
    # is None and every pre-existing field is bit-identical to the explicit-off
    # baseline -- the multi-feature axis is inert unless explicitly enabled.
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
        MhsDiagnosticRequest(**base, multi_feature_book=False),
    )
    assert default_report.status == "COMPLETE"
    assert default_report.multi_feature_diagnostic is None
    assert explicit_off.multi_feature_diagnostic is None
    for field in ("books", "blend", "blend_target_gross", "research_go", "folds"):
        assert getattr(default_report, field) == getattr(explicit_off, field)

@pytest.mark.slow
def test_multi_feature_diagnostic_reports_coverage_and_stability(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_MULTI_FEATURE_DIAGNOSTIC_REPORTS_COVERAGE_AND_STABILITY:
    # with multi_feature_book=True the report's multi_feature_diagnostic dict
    # carries, per admitted feature, its per-year coverage and its regime-split
    # stability fields, plus the combined book's net Sharpe per measured cost
    # tier and the effective breadth of the feature-book PnL panel; features
    # excluded by the coverage gate are listed under an explicit excluded key
    # with their failing year, never silently dropped.
    root, end = mhs_market
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, multi_feature_book=True,
    )
    report = run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    diag = report.multi_feature_diagnostic
    assert isinstance(diag, dict)
    admitted = diag["admitted"]
    assert isinstance(admitted, dict)
    assert admitted, "the fixture's feature columns must admit at least one feature"
    for feature_name, fields in admitted.items():
        assert isinstance(feature_name, str)
        coverage = fields["coverage"]
        assert isinstance(coverage, dict)
        for value in coverage.values():
            assert 0.0 <= value <= 1.0
        stability = fields["regime_split_stability"]
        assert isinstance(stability, dict)
        for label, sharpe in stability["window_sharpes"]:
            assert isinstance(label, str)
            assert sharpe is None or np.isfinite(sharpe)
        assert stability["min_window_sharpe"] is None or np.isfinite(
            stability["min_window_sharpe"]
        )
        assert isinstance(stability["sign_consistent"], bool)
        assert stability["decay"] is None or np.isfinite(stability["decay"])
    excluded = diag["excluded"]
    assert isinstance(excluded, dict)
    for fields in excluded.values():
        assert "failing_year" in fields
    combined = diag["combined"]
    assert set(combined["net_sharpe_per_tier"]) == set(ev.MEASURED_EXECUTION_COST_TIERS_BPS)
    for value in combined["net_sharpe_per_tier"].values():
        assert value is None or np.isfinite(value)
    breadth = diag["feature_book_effective_breadth"]
    assert isinstance(breadth, dict)
    assert "n_eff" in breadth
    assert "mean_corr" in breadth
    assert np.isfinite(breadth["n_eff"])
    assert np.isfinite(breadth["mean_corr"])

@pytest.mark.slow
def test_multi_feature_diagnostic_telemetry_stages_recorded(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_MHS_MULTI_FEATURE_TELEMETRY_STAGE_RECORDED: with
    # multi_feature_book=True the resource_measurements carry the diagnostic
    # feature panel load and the multi-feature diagnostic stage.
    root, end = mhs_market_long
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, multi_feature_book=True,
    )
    report = run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    stages = {m.stage for m in report.resource_measurements}
    assert "diagnostic_feature_panels" in stages
    assert "multi_feature_diagnostic" in stages

def test_multi_feature_streaming_combined_bit_identical() -> None:
    # SCENARIO_MHS_MULTI_FEATURE_STREAMING_BIT_IDENTICAL: the streaming
    # multi-feature diagnostic produces combined.book_mean_gross,
    # combined.net_sharpe_per_tier and feature_book_effective_breadth EXACTLY
    # equal to a batch reference built from the same panels with the existing
    # primitives (build_feature_books + mhs_ledger_pnl + equal_risk_combination).
    from src.mhs.execution import mhs_ledger_pnl
    from src.mhs.features import (
        FEATURE_REGISTRY,
        build_feature_books,
        equal_risk_combination,
        feature_coverage_audit,
    )

    grid = pd.date_range("2021-01-01", periods=2400, freq="1h", tz="UTC")
    symbols = [f"S{i:02d}" for i in range(10)]
    rng = np.random.default_rng(9)
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 1e-4, (len(grid), len(symbols))), axis=0)),
        index=grid, columns=symbols,
    )
    quote_vol = pd.DataFrame(rng.uniform(900.0, 1100.0, (len(grid), len(symbols))), index=grid, columns=symbols)
    taker_buy_quote = quote_vol * rng.uniform(0.4, 0.6, (len(grid), len(symbols)))
    panels = {
        "close": close,
        "high": close * 1.001,
        "low": close * 0.999,
        "quote_vol": quote_vol,
        "taker_buy_quote": taker_buy_quote,
        "no_trades": pd.DataFrame(1000, index=grid, columns=symbols),
    }
    mask = pd.DataFrame(True, index=grid, columns=symbols)
    decision_grid = pd.date_range(grid[0], grid[-1], freq="24h", tz="UTC")

    diag = ev._multi_feature_diagnostic(
        "ignored", _START, grid[-1], grid, symbols, mask, close, quote_vol * 0.0,
        panels=panels,
    )

    # Batch reference using the existing primitives.
    books = build_feature_books(FEATURE_REGISTRY, panels, mask, decision_grid, min_symbols=8)
    ref_admitted: dict[str, dict] = {}
    ref_excluded: dict[str, dict] = {}
    for spec in FEATURE_REGISTRY:
        feature = spec.builder(panels)
        coverage = feature_coverage_audit(feature, mask)
        failing = [year for year, cov in coverage.items() if cov < spec.min_coverage]
        if failing:
            ref_excluded[spec.name] = {"failing_year": min(failing)}
            continue
        if spec.name not in books:
            continue
        base_net, _ = mhs_ledger_pnl(
            books[spec.name], close, quote_vol * 0.0,
            ev.MEASURED_EXECUTION_COST_TIERS_BPS["base"],
        )
        ref_admitted[spec.name] = {"_net": base_net}
    net_panel = {name: fields["_net"] for name, fields in ref_admitted.items()}
    combinable = {}
    for name, net in net_panel.items():
        cleaned = net.dropna()
        sd = float(cleaned.std(ddof=1)) if len(cleaned) > 1 else 0.0
        if np.isfinite(sd) and sd > 0:
            combinable[name] = net
    combined = None
    ref_per_tier: dict[str, float | None] = {}
    if combinable:
        combined = equal_risk_combination(
            {name: books[name] for name in combinable}, combinable,
        )
        for tier, cost_bps in ev.MEASURED_EXECUTION_COST_TIERS_BPS.items():
            per_feature = {
                name: mhs_ledger_pnl(books[name], close, quote_vol * 0.0, cost_bps)[0]
                for name in combinable
            }
            combined_net = (
                sum(per_feature[name] / combinable[name].std(ddof=1) for name in combinable)
                / len(combinable)
            )
            ref_per_tier[tier] = statistics._annualized_1h_sharpe(combined_net)
    else:
        ref_per_tier = dict.fromkeys(ev.MEASURED_EXECUTION_COST_TIERS_BPS)

    ref_gross: float | None = None
    if combined is not None:
        ref_gross = float(
            (
                combined
                * len(combinable)
                / sum(1.0 / combinable[name].std(ddof=1) for name in combinable)
            ).abs().sum(axis=1).mean()
        )
    ref_breadth: dict[str, float] | None = None
    if len(net_panel) >= 2:
        n_eff, mean_corr = ev.effective_breadth(pd.DataFrame(net_panel).fillna(0.0))
        ref_breadth = {"n_eff": n_eff, "mean_corr": mean_corr}

    assert set(diag["admitted"]) == set(ref_admitted)
    assert set(diag["excluded"]) == set(ref_excluded)
    assert diag["combined"]["book_mean_gross"] == ref_gross
    for tier in ev.MEASURED_EXECUTION_COST_TIERS_BPS:
        got = diag["combined"]["net_sharpe_per_tier"][tier]
        want = ref_per_tier[tier]
        assert (got is None and want is None) or got == want
    if ref_breadth is not None:
        assert diag["feature_book_effective_breadth"] == ref_breadth
    else:
        assert diag["feature_book_effective_breadth"] is None
