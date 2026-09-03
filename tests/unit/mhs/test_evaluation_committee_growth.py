"""MHS evaluation contract tests (split by behavioral domain; shared builders live in the original module)."""

"""Contract coverage for the MHS application evaluation resource telemetry."""
import dataclasses
import pytest
from src.mhs import evaluation as ev
from src.mhs.diagnostic_run import run_mhs_horizon_diagnostic
from src.mhs.evaluation import (
    MhsDiagnosticRequest,
)

from tests.unit.mhs.test_evaluation_appresearch import (  # noqa: F401
    _START,
    _committee_growth_panels,
)

def test_committee_growth_headroom_discovery_only_causality() -> None:
    # SCENARIO_COMMITTEE_GROWTH_HEADROOM_DISCOVERY_ONLY_CAUSALITY: OOS bars
    # (>= COMMITTEE_OOS_START) never enter the discovery-only fit -- mutating
    # them to extreme values leaves the diagnostic byte-identical.
    gross, tc = _committee_growth_panels()
    base = ev._committee_growth_headroom(gross, tc, cost_bps=4.18)
    gross_mut = gross.copy()
    tc_mut = tc.copy()
    oos = gross.index >= ev.COMMITTEE_OOS_START
    gross_mut.loc[oos] *= 1e6
    tc_mut.loc[oos] *= 1e6
    assert ev._committee_growth_headroom(gross_mut, tc_mut, cost_bps=4.18) == base

def test_committee_growth_headroom_reference_risk_not_hardcoded() -> None:
    # SCENARIO_COMMITTEE_GROWTH_HEADROOM_REFERENCE_RISK_NOT_HARDCODED: fixtures
    # with different discovery-window volatility yield different reference_risk,
    # each equal to the discovery-window combined net series' std(ddof=1).
    low = ev._committee_growth_headroom(*_committee_growth_panels(discovery_vol_scale=1.0), cost_bps=4.18)
    high = ev._committee_growth_headroom(*_committee_growth_panels(discovery_vol_scale=3.0), cost_bps=4.18)
    assert low is not None
    assert high is not None
    assert low["reference_risk"] != high["reference_risk"]
    for scale, result in ((1.0, low), (3.0, high)):
        gross, tc = _committee_growth_panels(discovery_vol_scale=scale)
        discovery = gross.index < ev.COMMITTEE_OOS_START
        net = gross - tc * 4.18
        weights = ev.long_only_equal_risk_weights(net.loc[discovery])
        discovery_net = ev.score_weighted_net(
            weights, gross.loc[discovery], tc.loc[discovery], 4.18,
        )
        assert result["reference_risk"] == pytest.approx(float(discovery_net.std(ddof=1)))
        assert result["discovery_bars"] == int(discovery.sum())

def test_committee_growth_headroom_short_discovery_returns_none() -> None:
    # SCENARIO_COMMITTEE_GROWTH_HEADROOM_SHORT_DISCOVERY_RETURNS_NONE: fewer than
    # 30 discovery-window bars returns None, never a raised exception.
    gross, tc = _committee_growth_panels(n_days=20)
    assert ev._committee_growth_headroom(gross, tc, cost_bps=4.18) is None

def test_committee_growth_diagnostic_requires_committee_book() -> None:
    # SCENARIO_MHS_DIAGNOSTIC_COMMITTEE_GROWTH_DIAGNOSTIC_REQUIRES_COMMITTEE_BOOK:
    # committee_growth_diagnostic=True without committee_book=True fails closed
    # in __post_init__.
    assert MhsDiagnosticRequest().committee_growth_diagnostic is False
    with pytest.raises(ValueError, match="committee_growth_diagnostic requires committee_book"):
        MhsDiagnosticRequest(committee_growth_diagnostic=True, committee_book=False)
    with pytest.raises(ValueError, match="committee_growth_diagnostic must be a bool"):
        MhsDiagnosticRequest(committee_growth_diagnostic="yes")
    assert (
        MhsDiagnosticRequest(committee_book=True, committee_growth_diagnostic=True)
        .committee_growth_diagnostic
        is True
    )

@pytest.mark.slow
def test_committee_growth_diagnostic_default_off_byte_identical(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_MHS_DIAGNOSTIC_COMMITTEE_GROWTH_DIAGNOSTIC_DEFAULT_OFF_BYTE_IDENTICAL:
    # with committee_growth_diagnostic omitted (default False) the report's
    # growth_headroom is None and the vol-target walk-forward path is untouched.
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
    report = run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    assert report.committee_diagnostic["growth_headroom"] is None
    assert report.committee_diagnostic["walk_forward"]["sizing_mode"] == "vol_target"

@pytest.mark.slow
def test_committee_growth_diagnostic_observational_only(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_COMMITTEE_GROWTH_HEADROOM_OBSERVATIONAL_ONLY: enabling the growth
    # headroom diagnostic must not perturb the reported per-tier walk-forward --
    # the report field is observation-only, never a sizing feedback.
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
    off = run_mhs_horizon_diagnostic(base)
    on = run_mhs_horizon_diagnostic(
        dataclasses.replace(base, committee_growth_diagnostic=True),
    )
    assert off.status == "COMPLETE"
    assert on.status == "COMPLETE"
    assert on.committee_diagnostic["growth_headroom"] is not None
    assert (
        on.committee_diagnostic["walk_forward"]["per_tier"]
        == off.committee_diagnostic["walk_forward"]["per_tier"]
    )
