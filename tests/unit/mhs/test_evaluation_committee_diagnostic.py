"""MHS evaluation contract tests (split by behavioral domain; shared builders live in the original module)."""

"""Contract coverage for the MHS application evaluation resource telemetry."""
import logging
import numpy as np
import pandas as pd
import pytest
from src.application.research.mhs import evaluation as ev
from src.application.research.mhs.evaluation import (
    MhsDiagnosticRequest,
)

from tests.unit.application.research.mhs.test_evaluation import (  # noqa: F401
    _START,
)

def test_committee_request_validation() -> None:
    # SCENARIO_MHS_REQUEST_COMMITTEE_VALIDATION: MhsDiagnosticRequest gains
    # committee_book (bool, default False). A non-bool value raises ValueError
    # (fail closed -- no silent no-op); the default construction leaves it False.
    assert MhsDiagnosticRequest().committee_book is False
    with pytest.raises(ValueError, match="committee_book"):
        MhsDiagnosticRequest(committee_book="yes")
    on = MhsDiagnosticRequest(committee_book=True)
    assert on.committee_book is True

@pytest.mark.slow
def test_committee_default_off_bit_identical(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_COMMITTEE_DEFAULT_OFF_BIT_IDENTICAL: with the flag omitted
    # (committee_book=False) the report's committee_diagnostic is None and every
    # pre-existing field is bit-identical to the explicit-off baseline -- the
    # committee axis is inert unless explicitly enabled.
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
    default_report = ev.run_mhs_horizon_diagnostic(MhsDiagnosticRequest(**base))
    explicit_off = ev.run_mhs_horizon_diagnostic(
        MhsDiagnosticRequest(**base, committee_book=False),
    )
    assert default_report.status == "COMPLETE"
    assert default_report.committee_diagnostic is None
    assert explicit_off.committee_diagnostic is None
    for field in ("books", "blend", "blend_target_gross", "research_go", "folds"):
        assert getattr(default_report, field) == getattr(explicit_off, field)

@pytest.mark.slow
def test_committee_diagnostic_reports_walk_forward_wealth(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_MHS_COMMITTEE_DIAGNOSTIC_REPORTS_WALK_FORWARD_WEALTH: with
    # committee_book=True the report's committee_diagnostic dict carries the
    # declared member names, the admitted/excluded split against the coverage
    # gate (feature- and source-gated, each with a reason), per-required-column
    # source coverage audited before any fillna, and the purged walk-forward
    # wealth metrics (net Sharpe, CAGR, MDD, logret) per measured cost tier --
    # every reported value finite or an explicit None. The fixture spans past
    # COMMITTEE_OOS_START so the block grid has real test bars (B1).
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
    diag = report.committee_diagnostic
    assert isinstance(diag, dict)
    members = diag["members"]
    assert isinstance(members, list)
    assert len(members) == 5
    assert len(set(members)) == 5
    admitted = diag["admitted"]
    excluded = diag["excluded"]
    assert isinstance(admitted, list)
    assert isinstance(excluded, list)
    assert set(admitted) <= set(members)
    assert all(isinstance(e, dict) and e["name"] in members for e in excluded)
    for entry in excluded:
        assert entry["reason"] in ("feature_coverage", "source_coverage")
        if entry["reason"] == "source_coverage":
            assert entry["failing_source"] in (
                "close", "open", "high", "low", "quote_vol",
                "taker_buy_quote", "no_trades",
            )
            assert isinstance(entry["failing_year"], int)
    source_coverage = diag["source_coverage"]
    assert isinstance(source_coverage, dict)
    for per_source in source_coverage.values():
        assert isinstance(per_source, dict)
        for coverage in per_source.values():
            assert isinstance(coverage, dict)
            for value in coverage.values():
                assert 0.0 <= value <= 1.0
    wf = diag["walk_forward"]
    assert isinstance(wf["block_edges"], list)
    assert wf["block_edges"][0] == ev.COMMITTEE_OOS_START.isoformat()
    assert wf["purge_hours"] == 720
    assert wf["target_vol"] == pytest.approx(0.15)
    assert isinstance(wf["skipped_blocks"], list)
    per_tier = wf["per_tier"]
    assert set(per_tier) == set(ev.MEASURED_EXECUTION_COST_TIERS_BPS)
    for fields in per_tier.values():
        assert isinstance(fields["bars"], int)
        assert fields["bars"] >= 0
        for key in ("net_sharpe", "cagr", "mdd", "logret"):
            value = fields[key]
            assert value is None or np.isfinite(value)

@pytest.mark.slow
def test_committee_diagnostic_per_tier_blocks_present(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_MHS_COMMITTEE_PER_TIER_BLOCKS_PRESENT: every tier's walk-forward
    # dict carries a per-block breakdown (same edges logic as skipped_blocks)
    # that partitions the tier's aggregate bar count exactly -- no
    # double-count or calendar gap against the total.
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
    per_tier = report.committee_diagnostic["walk_forward"]["per_tier"]
    assert set(per_tier) == set(ev.MEASURED_EXECUTION_COST_TIERS_BPS)
    for fields in per_tier.values():
        assert isinstance(fields["blocks"], list)
        for block in fields["blocks"]:
            assert isinstance(block["bars"], int)
            assert block["bars"] > 0
            assert isinstance(block["block_start"], str)
            for key in ("net_sharpe", "cagr", "mdd"):
                assert block[key] is None or np.isfinite(block[key])
        assert sum(b["bars"] for b in fields["blocks"]) == fields["bars"]

@pytest.mark.slow
def test_committee_diagnostic_block_logret_share_reported(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_MHS_COMMITTEE_DIAGNOSTIC_BLOCK_LOGRET_SHARE_REPORTED: every block
    # carries 'logret' and 'logret_share' keys, and the non-None shares across a
    # tier sum to ~1.0 -- a structural ratio (report-only, never a gate) that
    # surfaces single-block dominance, mirroring top1_event_share.
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
    per_tier = report.committee_diagnostic["walk_forward"]["per_tier"]
    for tier, fields in per_tier.items():
        if fields["bars"] == 0:
            continue
        shares = []
        for block in fields["blocks"]:
            assert "logret" in block
            assert "logret_share" in block
            if block["logret_share"] is not None:
                assert np.isfinite(block["logret_share"])
                shares.append(block["logret_share"])
        if shares:
            assert sum(shares) == pytest.approx(1.0, abs=1e-9), tier

@pytest.mark.slow
def test_committee_diagnostic_block_return_autocorr_lag1_present(
    mhs_market_long, monkeypatch,
) -> None:
    # SCENARIO_MHS_COMMITTEE_DIAGNOSTIC_BLOCK_RETURN_AUTOCORR_LAG1_PRESENT:
    # every walk-forward block carries 'return_autocorr_lag1', either None
    # (non-finite, e.g. a <=2-bar or zero-variance block) or a finite float in
    # [-1.0, 1.0] -- the block-scoped lag-1 autocorrelation of the raw
    # tranche_count=1 committee net returns.
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
    per_tier = report.committee_diagnostic["walk_forward"]["per_tier"]
    for fields in per_tier.values():
        for block in fields["blocks"]:
            assert "return_autocorr_lag1" in block
            value = block["return_autocorr_lag1"]
            assert value is None or (
                isinstance(value, float)
                and np.isfinite(value)
                and -1.0 <= value <= 1.0
            )

@pytest.mark.slow
def test_committee_diagnostic_block_return_autocorr_lag1_matches_manual_computation(
    mhs_market_long, monkeypatch,
) -> None:
    # SCENARIO_MHS_COMMITTEE_DIAGNOSTIC_BLOCK_RETURN_AUTOCORR_LAG1_MATCHES_MANUAL_COMPUTATION:
    # the reported value is the true block-scoped pandas .autocorr(1) on that
    # block's own net-return slice at the 'base' cost tier -- recomputed
    # independently by capturing the purged walk-forward series during the run
    # and slicing it on the reported block edges.
    root, end = mhs_market_long
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    captured: dict[str, pd.Series] = {}
    real_wf = ev.purged_walk_forward

    def _recording_wf(*args, **kwargs):
        result = real_wf(*args, **kwargs)
        captured[args[2]] = result
        return result

    monkeypatch.setattr(ev, "purged_walk_forward", _recording_wf)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, committee_book=True,
    )
    report = ev.run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    base_wf = captured.get(ev.MEASURED_EXECUTION_COST_TIERS_BPS["base"])
    assert base_wf is not None
    base_fields = report.committee_diagnostic["walk_forward"]["per_tier"]["base"]
    for block in base_fields["blocks"]:
        if block["bars"] <= 2:
            continue
        slice_start = pd.Timestamp(block["block_start"])
        block_slice = base_wf[base_wf.index >= slice_start].iloc[: block["bars"]]
        expected = block_slice.autocorr(1)
        if not np.isfinite(expected):
            assert block["return_autocorr_lag1"] is None
        else:
            assert block["return_autocorr_lag1"] == pytest.approx(
                float(expected), abs=1e-9,
            )

@pytest.mark.slow
def test_committee_diagnostic_block_existing_fields_unchanged(
    mhs_market_long, monkeypatch,
) -> None:
    # SCENARIO_MHS_COMMITTEE_DIAGNOSTIC_EXISTING_BLOCK_FIELDS_UNCHANGED: adding
    # the new key leaves the prior per-block fields ('bars', 'block_start',
    # 'net_sharpe', 'cagr', 'mdd', 'logret', 'logret_share') intact -- same keys,
    # same values, same types -- so pre-existing per-block consumers are
    # unaffected.
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
    prior_keys = {
        "block_start", "bars", "net_sharpe", "cagr", "mdd", "logret", "logret_share",
    }
    for fields in report.committee_diagnostic["walk_forward"]["per_tier"].values():
        for block in fields["blocks"]:
            assert prior_keys.issubset(set(block))
            assert isinstance(block["block_start"], str)
            assert isinstance(block["bars"], int)
            for key in ("net_sharpe", "cagr", "mdd", "logret", "logret_share"):
                assert block[key] is None or isinstance(block[key], float)

@pytest.mark.slow
def test_committee_diagnostic_off_by_default_unchanged(
    mhs_market_long, monkeypatch,
) -> None:
    # SCENARIO_MHS_COMMITTEE_DIAGNOSTIC_OFF_BY_DEFAULT_UNCHANGED: with
    # committee_book and committee_capital both False (defaults) the report's
    # committee_diagnostic stays exactly None -- the new field only ever appears
    # inside an already-opt-in diagnostic block.
    root, end = mhs_market_long
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8,
    )
    report = ev.run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    assert report.committee_diagnostic is None

@pytest.mark.slow
def test_committee_diagnostic_debug_logs_emitted(mhs_market_long, monkeypatch, caplog) -> None:
    # SCENARIO_MHS_COMMITTEE_DEBUG_LOGS_EMITTED: at DEBUG level the
    # MhsHorizonDiagnostic logger emits all four committee checkpoints --
    # source coverage, member PnL, per-block walk-forward, per-tier summary.
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
    with caplog.at_level(logging.DEBUG, logger="MhsHorizonDiagnostic"):
        report = ev.run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    messages = [r.message for r in caplog.records]
    for tag in (
        "stage=committee_source_coverage",
        "stage=committee_member",
        "stage=committee_block",
        "stage=committee_tier_summary",
    ):
        assert any(tag in m for m in messages), tag

@pytest.mark.slow
def test_committee_diagnostic_telemetry_stages_recorded(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_MHS_COMMITTEE_TELEMETRY_STAGES_RECORDED: with committee_book=True
    # the report's resource_measurements carry the diagnostic-feature panel
    # load, the whole committee diagnostic, and one walk-forward checkpoint per
    # measured cost tier -- so a production timeout can be attributed precisely.
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
    stages = {m.stage for m in report.resource_measurements}
    assert "diagnostic_feature_panels" in stages
    assert "committee_diagnostic" in stages
    for tier in ev.MEASURED_EXECUTION_COST_TIERS_BPS:
        assert f"committee_walk_forward_{tier}" in stages

@pytest.mark.slow
def test_committee_diagnostic_uses_oos_start_not_raw_start(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_COMMITTEE_DIAGNOSTIC_USES_OOS_START_NOT_RAW_START (B1): on a
    # panel spanning 2021-2025 the committee diagnostic's walk-forward block
    # grid is anchored at COMMITTEE_OOS_START (2023-01-01), never the
    # diagnostic's own 2021 start; monkeypatching the constant to a different
    # date shifts the first edge, proving the constant is actually read.
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
    first_edge = report.committee_diagnostic["walk_forward"]["block_edges"][0]
    assert first_edge == ev.COMMITTEE_OOS_START.isoformat()
    assert first_edge == "2023-01-01T00:00:00+00:00"

    shifted = pd.Timestamp("2023-07-01", tz="UTC")
    monkeypatch.setattr(ev, "COMMITTEE_OOS_START", shifted)
    report2 = ev.run_mhs_horizon_diagnostic(request)
    assert report2.status == "COMPLETE"
    first_edge2 = report2.committee_diagnostic["walk_forward"]["block_edges"][0]
    assert first_edge2 == shifted.isoformat()
    assert first_edge2 == "2023-07-01T00:00:00+00:00"

def test_search_trials_attempted_raised_and_deflation_more_conservative() -> None:
    # SCENARIO_SEARCH_TRIALS_ATTEMPTED_RAISED_AND_DEFLATED_SHARPE_MORE_CONSERVATIVE
    # (B4): SEARCH_TRIALS_ATTEMPTED is raised to 70 (prior 20 + ~50 committee
    # configurations), and deflated_sharpe_ratio is strictly non-increasing in
    # the trial count, so the raised constant can only make the top-level
    # statistic more conservative, never more optimistic.
    from src.mhs.types import SEARCH_TRIALS_ATTEMPTED
    from src.mhs.evidence import deflated_sharpe_ratio

    assert SEARCH_TRIALS_ATTEMPTED == 70
    kwargs = {"observed_sr": 0.12, "trial_sr_variance": 0.0025, "n_obs": 1200, "skew": 0.0, "kurtosis": 3.0}
    d70 = deflated_sharpe_ratio(n_trials=70, **kwargs)
    d20 = deflated_sharpe_ratio(n_trials=20, **kwargs)
    assert np.isfinite(d70)
    assert np.isfinite(d20)
    assert d70 <= d20

@pytest.mark.slow
def test_committee_source_coverage_gates_admission(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_COMMITTEE_SOURCE_COVERAGE_GATES_ADMISSION (B3): a member whose
    # required RAW source column has coverage below FEATURE_MIN_COVERAGE in
    # any year is fail-closed excluded from admission -- the fixture's missing
    # taker_buy_quote column (mirroring the funding 45/452-symbol gap) gates
    # both flow_imb members BEFORE build_feature_books, and the excluded list
    # carries the failing source/year. With a full-coverage taker_buy_quote the
    # gate is a no-op and all 6 members are admitted (regression).
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
    diag = report.committee_diagnostic
    assert "flow_imb_720h" not in diag["admitted"]
    assert "flow_imb_168h" not in diag["admitted"]
    flow_excluded = {e["name"]: e for e in diag["excluded"] if e["name"] in ("flow_imb_720h", "flow_imb_168h")}
    assert set(flow_excluded) == {"flow_imb_720h", "flow_imb_168h"}
    for entry in flow_excluded.values():
        assert entry["reason"] == "source_coverage"
        assert entry["failing_source"] == "taker_buy_quote"
        assert isinstance(entry["failing_year"], int)

    # Regression: with a full-coverage taker_buy_quote the source gate admits
    # every member -- B3 is fail-closed-only and non-disruptive to the shipped
    # committee.
    real_load = ev._load_feature_panels
    def _full_coverage_panels(root_arg, start_arg, end_arg, grid_1h, aligned_symbols, columns=None):
        panels = real_load(root_arg, start_arg, end_arg, grid_1h, aligned_symbols, columns=columns)
        quote_vol = panels["quote_vol"]
        panels["taker_buy_quote"] = quote_vol * 0.5
        return panels
    from src.application.research.mhs import stage_services
    import src.mhs.pipeline.stages.fold as fold_stage
    monkeypatch.setattr(ev, "_load_feature_panels", _full_coverage_panels)
    monkeypatch.setattr(stage_services, "_load_feature_panels", _full_coverage_panels)
    monkeypatch.setattr(fold_stage, "_load_feature_panels", _full_coverage_panels)
    report_full = ev.run_mhs_horizon_diagnostic(request)
    assert report_full.status == "COMPLETE"
    assert set(report_full.committee_diagnostic["admitted"]) == set(
        report_full.committee_diagnostic["members"]
    )

@pytest.mark.slow
def test_committee_diagnostic_reports_trials_and_warning(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_COMMITTEE_DIAGNOSTIC_REPORTS_TRIALS_AND_WARNING (B4/B5): the
    # committee diagnostic reports trials_explored == 50 and a non-empty
    # selection_bias_warning naming the configuration count, and tags its
    # evaluation protocol as purged walk-forward OOS (B5).
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
    diag = report.committee_diagnostic
    assert diag["trials_explored"] == 50
    assert isinstance(diag["selection_bias_warning"], str)
    assert diag["selection_bias_warning"]
    assert "~50" in diag["selection_bias_warning"]
    assert diag["evaluation_protocol"] == "purged_walk_forward_oos"

@pytest.mark.slow
def test_evaluation_protocol_field_distinguishes_in_sample_from_oos(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_EVALUATION_PROTOCOL_FIELD_DISTINGUISHES_IN_SAMPLE_FROM_OOS (B5):
    # the two opt-in diagnostics carry distinct protocol tags on every call, so
    # a reader can never mistake the in-sample full-period net Sharpe for the
    # purged walk-forward OOS numbers.
    from src.mhs.features import FEATURE_REGISTRY

    root, end = mhs_market_long
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, committee_book=True, multi_feature_book=True,
    )
    report = ev.run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    assert report.committee_diagnostic["evaluation_protocol"] == "purged_walk_forward_oos"
    assert report.multi_feature_diagnostic["evaluation_protocol"] == "in_sample_full_period"
    assert report.multi_feature_diagnostic["trials_explored"] == len(FEATURE_REGISTRY)
    assert "selection_bias_warning" not in report.multi_feature_diagnostic

@pytest.mark.slow
def test_committee_diagnostic_reports_skipped_blocks(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_COMMITTEE_DIAGNOSTIC_REPORTS_SKIPPED_BLOCKS (B6): the committee
    # diagnostic reports skipped_blocks as a list of {block_start, reason}
    # entries computed independently of purged_walk_forward's internal skip
    # loop. On a 2021-2025 panel anchored at OOS_START 2023-01-01 every 6-month
    # block has both sufficient train and at least one test bar, so the list is
    # empty (report-only, never raises).
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
    skipped = report.committee_diagnostic["walk_forward"]["skipped_blocks"]
    assert isinstance(skipped, list)
    for entry in skipped:
        assert set(entry) == {"block_start", "reason"}
        assert entry["reason"] in ("insufficient_train", "no_test_bars")
    assert skipped == []

@pytest.mark.slow
def test_committee_books_regression_unchanged_by_b1_b2(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_COMMITTEE_BOOKS_REGRESSION_UNCHANGED_BY_B1_B2: enabling
    # committee_book must not perturb any pre-existing non-committee report
    # field (books, blend, folds, research_go, trend_sleeve_diagnostic,
    # multi_feature_diagnostic) -- only committee_diagnostic's own walk-forward
    # numbers change by design (B1/B2).
    root, end = mhs_market_long
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    base = {
        "start": str(_START), "end": str(end), "data_root": str(root),
        "mark_mode": "cache_required", "execution_timeframe": "1m", "log_run": False,
        "execution_universe_size": 8,
    }
    off_report = ev.run_mhs_horizon_diagnostic(MhsDiagnosticRequest(**base))
    on_report = ev.run_mhs_horizon_diagnostic(
        MhsDiagnosticRequest(**base, committee_book=True),
    )
    assert off_report.status == "COMPLETE"
    assert on_report.status == "COMPLETE"
    for field in (
        "books", "blend", "blend_target_gross", "research_go", "folds",
        "trend_sleeve_diagnostic", "multi_feature_diagnostic",
    ):
        assert getattr(on_report, field) == getattr(off_report, field)
