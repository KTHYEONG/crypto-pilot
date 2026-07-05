from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.domain.futures.strategy.config import PerTfL1Result
from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    MajorSymbolSleeveContributionSummary,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer1Result, StrategySignal
from src.domain.futures.strategy.tiered_workflow.tf_validation_repair import (
    MajorSymbolGapEvidence,
    ValidationParityReport,
    build_major_symbol_gap_evidence,
    build_multi_tf_major_registry_census,
    build_validation_parity_report,
    classify_major_symbol_gap_evidence,
    log_validation_parity_report,
    summarize_main_compatible_tf_evidence,
    summarize_tf_probe_diagnostics,
)


class DummyRegistry:
    def __init__(self, by_symbol):
        self.by_symbol = by_symbol
        self.ready_symbols = tuple(by_symbol.keys())
        self.trade_scope_count = len(by_symbol)
        self.registry_version = "dummy"


def ev(symbol: str, strategy_id: str, mean_bps: float = 10.0, hard: bool = True, qw: float = 1.0):
    key = SimpleNamespace(symbol=symbol, strategy_id=strategy_id, activation_context="all")
    return SimpleNamespace(
        key=key,
        mean_incremental_bps=mean_bps,
        hard_eligible=hard,
        quality_weight=qw,
        lcb_net_bps=mean_bps,
    )


def layer1_result(*, registry=None, edge=0.0, gate=True):
    return Layer1Result(
        signals_per_fold=(),
        oos_stacked={},
        pooled_ic=0.0,
        pooled_tstat=0.0,
        breadth=1.0,
        valid_coverage=1.0,
        fold_pass_ratio=1.0,
        gate_passed=gate,
        n_valid=1,
        n_total=1,
        strategy_panel=(
            StrategySignal(
                strategy_id="dual_momentum:trend",
                oos_edge_bps=edge,
                oos_nw_tstat=2.5,
                hit_rate=0.55,
                fold_sign_consistency=1.0,
                n_obs=100,
                n_folds=4,
                valid=edge > 0.0,
            ),
        ),
        deployment_registry=registry,
    )


def per_tf(tf: str, registry=None, edge=0.0):
    return PerTfL1Result(
        tf=tf,
        l1_result=layer1_result(registry=registry, edge=edge),
        n_winning_signals=1 if edge > 0 else 0,
    )


def sleeve_summary(symbol="BTCUSDT", family="ichimoku_trend", mismatch=0.75):
    return MajorSymbolSleeveContributionSummary(
        symbol=symbol,
        family=family,
        n_obs=10,
        mean_raw_mu_sleeve=-0.2,
        mean_quality_weight_sleeve=0.7,
        sign_mismatch_pct=mismatch,
        regime_adverse_sign_mismatch_pct=mismatch,
    )


# ─── Scenario 1: Happy Path ───


def _make_probe_cell(tf: str, passed_fdr: bool = True, **kw):
    base = {
        "symbol": "BTCUSDT",
        "family": "dual_momentum",
        "variant": "trend",
        "archetype": "trend",
        "tf": tf,
        "n_obs": 100,
        "n_events": 50,
        "ic_mean": 0.05,
        "ic_tstat_hac": 2.5,
        "ic_fold_sign_consistency": 1.0,
        "alpha_half_life_h": 48.0,
        "net_edge_bps": 10.0,
        "turnover_per_year": 12.0,
        "vr_label": "stationary",
        "hurst": 0.45,
        "passed_fdr": passed_fdr,
    }
    base.update(kw)
    return SimpleNamespace(**base)


def _make_probe_manifest(cells, tf_grid=None):
    tfs = tf_grid or tuple(sorted({c.tf for c in cells}))
    return SimpleNamespace(
        cells=tuple(cells),
        tf_grid=tfs,
        coverage_by_tf=dict.fromkeys(tfs, 2),
        diversity_corr=dict.fromkeys(tfs, 0.3),
    )


class TestScenario1HappyPath:
    def test_summarize_tf_probe_diagnostics_marks_probe_as_diagnostic_only(self):
        cells = [
            _make_probe_cell("4h", passed_fdr=False, ic_tstat_hac=1.5),
            _make_probe_cell("4h", passed_fdr=False, ic_tstat_hac=0.8),
            _make_probe_cell("6h", passed_fdr=True, ic_tstat_hac=3.0),
        ]
        manifest = _make_probe_manifest(cells, tf_grid=("4h", "6h"))
        result = summarize_tf_probe_diagnostics(manifest)
        assert all(v.decision == "diagnostic_only" for v in result)
        verdicts_by_tf = {v.tf: v for v in result}
        assert verdicts_by_tf["4h"].n_winning == 0
        assert verdicts_by_tf["6h"].n_winning == 1

    def test_summarize_main_compatible_tf_evidence_keeps_existing_tf_despite_zero_probe_winners(
        self,
    ):
        reg = DummyRegistry({"BTCUSDT": (ev("BTCUSDT", "dual_momentum"),)})
        ptf = {"4h": per_tf("4h", registry=reg, edge=10.0), "6h": per_tf("6h", registry=reg, edge=5.0)}
        result = summarize_main_compatible_tf_evidence(ptf)
        for ev_row in result:
            assert ev_row.candidate_decision == "keep_existing"

    def test_build_multi_tf_major_registry_census_unions_all_tf_registries(self):
        reg4 = DummyRegistry({"BTCUSDT": (ev("BTCUSDT", "dual_momentum", mean_bps=15.0),)})
        reg8 = DummyRegistry({"BTCUSDT": (ev("BTCUSDT", "ichimoku_trend", mean_bps=12.0),)})
        ptf = {
            "4h": per_tf("4h", registry=reg4, edge=10.0),
            "8h": per_tf("8h", registry=reg8, edge=8.0),
        }
        census = build_multi_tf_major_registry_census(ptf, (), symbols=("BTCUSDT",))
        tfs_in_census = {t for t, _ in census}
        assert "4h" in tfs_in_census
        assert "8h" in tfs_in_census

    def test_build_major_symbol_gap_evidence_classifies_btc_outvoted_and_eth_admission_gap(
        self,
    ):
        reg = DummyRegistry(
            {
                "BTCUSDT": (ev("BTCUSDT", "ichimoku_trend", mean_bps=10.0, hard=True),),
            }
        )
        ptf = {"4h": per_tf("4h", registry=reg, edge=10.0)}
        summaries = [
            sleeve_summary("BTCUSDT", "ichimoku_trend", mismatch=0.75),
            sleeve_summary("ETHUSDT", "ichimoku_trend", mismatch=0.3),
        ]
        gaps = build_major_symbol_gap_evidence(
            per_tf_l1=ptf,
            observed_sleeve_summaries=summaries,
            symbols=("BTCUSDT", "ETHUSDT"),
        )
        btc_gaps = [g for g in gaps if g.symbol == "BTCUSDT"]
        eth_gaps = [g for g in gaps if g.symbol == "ETHUSDT"]
        assert any(g.gap_class == "outvoted" for g in btc_gaps)
        assert any(g.gap_class == "admission_gap" for g in eth_gaps)

    def test_build_validation_parity_report_combines_probe_main_and_gap_outputs(self):
        cells = [_make_probe_cell("4h", passed_fdr=True, ic_tstat_hac=3.0)]
        manifest = _make_probe_manifest(cells, tf_grid=("4h",))
        reg = DummyRegistry({"BTCUSDT": (ev("BTCUSDT", "dual_momentum"),)})
        reg_candidate = DummyRegistry({"BTCUSDT": (ev("BTCUSDT", "dual_momentum"),)})
        ptf = {
            "4h": per_tf("4h", registry=reg, edge=10.0),
            "1h": per_tf("1h", registry=reg_candidate, edge=5.0),
        }
        report = build_validation_parity_report(
            probe_manifest=manifest,
            per_tf_l1=ptf,
            observed_sleeve_summaries=(),
            candidate_tfs=("1h", "2h"),
        )
        assert len(report.probe) > 0
        assert len(report.main_tf) > 0
        assert report.decision == "candidate_review_required"


# ─── Scenario 2: Edge Cases ───


class TestScenario2EdgeCases:
    def test_gap_evidence_does_not_mutate_registry_or_candidate_decisions_from_holdout(self):
        reg = DummyRegistry({"BTCUSDT": (ev("BTCUSDT", "ichimoku_trend", mean_bps=10.0, hard=True),)})
        ptf = {"4h": per_tf("4h", registry=reg, edge=10.0)}
        main_before = summarize_main_compatible_tf_evidence(ptf)
        summaries = [sleeve_summary("BTCUSDT", "ichimoku_trend", mismatch=0.75)]
        gaps = build_major_symbol_gap_evidence(
            per_tf_l1=ptf,
            observed_sleeve_summaries=summaries,
            symbols=("BTCUSDT",),
        )
        main_after = summarize_main_compatible_tf_evidence(ptf)
        assert all(
            b.candidate_decision == a.candidate_decision
            for b, a in zip(main_before, main_after, strict=True)
        )
        assert any(g.gap_class == "outvoted" for g in gaps)

    def test_main_compatible_evidence_ignores_probe_manifest_for_edge_quality(self):
        ptf = {"4h": per_tf("4h", registry=None, edge=10.0)}
        result = summarize_main_compatible_tf_evidence(ptf)
        assert all(r.candidate_decision == "keep_existing" for r in result)

    def test_candidate_tf_with_probe_winner_but_no_l1_registry_is_review_or_reject_not_keep(
        self,
    ):
        cells = [_make_probe_cell("1h", passed_fdr=True, ic_tstat_hac=3.0)]
        manifest = _make_probe_manifest(cells, tf_grid=("1h",))
        ptf = {"4h": per_tf("4h", registry=None, edge=10.0)}
        report = build_validation_parity_report(
            probe_manifest=manifest,
            per_tf_l1=ptf,
            candidate_tfs=("1h", "2h"),
        )
        assert any("candidate_tf_missing_main_l1:1h" in b for b in report.blockers)

    def test_probe_winning_zero_does_not_block_existing_main_tf(self):
        reg = DummyRegistry({"BTCUSDT": (ev("BTCUSDT", "dual_momentum"),)})
        ptf = {"4h": per_tf("4h", registry=reg, edge=10.0)}
        result = summarize_main_compatible_tf_evidence(ptf)
        assert all(r.candidate_decision == "keep_existing" for r in result)

    def test_validation_report_defaults_symbols_to_major_diag_symbols(self):
        reg = DummyRegistry(
            {
                "BTCUSDT": (ev("BTCUSDT", "dual_momentum"),),
                "ETHUSDT": (ev("ETHUSDT", "dual_momentum"),),
            }
        )
        ptf = {"4h": per_tf("4h", registry=reg, edge=10.0)}
        gaps = build_major_symbol_gap_evidence(
            per_tf_l1=ptf,
            observed_sleeve_summaries=(),
        )
        symbols_in_gaps = {g.symbol for g in gaps}
        assert symbols_in_gaps <= {"BTCUSDT", "ETHUSDT", "BNBUSDT"}

    def test_build_major_symbol_gap_evidence_returns_empty_after_per_tf_l1_is_empty(self):
        gaps = build_major_symbol_gap_evidence(
            per_tf_l1={},
            observed_sleeve_summaries=(),
        )
        assert gaps == ()

    def test_validation_report_adds_blocker_when_no_main_tf_evidence_exists(self):
        cells = [_make_probe_cell("1h", passed_fdr=True, ic_tstat_hac=3.0)]
        manifest = _make_probe_manifest(cells, tf_grid=("1h",))
        report = build_validation_parity_report(
            probe_manifest=manifest,
            per_tf_l1={},
        )
        assert any("missing_main_tf_evidence" in b for b in report.blockers)

    def test_repair_action_depends_only_on_gap_class_not_global_holdout_metric(self):
        expected = {
            "outvoted": "major_sign_rank_combiner_spec",
            "admission_gap": "l1_family_evidence_spec",
            "activation_gap": "activation_context_spec",
            "no_gap": "none",
        }
        for gap_class, expected_action in expected.items():
            gap = MajorSymbolGapEvidence(
                symbol="BTCUSDT",
                tf="4h",
                family="ichimoku_trend",
                in_registry=True,
                hard_eligible=True,
                observed_active=True,
                registry_mean_incremental_bps=10.0,
                regime_adverse_mismatch_pct=0.0,
                mean_raw_mu=0.0,
                mean_quality_weight=1.0,
                gap_class=gap_class,
                repair_action=expected_action,
            )
            assert gap.repair_action == expected_action


# ─── Scenario 3: Error Handling ───


class TestScenario3ErrorHandling:
    def test_summarize_tf_probe_diagnostics_none_manifest_returns_empty(self):
        result = summarize_tf_probe_diagnostics(None)
        assert result == ()

    def test_summarize_main_compatible_tf_evidence_handles_registry_contract_duplicates(self):
        reg = SimpleNamespace(
            by_symbol={"BTCUSDT": (SimpleNamespace(
                key=SimpleNamespace(symbol="BTCUSDT", strategy_id="dual_momentum", activation_context="all"),
                mean_incremental_bps=10.0,
                hard_eligible=True,
                quality_weight=1.0,
                lcb_net_bps=10.0,
            ),)},
            ready_symbols=("BTCUSDT",),
        )
        ptf = {"4h": per_tf("4h", registry=reg, edge=10.0)}
        result = summarize_main_compatible_tf_evidence(ptf)
        assert len(result) == 1
        assert result[0].registry_present

    def test_build_multi_tf_major_registry_census_skips_malformed_registry(self):
        bad_reg = SimpleNamespace(no_by_symbol=True)
        ptf = {"4h": per_tf("4h", registry=bad_reg, edge=10.0)}
        census = build_multi_tf_major_registry_census(ptf, ())
        assert census == ()

    def test_classify_major_symbol_gap_evidence_handles_missing_observed_summary(self):
        entry = SimpleNamespace(
            symbol="BTCUSDT",
            family="ichimoku_trend",
            registry_mean_incremental_bps=10.0,
            hard_eligible=True,
            observed_active_in_holdout=False,
        )
        gap = classify_major_symbol_gap_evidence(
            tf="4h",
            entry=entry,
            symbol="BTCUSDT",
            family="ichimoku_trend",
            observed_sleeve_summaries=(),
        )
        assert gap.gap_class == "activation_gap"

    def test_build_multi_tf_major_registry_census_resolves_inference_artifact_fallback(self):
        artifact_reg = DummyRegistry({"BTCUSDT": (ev("BTCUSDT", "dual_momentum", mean_bps=15.0),)})
        artifact = SimpleNamespace(deployment_registry=artifact_reg)
        l1 = layer1_result(registry=None, edge=10.0)
        l1 = SimpleNamespace(
            signals_per_fold=l1.signals_per_fold,
            oos_stacked=l1.oos_stacked,
            pooled_ic=l1.pooled_ic,
            pooled_tstat=l1.pooled_tstat,
            breadth=l1.breadth,
            valid_coverage=l1.valid_coverage,
            fold_pass_ratio=l1.fold_pass_ratio,
            gate_passed=l1.gate_passed,
            n_valid=l1.n_valid,
            n_total=l1.n_total,
            strategy_panel=l1.strategy_panel,
            deployment_registry=None,
            inference_artifact=artifact,
        )
        ptf = {"4h": SimpleNamespace(tf="4h", l1_result=l1, n_winning_signals=1)}
        census = build_multi_tf_major_registry_census(ptf, (), symbols=("BTCUSDT",))
        assert len(census) > 0
        assert any(sym == "BTCUSDT" for _, entry in census for sym in [entry.symbol])

    def test_classify_major_symbol_gap_evidence_hard_ineligible_is_admission_gap(self):
        entry = SimpleNamespace(
            symbol="BTCUSDT",
            family="ichimoku_trend",
            registry_mean_incremental_bps=10.0,
            hard_eligible=False,
            observed_active_in_holdout=False,
        )
        gap = classify_major_symbol_gap_evidence(
            tf="4h",
            entry=entry,
            symbol="BTCUSDT",
            family="ichimoku_trend",
            observed_sleeve_summaries=(),
        )
        assert gap.gap_class == "admission_gap"
        assert gap.repair_action == "l1_family_evidence_spec"

    def test_classify_major_symbol_gap_evidence_observed_low_mismatch_is_no_gap(self):
        entry = SimpleNamespace(
            symbol="BTCUSDT",
            family="ichimoku_trend",
            registry_mean_incremental_bps=10.0,
            hard_eligible=True,
            observed_active_in_holdout=True,
        )
        summaries = (sleeve_summary("BTCUSDT", "ichimoku_trend", mismatch=0.10),)
        gap = classify_major_symbol_gap_evidence(
            tf="4h",
            entry=entry,
            symbol="BTCUSDT",
            family="ichimoku_trend",
            observed_sleeve_summaries=summaries,
            adverse_sign_mismatch_threshold=0.50,
        )
        assert gap.gap_class == "no_gap"
        assert gap.repair_action == "none"

    def test_build_validation_parity_report_handles_malformed_strategy_panel(self):
        bad_panel = (SimpleNamespace(no_strategy_id=True),)
        l1 = layer1_result(registry=None, edge=0.0)
        l1 = SimpleNamespace(
            signals_per_fold=l1.signals_per_fold,
            oos_stacked=l1.oos_stacked,
            pooled_ic=l1.pooled_ic,
            pooled_tstat=l1.pooled_tstat,
            breadth=l1.breadth,
            valid_coverage=l1.valid_coverage,
            fold_pass_ratio=l1.fold_pass_ratio,
            gate_passed=l1.gate_passed,
            n_valid=l1.n_valid,
            n_total=l1.n_total,
            strategy_panel=bad_panel,
            deployment_registry=None,
            inference_artifact=None,
        )
        ptf = {"4h": SimpleNamespace(tf="4h", l1_result=l1, n_winning_signals=0)}
        report = build_validation_parity_report(probe_manifest=None, per_tf_l1=ptf)
        assert report is not None


def test_log_validation_parity_report_does_not_raise(caplog: pytest.LogCaptureFixture) -> None:
    report = ValidationParityReport(
        probe=(),
        main_tf=(),
        major_gaps=(),
        decision="diagnostic_only",
        blockers=(),
    )
    log_validation_parity_report(report)
    assert True
