from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.application.research.expert_portfolio import admission_pipeline as pipeline_app
from src.application.research.expert_portfolio import window as window_module
from src.application.research.expert_portfolio.admission_pipeline import (
    run_technical_library_admission_pipeline,
)
from src.application.research.expert_portfolio.window import (
    ResolvedTechnicalWindow,
    resolve_common_technical_window,
)
from src.common.errors import DataIntegrityError
from src.research.evaluation.metrics import Metrics
from src.research.evaluation.promotion import PromotionResult
from src.research.evaluation.reliability import (
    FoldDistributionResult,
    ReliabilityGateResult,
)
from src.research.expert_portfolio.admission_reports import (
    LibraryAdmissionBacktestReport,
    LibraryAdmissionReport,
)
from src.research.expert_portfolio.admission_types import (
    AdmissionProposal,
    CandidateAdmissionResult,
    LibraryAdmissionConfig,
    TechnicalLibraryAdmissionBacktestRequest,
    TechnicalLibraryAdmissionPipelineRequest,
    admission_proposal_id,
    technical_5symbol_2022_v1_profile,
)
from src.research.expert_portfolio.catalog import default_catalog
from src.research.expert_portfolio.models import ContextualRouterSpec, ExpertDefinition


def _frame_from_index(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"close": np.ones(len(index))}, index=index)


def _canned_gate() -> ReliabilityGateResult:
    return ReliabilityGateResult(
        lcb90_cagr=0.01, lcb95_cagr=0.01, p_negative=0.1, point_cagr=0.02,
        t_stat=1.0, trade_count=5, block_size_used=1, verdict="PASS",
    )


def _canned_folds() -> FoldDistributionResult:
    return FoldDistributionResult(
        n_folds=4, median_fold_cagr=0.01, worst_fold_cagr=0.0,
        median_fold_calmar=0.5, max_period_contribution=0.25, gate_pass=True,
    )


def _canned_metrics() -> Metrics:
    return Metrics(
        cagr=0.02, mdd=-0.01, sharpe=0.5, sortino=0.6, calmar=2.0,
        profit_factor=1.5, expectancy=0.001, win_rate=0.5, payoff_ratio=1.1,
        trade_count=5, exposure=0.5, turnover=0.1, trades_per_year={"2025": 5},
    )


def _canned_backtest_report(proposal_id: str) -> LibraryAdmissionBacktestReport:
    return LibraryAdmissionBacktestReport(
        status="COMPLETE",
        proposal_id=proposal_id,
        expert_ids=(),
        router=ContextualRouterSpec("BTCUSDT", 48, 48, 96),
        window_start="2025-01-01 00:00:00+00:00",
        window_end="2025-12-31 20:00:00+00:00",
        observation_metrics=_canned_metrics(),
        observation_gate=_canned_gate(),
        observation_folds=_canned_folds(),
        stress_metrics=_canned_metrics(),
        stress_gate=_canned_gate(),
        stress_folds=_canned_folds(),
        promotion=PromotionResult(
            status="OBSERVATION_PASS", observation_verdict="PASS",
            fold_gate_pass=True, stress_verdict="PASS", holdout_verdict=None,
        ),
        allocation_cost_total=0.0,
        stress_allocation_cost_total=0.0,
        execution_workers=1,
    )


def _canned_window() -> ResolvedTechnicalWindow:
    return ResolvedTechnicalWindow(
        requested_start="2022-04-01 00:00:00+00:00",
        common_start=pd.Timestamp("2022-04-01 00:00:00", tz="UTC"),
        common_end=pd.Timestamp("2024-12-31 20:00:00", tz="UTC"),
        effective_start=pd.Timestamp("2022-05-04 12:00:00", tz="UTC"),
        end="2024-12-31 20:00:00+00:00",
        symbol_sources={"BTCUSDT": {"ohlcv": "2020-01-01", "funding": "2020-01-01"}},
    )


def _canned_selection_report() -> LibraryAdmissionReport:
    experts = tuple(
        ExpertDefinition(
            f"e{i}", f"source_{i}", f"f{i}", (f"S{i}",), "run_technical_expert", "h" * 64,
        )
        for i in range(5)
    )
    candidates = tuple(
        CandidateAdmissionResult(e.expert_id, 25, 250, True, None) for e in experts
    )
    proposals = (
        AdmissionProposal(("e0", "e1"), True, 0.10, 0.05, 0.10, 0.05),
        AdmissionProposal(("e0", "e2"), True, 0.20, 0.06, 0.20, 0.06),
        AdmissionProposal(("e0", "e1", "e2"), True, 0.20, 0.06, 0.15, 0.055),
    )
    return LibraryAdmissionReport(
        status="COMPLETE",
        window_start="2022-04-01 00:00:00+00:00",
        window_end="2024-12-31 20:00:00+00:00",
        experts=experts,
        candidates=candidates,
        proposals=proposals,
        context_coverage=dict.fromkeys(
            ("up_low_vol", "up_high_vol", "down_low_vol", "down_high_vol", "flat_low_vol", "flat_high_vol"),
            500,
        ),
        covered_states=6,
        coverage_sufficient=True,
        router=ContextualRouterSpec("BTCUSDT", 48, 48, 96),
        admission=LibraryAdmissionConfig(2, 5, 20, 200, 0.50, 0.15, 6, 1_000_000),
        structural_combinations=1000,
    )


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch) -> list[TechnicalLibraryAdmissionBacktestRequest]:
    captured: list[TechnicalLibraryAdmissionBacktestRequest] = []

    def _spy_backtest(req: TechnicalLibraryAdmissionBacktestRequest) -> LibraryAdmissionBacktestReport:
        captured.append(req)
        return _canned_backtest_report(admission_proposal_id(req.expert_ids))

    monkeypatch.setattr(pipeline_app, "resolve_common_technical_window", lambda *_: _canned_window())
    monkeypatch.setattr(pipeline_app, "run_technical_library_admission", lambda req: _canned_selection_report())
    monkeypatch.setattr(pipeline_app, "run_technical_library_admission_backtest", _spy_backtest)
    return captured


class TestCommonWindowResolver:
    def test_resolver_rejects_empty_and_duplicate_symbols(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            resolve_common_technical_window((), None, None)
        with pytest.raises(ValueError, match="duplicates"):
            resolve_common_technical_window(("BTCUSDT", "BTCUSDT"), None, None)

    def test_resolver_rejects_earlier_requested_start_and_names_blocker(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # LAP-02: the resolver rejects an earlier requested start and identifies
        # the OHLCV or funding blocker; a too-early window is never intersected.
        def fake_ohlcv(path, *, start=None, end=None):
            return _frame_from_index(
                pd.date_range("2020-01-01", periods=1000, freq="4h", tz="UTC"),
            )

        def fake_funding(path):
            if "SOLUSDT" in str(path):
                index = pd.date_range("2022-04-01", periods=10, freq="4h", tz="UTC")
            else:
                index = pd.date_range("2020-01-01", periods=10, freq="4h", tz="UTC")
            return pd.Series(0.0, index=index)

        monkeypatch.setattr(window_module, "load_ohlcv_4h", fake_ohlcv)
        monkeypatch.setattr(window_module, "load_funding_rates", fake_funding)

        with pytest.raises(DataIntegrityError, match="SOLUSDT") as excinfo:
            resolve_common_technical_window(("BTCUSDT", "SOLUSDT"), "2021-01-01", None)
        assert "funding" in str(excinfo.value)

    def test_resolver_returns_tz_aware_4h_aligned_effective_start(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_ohlcv(path, *, start=None, end=None):
            return _frame_from_index(
                pd.date_range("2020-01-01", periods=1000, freq="4h", tz="UTC"),
            )

        monkeypatch.setattr(window_module, "load_ohlcv_4h", fake_ohlcv)
        monkeypatch.setattr(
            window_module, "load_funding_rates",
            lambda path: pd.Series(0.0, index=pd.date_range("2020-01-01", periods=10, freq="4h", tz="UTC")),
        )
        resolved = resolve_common_technical_window(
            ("BTCUSDT",), None, "2025-12-31 20:00:00+00:00",
        )
        assert resolved.effective_start.tzinfo is not None
        assert resolved.common_start <= resolved.effective_start
        assert resolved.effective_start.minute == 0
        assert resolved.effective_start.hour % 4 == 0


class TestLibraryAdmissionPipeline:
    def test_frozen_profile_selects_and_backtests_without_mutation(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # LAP-01: the frozen profile performs selection plus OOS backtests in one
        # call, is deterministic, and never mutates the catalog or ledger.
        _patch_pipeline(monkeypatch)
        request = TechnicalLibraryAdmissionPipelineRequest(
            selection=technical_5symbol_2022_v1_profile(),
            evaluation_start="2025-01-01",
            evaluation_end="2025-12-31 20:00",
            max_backtest_proposals=24,
        )
        report = run_technical_library_admission_pipeline(request)

        assert report.status == "COMPLETE"
        assert report.profile == "technical-5symbol-2022-v1"
        assert report.structural_combinations == 1000
        assert report.pair_compatible_count == 3
        assert report.shortlist_count == 3
        assert len(report.backtests) == 3
        assert [b.proposal_id for b in report.backtests] == [
            p.proposal_id for p in report.shortlist
        ]
        assert report.effective_start == "2022-05-04 12:00:00+00:00"
        serialized = report.to_report_dict()
        assert serialized["profile"] == "technical-5symbol-2022-v1"
        assert serialized["selection_counts"]["shortlist"] == 3
        assert len(serialized["shortlist"]) == 3
        assert len(serialized["backtests"]) == 3
        assert default_catalog().blueprints == {}

    def test_child_requests_begin_at_oos_boundary_and_disable_run_logging(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # LAP-05: every child request begins at the OOS boundary and disables run
        # logging, so 2025 metrics can never influence selection.
        captured = _patch_pipeline(monkeypatch)
        request = TechnicalLibraryAdmissionPipelineRequest(
            selection=technical_5symbol_2022_v1_profile(),
            evaluation_start="2025-01-01",
            evaluation_end="2025-12-31 20:00",
            max_backtest_proposals=24,
        )
        run_technical_library_admission_pipeline(request)

        assert captured
        for child in captured:
            assert pd.Timestamp(child.start, tz="UTC") == pd.Timestamp(
                "2025-01-01 00:00:00", tz="UTC",
            )
            assert child.log_run is False

    def test_fail_closed_selection_returns_report_without_backtests(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        failed = _canned_selection_report()
        failed = pytest_util_replace(failed, status="FAIL_CLOSED", proposals=())
        monkeypatch.setattr(pipeline_app, "resolve_common_technical_window", lambda *_: _canned_window())
        monkeypatch.setattr(pipeline_app, "run_technical_library_admission", lambda req: failed)
        request = TechnicalLibraryAdmissionPipelineRequest(
            selection=technical_5symbol_2022_v1_profile(),
            evaluation_start="2025-01-01",
            evaluation_end="2025-12-31 20:00",
            max_backtest_proposals=24,
        )
        report = run_technical_library_admission_pipeline(request)
        assert report.status == "FAIL_CLOSED"
        assert report.shortlist_count == 0
        assert report.backtests == ()


def pytest_util_replace(report: LibraryAdmissionReport, **changes: object) -> LibraryAdmissionReport:
    import dataclasses
    return dataclasses.replace(report, **changes)
