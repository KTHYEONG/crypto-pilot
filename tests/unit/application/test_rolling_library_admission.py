from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.application.research.expert_portfolio import rolling_admission as ra
from src.application.research.expert_portfolio import window as window_module
from src.application.research.expert_portfolio.rebalance_ledger import load_rebalance_records
from src.application.research.expert_portfolio.window import (
    ResolvedTechnicalWindow,
    resolve_common_technical_window,
)
from src.common.errors import DataIntegrityError
from src.research.baseline.backtest import BacktestResult
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
    admission_proposal_id,
    technical_5symbol_rolling_v1_profile,
)
from src.research.expert_portfolio.backtest import ExpertPortfolioBacktestResult
from src.research.expert_portfolio.models import ContextualRouterSpec, ExpertDefinition
from src.research.expert_portfolio.rolling import build_rolling_rebalance_schedule


def _frame_from_index(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"close": np.ones(len(index))}, index=index)


def _canned_gate(lcb: float = 0.20) -> ReliabilityGateResult:
    return ReliabilityGateResult(
        lcb90_cagr=lcb, lcb95_cagr=lcb - 0.05, p_negative=0.0, point_cagr=lcb + 0.02,
        t_stat=2.5, trade_count=40, block_size_used=1, verdict="PASS",
    )


def _canned_folds() -> FoldDistributionResult:
    return FoldDistributionResult(
        n_folds=4, median_fold_cagr=0.01, worst_fold_cagr=0.0,
        median_fold_calmar=0.5, max_period_contribution=0.2, gate_pass=True,
    )


def _canned_metrics() -> Metrics:
    return Metrics(
        cagr=0.02, mdd=-0.01, sharpe=0.5, sortino=0.6, calmar=2.0,
        profit_factor=1.5, expectancy=0.001, win_rate=0.5, payoff_ratio=1.1,
        trade_count=40, exposure=0.5, turnover=0.1, trades_per_year={"2024": 20, "2025": 20},
    )


def _training_report(proposal_id: str) -> LibraryAdmissionBacktestReport:
    expert_ids = tuple(sorted(proposal_id.split(":")[1].split("|")))
    return LibraryAdmissionBacktestReport(
        status="COMPLETE",
        proposal_id=proposal_id,
        expert_ids=expert_ids,
        router=ContextualRouterSpec("BTCUSDT", 48, 48, 96),
        window_start="2022-07-01 00:00:00+00:00",
        window_end="2024-06-30 20:00:00+00:00",
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
    return LibraryAdmissionReport(
        status="COMPLETE",
        window_start="2022-07-01 00:00:00+00:00",
        window_end="2024-06-30 20:00:00+00:00",
        experts=experts,
        candidates=candidates,
        proposals=(),
        context_coverage=dict.fromkeys(
            ("up_low_vol", "up_high_vol", "down_low_vol", "down_high_vol", "flat_low_vol", "flat_high_vol"),
            500,
        ),
        covered_states=6,
        coverage_sufficient=True,
        router=ContextualRouterSpec("BTCUSDT", 48, 48, 96),
        admission=LibraryAdmissionConfig(2, 5, 20, 200, 0.50, 0.15, 6, 1_000_000),
        structural_combinations=100,
    )


def _proposal_for_window(window: object) -> str:
    rebalance_start = window.rebalance_start
    return admission_proposal_id((f"e{rebalance_start.month}", f"f{rebalance_start.day}"))


def _fake_select(profile, window, config, incumbent):
    proposal_id = _proposal_for_window(window)
    proposal = AdmissionProposal(
        tuple(sorted(proposal_id.split(":")[1].split("|"))), True,
    )
    return _canned_selection_report(), (proposal,), (_training_report(proposal_id),)


def _fake_deployment(selected, profile, window, config) -> pd.Series:
    index = pd.date_range(
        window.deploy_start, window.deploy_end, freq="4h", tz="UTC",
    )
    return pd.Series(1.0, index=index, name="equity", dtype="float64")


def _canned_window(
    common_start: str = "2022-04-01 00:00:00+00:00",
    common_end: str = "2026-07-07 20:00:00+00:00",
) -> ResolvedTechnicalWindow:
    return ResolvedTechnicalWindow(
        requested_start=None,
        common_start=pd.Timestamp(common_start, tz="UTC"),
        common_end=pd.Timestamp(common_end, tz="UTC"),
        effective_start=pd.Timestamp("2022-05-04 12:00:00", tz="UTC"),
        end=None,
        symbol_sources={},
    )


def _patch_service(monkeypatch: pytest.MonkeyPatch, *, data_hash: str = "hash") -> None:
    monkeypatch.setattr(ra, "resolve_common_technical_window", lambda *_: _canned_window())
    monkeypatch.setattr(ra, "_select_for_window", _fake_select)
    monkeypatch.setattr(ra, "_deployment_equity", _fake_deployment)
    monkeypatch.setattr(ra, "technical_data_hashes", lambda symbol: {symbol: data_hash})


def _request(as_of: str, **changes: object) -> ra.RollingLibraryAdmissionRequest:
    import dataclasses
    request = ra.RollingLibraryAdmissionRequest(
        profile=technical_5symbol_rolling_v1_profile(),
        as_of=as_of,
        config=ra.RollingAdmissionConfig(),
    )
    if changes:
        return dataclasses.replace(request, **changes)
    return request


class TestCommonWindowResolverEnd:
    def test_resolver_reports_earliest_full_common_end(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # RLA-01: the resolver reports the earliest full common end across every
        # symbol and never extends past a settled funding boundary.
        def fake_ohlcv(path, *, start=None, end=None):
            if "ETHUSDT" in str(path):
                index = pd.date_range("2020-01-01", "2026-07-05 08:00", freq="4h", tz="UTC")
            else:
                index = pd.date_range("2020-01-01", "2026-07-07 16:00", freq="4h", tz="UTC")
            return _frame_from_index(index)

        def fake_funding(path):
            if "ETHUSDT" in str(path):
                index = pd.date_range("2020-01-01", "2026-07-05 12:00", freq="4h", tz="UTC")
            else:
                index = pd.date_range("2020-01-01", "2026-07-07 12:00", freq="4h", tz="UTC")
            return pd.Series(0.0, index=index)

        monkeypatch.setattr(window_module, "load_ohlcv_4h", fake_ohlcv)
        monkeypatch.setattr(window_module, "load_funding_rates", fake_funding)
        resolved = resolve_common_technical_window(
            ("BTCUSDT", "ETHUSDT"), None, None,
        )
        assert resolved.common_end == pd.Timestamp("2026-07-05 08:00", tz="UTC")
        assert resolved.symbol_sources["BTCUSDT"]["ohlcv_end"] == "2026-07-07 16:00:00+00:00"
        assert resolved.symbol_sources["BTCUSDT"]["funding_end"] == "2026-07-07 12:00:00+00:00"

    def test_funding_boundary_never_extends_the_last_settled_bar(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # RLA-01: a funding timestamp exactly on a 4h boundary settles the
        # preceding bar; the partial current bar is excluded.
        def fake_ohlcv(path, *, start=None, end=None):
            index = pd.date_range("2020-01-01", "2026-07-07 12:00", freq="4h", tz="UTC")
            return _frame_from_index(index)

        def fake_funding(path):
            index = pd.date_range("2020-01-01", "2026-07-07 12:00", freq="4h", tz="UTC")
            return pd.Series(0.0, index=index)

        monkeypatch.setattr(window_module, "load_ohlcv_4h", fake_ohlcv)
        monkeypatch.setattr(window_module, "load_funding_rates", fake_funding)
        resolved = resolve_common_technical_window(("BTCUSDT",), None, None)
        assert resolved.common_end == pd.Timestamp("2026-07-07 08:00", tz="UTC")

    def test_requested_end_caps_common_end_to_the_causal_snapshot(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_ohlcv(path, *, start=None, end=None):
            index = pd.date_range("2020-01-01", "2026-07-07 12:00", freq="4h", tz="UTC")
            return _frame_from_index(index)

        def fake_funding(path):
            index = pd.date_range("2020-01-01", "2026-07-07 12:00", freq="4h", tz="UTC")
            return pd.Series(0.0, index=index)

        monkeypatch.setattr(window_module, "load_ohlcv_4h", fake_ohlcv)
        monkeypatch.setattr(window_module, "load_funding_rates", fake_funding)
        resolved = resolve_common_technical_window(
            ("BTCUSDT",), None, "2026-01-01 00:00:00+00:00",
        )
        assert resolved.common_end == pd.Timestamp("2025-12-31 20:00", tz="UTC")

    def test_non_overlapping_sources_fail_closed_naming_the_symbol(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_ohlcv(path, *, start=None, end=None):
            index = pd.date_range("2020-01-01", "2020-06-01", freq="4h", tz="UTC")
            return _frame_from_index(index)

        def fake_funding(path):
            index = pd.date_range("2021-01-01", "2021-06-01", freq="8h", tz="UTC")
            return pd.Series(0.0, index=index)

        monkeypatch.setattr(window_module, "load_ohlcv_4h", fake_ohlcv)
        monkeypatch.setattr(window_module, "load_funding_rates", fake_funding)
        with pytest.raises(DataIntegrityError, match=r"BTCUSDT.*overlap"):
            resolve_common_technical_window(("BTCUSDT",), None, None)


class TestRollingLibraryAdmission:
    def test_future_data_cannot_alter_a_completed_rebalance_decision(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory,
    ) -> None:
        # RLA-02: no OHLCV, funding, candidate return, or context label later
        # than R-4h enters selection, and appending future data cannot change an
        # already completed rebalance record.
        _patch_service(monkeypatch)
        ledger = tmp_path / "rebalance.jsonl"
        pointer = tmp_path / "current.json"

        report1 = ra.run_rolling_library_admission(
            _request("2024-10-01 00:00:00+00:00"),
            ledger_path=ledger, pointer_path=pointer,
            audit_path=tmp_path / "rolling_candidate_audit.jsonl",
        )
        first_record = next(r for r in report1.records if r.rebalance_start.startswith("2024-07-01"))
        assert first_record.observed_end == "2024-06-30 20:00:00+00:00"

        report2 = ra.run_rolling_library_admission(
            _request("2026-07-07 20:00:00+00:00"),
            ledger_path=ledger, pointer_path=pointer,
            audit_path=tmp_path / "rolling_candidate_audit.jsonl",
        )
        replay_record = next(r for r in report2.records if r.rebalance_start.startswith("2024-07-01"))
        assert replay_record == first_record
        assert replay_record.proposal_id == first_record.proposal_id
        assert replay_record.snapshot_key == first_record.snapshot_key
        matching = [
            r for r in load_rebalance_records(ledger)
            if r.rebalance_start == first_record.rebalance_start
        ]
        assert len(matching) == 1

    def test_replay_of_an_identical_snapshot_is_idempotent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory,
    ) -> None:
        # RLA-05: re-running the same R and identical snapshot appends nothing.
        _patch_service(monkeypatch)
        ledger = tmp_path / "rebalance.jsonl"
        pointer = tmp_path / "current.json"

        report1 = ra.run_rolling_library_admission(
            _request("2026-07-07 20:00:00+00:00"),
            ledger_path=ledger, pointer_path=pointer,
            audit_path=tmp_path / "rolling_candidate_audit.jsonl",
        )
        report2 = ra.run_rolling_library_admission(
            _request("2026-07-07 20:00:00+00:00"),
            ledger_path=ledger, pointer_path=pointer,
            audit_path=tmp_path / "rolling_candidate_audit.jsonl",
        )
        lines = [
            line for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        assert len(lines) == len(report1.records)
        assert [r.to_payload() for r in report2.records] == [r.to_payload() for r in report1.records]

    def test_switch_cost_is_charged_once_and_unchanged_selections_do_not_turn_over(
        self,
    ) -> None:
        # RLA-06: a switch charges execution costs once at the next available
        # bar; a same-library rebalance never creates a synthetic switch.
        config = ra.RollingAdmissionConfig(switch_cost=0.0008, initial_equity=10_000.0)
        index_a = pd.date_range("2024-07-01", periods=4, freq="4h", tz="UTC")
        index_b = pd.date_range("2024-10-01", periods=4, freq="4h", tz="UTC")
        flat = lambda index: pd.Series(1.0, index=index)  # noqa: E731

        switched = ra._stitch_segments(
            [flat(index_a), flat(index_b)],
            ["lae-v1:a|b", "lae-v1:a|c"],
            config,
        )
        assert switched.iloc[-1] == pytest.approx(10_000.0 * (1.0 - 0.0008))
        second_segment_start = switched.loc[index_b[0]]
        assert second_segment_start == pytest.approx(10_000.0 * (1.0 - 0.0008))

        unchanged = ra._stitch_segments(
            [flat(index_a), flat(index_b)],
            ["lae-v1:a|b", "lae-v1:a|b"],
            config,
        )
        assert unchanged.iloc[-1] == pytest.approx(10_000.0)

    def test_incomplete_historical_segment_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory,
    ) -> None:
        # The scheduler leaf rejects a request for an incomplete historical
        # segment: the current 2026-Q3 quarter is live at 2026-07-07.
        _patch_service(monkeypatch)
        request = _request(
            "2026-07-07 20:00:00+00:00",
            require_complete_history=True,
        )
        with pytest.raises(ValueError, match="incomplete historical segment"):
            ra.run_rolling_library_admission(
                request,
                ledger_path=tmp_path / "rebalance.jsonl",
                pointer_path=tmp_path / "current.json",
                audit_path=tmp_path / "rolling_candidate_audit.jsonl",
            )

    def test_completed_window_audit_is_persisted_before_later_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory,
    ) -> None:
        _patch_service(monkeypatch)
        calls = 0

        def fail_on_second(profile, window, config, incumbent):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic later-window failure")
            return _fake_select(profile, window, config, incumbent)

        monkeypatch.setattr(ra, "_select_for_window", fail_on_second)
        audit_path = tmp_path / "rolling_candidate_audit.jsonl"
        with pytest.raises(RuntimeError, match="synthetic later-window failure"):
            ra.run_rolling_library_admission(
                _request("2025-01-01 00:00:00+00:00"),
                ledger_path=tmp_path / "rebalance.jsonl",
                pointer_path=tmp_path / "current.json",
                audit_path=audit_path,
            )

        assert len(audit_path.read_text(encoding="utf-8").splitlines()) == 1

    def test_live_mode_fails_closed_without_separate_authorization(self) -> None:
        with pytest.raises(RuntimeError, match="separate authorization"):
            _request("2026-07-07 20:00:00+00:00", mode="live", live_authorized=False)
        request = _request("2026-07-07 20:00:00+00:00", mode="live", live_authorized=True)
        assert request.mode == "live"


def _canned_portfolio(index: pd.DatetimeIndex, equity: pd.Series | None = None) -> ExpertPortfolioBacktestResult:
    if equity is None:
        equity = pd.Series(1.001, index=index, name="equity")
    return ExpertPortfolioBacktestResult(
        backtest_result=BacktestResult(
            equity=equity, trades=pd.DataFrame(), signals=pd.DataFrame(),
        ),
        target_weights=pd.DataFrame(
            {"A": 0.5, "B": 0.0, "CASH": 0.5}, index=index,
        ),
        allocation_cost=pd.Series(0.0, index=index),
        component_returns=pd.DataFrame(index=index),
    )


class TestRollingInternalPaths:
    def test_request_validates_config_universe_and_mode(self) -> None:
        with pytest.raises(ValueError, match="must match the profile symbols"):
            ra.RollingLibraryAdmissionRequest(
                profile=technical_5symbol_rolling_v1_profile(),
                as_of="2026-07-07 20:00:00+00:00",
                config=ra.RollingAdmissionConfig(symbols=("BTCUSDT",)),
            )
        with pytest.raises(ValueError, match="mode must be"):
            _request("2026-07-07 20:00:00+00:00", mode="paperless")

    def test_no_eligible_window_returns_no_windows_report(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory,
    ) -> None:
        _patch_service(monkeypatch)
        report = ra.run_rolling_library_admission(
            _request("2022-06-01 00:00:00+00:00"),
            ledger_path=tmp_path / "rebalance.jsonl",
            pointer_path=tmp_path / "current.json",
            audit_path=tmp_path / "rolling_candidate_audit.jsonl",
        )
        assert report.status == "NO_WINDOWS"
        assert report.records == ()

    def test_build_candidate_panel_runs_the_full_frozen_universe(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        profile = technical_5symbol_rolling_v1_profile()
        index = pd.date_range("2022-05-28", "2024-06-30 20:00", freq="4h", tz="UTC")
        returns = pd.Series(
            np.where(np.arange(len(index)) % 2 == 0, 0.001, -0.001), index=index,
        )

        def fake_worker(symbol, sources, start, end):
            return dict.fromkeys(sources, (returns, 30))

        monkeypatch.setattr(ra, "_symbol_admission_worker", fake_worker)
        panel, trade_counts, definitions, code_hash = ra._build_candidate_panel(
            profile, "2022-05-28 12:00", "2024-06-30 20:00",
        )
        assert len(definitions) == 90
        assert len(trade_counts) == 90
        assert panel.shape[0] == len(index)
        assert code_hash

    def test_slice_scored_panel_restores_the_initial_nan_row(self) -> None:
        index = pd.date_range("2022-05-28", periods=400, freq="4h", tz="UTC")
        panel = pd.DataFrame(
            {
                "a": np.linspace(0.001, 0.01, len(index)),
                "b": np.linspace(0.002, 0.02, len(index)),
            },
            index=index,
        )
        scored = ra._slice_scored_panel(panel, pd.Timestamp("2022-07-01", tz="UTC"))
        assert scored.index[0] == pd.Timestamp("2022-07-01", tz="UTC")
        assert scored.iloc[0].isna().all()
        assert len(scored) < len(panel)
        with pytest.raises(DataIntegrityError, match="fewer than 2 bars"):
            ra._slice_scored_panel(panel, pd.Timestamp("2099-01-01", tz="UTC"))

    def test_run_proposal_backtest_composes_gates_without_holdout_seal(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        index = pd.date_range("2022-07-01", periods=200, freq="4h", tz="UTC")
        panel = pd.DataFrame(
            {
                "technical_ema_alignment_long_v1:BTCUSDT": np.linspace(0.001, 0.01, 200),
                "technical_macd_histogram_regime_long_v1:ETHUSDT": np.linspace(0.002, 0.02, 200),
            },
            index=index,
        )
        trades = pd.DataFrame({"entry_bar": [0], "exit_bar": [10], "pnl": [1.0], "return_pct": [0.01]})
        context = pd.Series(["up_low_vol"] * 200, index=index)

        monkeypatch.setattr(ra, "_run_selected_tasks", lambda *a, **k: ({}, 1))
        monkeypatch.setattr(ra, "_assemble_selected_panel", lambda ev, defs: (panel, trades))
        monkeypatch.setattr(ra, "_build_admission_context", lambda router, idx, s, e: context)
        monkeypatch.setattr(ra, "run_expert_portfolio", lambda *a, **k: _canned_portfolio(index))
        monkeypatch.setattr(ra, "compute_metrics", lambda eq, tr: _canned_metrics())

        def _fake_gate(eq, n, config=None):
            return _canned_gate()

        monkeypatch.setattr(ra, "compute_equity_reliability_gate", _fake_gate)
        monkeypatch.setattr(ra, "compute_fold_distribution", lambda result: _canned_folds())
        monkeypatch.setattr(
            ra, "compose_promotion_verdict",
            lambda *a, **k: PromotionResult(
                status="OBSERVATION_PASS", observation_verdict="PASS",
                fold_gate_pass=True, stress_verdict="PASS", holdout_verdict=None,
            ),
        )
        monkeypatch.setattr(ra, "technical_data_hashes", lambda symbol: {symbol: "h"})

        report = ra._run_proposal_backtest(
            (
                "technical_ema_alignment_long_v1:BTCUSDT",
                "technical_macd_histogram_regime_long_v1:ETHUSDT",
            ),
            ContextualRouterSpec("BTCUSDT", 48, 48, 96),
            "2022-07-01 00:00:00+00:00",
            "2024-06-30 20:00:00+00:00",
            10_000.0,
            None,
        )
        assert report.status == "COMPLETE"
        assert report.observation_gate.verdict == "PASS"
        assert report.stress_gate.verdict == "PASS"
        assert report.observation_folds.gate_pass is True

    def test_select_for_window_shortlists_backtests_and_reincludes_incumbent(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        profile = technical_5symbol_rolling_v1_profile()
        config = ra.RollingAdmissionConfig()
        window = build_rolling_rebalance_schedule(
            pd.Timestamp("2022-04-01", tz="UTC"),
            pd.Timestamp("2024-10-01", tz="UTC"),
            config,
        )[0]
        proposal_a = AdmissionProposal(("e0", "e1"), True)
        proposal_b = AdmissionProposal(("e0", "e2"), True)
        selection_report = _canned_selection_report()
        import dataclasses
        selection_report = dataclasses.replace(
            selection_report, proposals=(proposal_a, proposal_b),
        )
        ctx_index = pd.date_range("2022-07-01", periods=50, freq="4h", tz="UTC")
        monkeypatch.setattr(ra, "_build_candidate_panel", lambda p, s, e: (pd.DataFrame(), {}, (), "h"))
        monkeypatch.setattr(ra, "_slice_scored_panel", lambda panel, start: pd.DataFrame(index=ctx_index))
        monkeypatch.setattr(
            ra, "_build_admission_context",
            lambda router, idx, s, e: pd.Series(["up_low_vol"] * len(ctx_index), index=ctx_index),
        )
        monkeypatch.setattr(ra, "evaluate_library_admission", lambda *a, **k: selection_report)
        monkeypatch.setattr(
            ra, "shortlist_admission_proposals",
            lambda proposals, budget: tuple(p for p in proposals if p.proposal_id == proposal_a.proposal_id),
        )
        monkeypatch.setattr(
            ra, "_run_proposal_backtest",
            lambda ids, router, s, e, eq, w: _training_report(admission_proposal_id(ids)),
        )
        selection, shortlist, reports = ra._select_for_window(profile, window, config, proposal_b)
        assert selection.status == "COMPLETE"
        assert {p.proposal_id for p in shortlist} == {proposal_a.proposal_id, proposal_b.proposal_id}
        assert len(reports) == 2
        assert {r.proposal_id for r in reports} == {proposal_a.proposal_id, proposal_b.proposal_id}

    def test_deployment_equity_cash_is_flat_and_selected_slices_from_deploy_start(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        profile = technical_5symbol_rolling_v1_profile()
        config = ra.RollingAdmissionConfig()
        window = build_rolling_rebalance_schedule(
            pd.Timestamp("2022-04-01", tz="UTC"),
            pd.Timestamp("2024-10-01", tz="UTC"),
            config,
        )[0]
        cash = ra._deployment_equity(None, profile, window, config)
        assert cash.index[0] == window.deploy_start
        assert cash.index[-1] == window.deploy_end
        assert (cash == 1.0).all()

        index = pd.date_range(window.load_start, window.deploy_end, freq="4h", tz="UTC")
        equity = pd.Series(2.0, index=index, name="equity")
        monkeypatch.setattr(ra, "_run_selected_tasks", lambda *a, **k: ({}, 1))
        monkeypatch.setattr(
            ra, "_assemble_selected_panel",
            lambda ev, defs: (pd.DataFrame(index=index), pd.DataFrame()),
        )
        monkeypatch.setattr(
            ra, "_build_admission_context",
            lambda router, idx, s, e: pd.Series(["up_low_vol"] * len(idx), index=idx),
        )
        monkeypatch.setattr(
            ra, "run_expert_portfolio",
            lambda *a, **k: _canned_portfolio(index, equity=equity),
        )
        selected = AdmissionProposal(("technical_ema_alignment_long_v1:BTCUSDT",), True)
        segment = ra._deployment_equity(selected, profile, window, config)
        assert segment.index[0] == window.deploy_start
        assert segment.iloc[0] == pytest.approx(1.0)
        assert segment.iloc[-1] == pytest.approx(1.0)

    def test_stitch_segments_empty_and_fold_summary_short(self) -> None:
        config = ra.RollingAdmissionConfig()
        assert ra._stitch_segments([], [], config).empty
        folds = ra._fold_summary(pd.Series([1.0], dtype="float64"))
        assert folds.n_folds == 0
        assert folds.gate_pass is True


class TestRollingCandidateAuditLedger:
    def test_repeated_audit_append_leaves_one_line_and_returns_original(
        self, tmp_path: pytest.TempPathFactory,
    ) -> None:
        # RAP-02: an identical replay returns the stored record without writing.
        from src.application.research.expert_portfolio.rebalance_ledger import (
            append_rolling_candidate_audit,
        )
        from src.research.expert_portfolio.rolling import (
            RollingCandidateAuditRecord,
        )

        audit_path = tmp_path / "rolling_candidate_audit.jsonl"
        window = build_rolling_rebalance_schedule(
            pd.Timestamp("2022-04-01", tz="UTC"),
            pd.Timestamp("2024-10-01", tz="UTC"),
            ra.RollingAdmissionConfig(),
        )[0]
        selection = _canned_selection_report()
        proposal = AdmissionProposal(("e0", "e1"), True)
        audit = RollingCandidateAuditRecord.from_selection(
            window, selection, (proposal,), (_training_report(proposal.proposal_id),),
            proposal, "snap-key",
        )
        stored1 = append_rolling_candidate_audit(audit, audit_path)
        stored2 = append_rolling_candidate_audit(audit, audit_path)
        assert stored2 == stored1
        assert stored2.snapshot_key == "snap-key"
        lines = [
            line for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        assert len(lines) == 1

    def test_malformed_existing_audit_line_fails_closed(
        self, tmp_path: pytest.TempPathFactory,
    ) -> None:
        # RAP-02: a malformed audit JSONL line raises ValueError, never skipped.
        from src.application.research.expert_portfolio.rebalance_ledger import (
            append_rolling_candidate_audit,
        )
        from src.research.expert_portfolio.rolling import RollingCandidateAuditRecord

        audit_path = tmp_path / "rolling_candidate_audit.jsonl"
        audit_path.write_text("not-json\n", encoding="utf-8")
        window = build_rolling_rebalance_schedule(
            pd.Timestamp("2022-04-01", tz="UTC"),
            pd.Timestamp("2024-10-01", tz="UTC"),
            ra.RollingAdmissionConfig(),
        )[0]
        audit = RollingCandidateAuditRecord.from_selection(
            window, _canned_selection_report(), (), (), None, "snap-key",
        )
        with pytest.raises(ValueError, match="malformed rolling candidate audit line"):
            append_rolling_candidate_audit(audit, audit_path)
