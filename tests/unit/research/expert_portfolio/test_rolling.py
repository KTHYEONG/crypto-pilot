from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.evaluation.metrics import Metrics
from src.research.evaluation.promotion import PromotionResult
from src.research.evaluation.reliability import (
    FoldDistributionResult,
    ReliabilityGateResult,
    compute_equal_duration_fold_distribution,
)
from src.research.expert_portfolio.admission_reports import LibraryAdmissionBacktestReport
from src.research.expert_portfolio.admission_types import AdmissionProposal, admission_proposal_id
from src.research.expert_portfolio.models import ContextualRouterSpec
from src.research.expert_portfolio.rolling import (
    RollingAdmissionConfig,
    build_rolling_rebalance_schedule,
    select_rebalance_proposal,
)


def _canned_report(
    proposal_id: str,
    lcb_obs: float,
    lcb_stress: float,
    *,
    verdict: str = "PASS",
    folds_pass: bool = True,
    mdd: float = -0.10,
) -> LibraryAdmissionBacktestReport:
    gate_obs = ReliabilityGateResult(
        lcb90_cagr=lcb_obs, lcb95_cagr=0.0, p_negative=0.1, point_cagr=0.02,
        t_stat=2.5, trade_count=40, block_size_used=1, verdict=verdict,
    )
    gate_stress = ReliabilityGateResult(
        lcb90_cagr=lcb_stress, lcb95_cagr=0.0, p_negative=0.2, point_cagr=0.01,
        t_stat=2.1, trade_count=40, block_size_used=1, verdict=verdict,
    )
    folds = FoldDistributionResult(
        n_folds=4, median_fold_cagr=0.01, worst_fold_cagr=0.0,
        median_fold_calmar=0.5, max_period_contribution=0.2, gate_pass=folds_pass,
    )
    metrics = Metrics(
        cagr=0.02, mdd=mdd, sharpe=0.5, sortino=0.6, calmar=2.0, profit_factor=1.5,
        expectancy=0.001, win_rate=0.5, payoff_ratio=1.1, trade_count=40,
        exposure=0.5, turnover=0.1, trades_per_year={"2024": 20, "2025": 20},
    )
    return LibraryAdmissionBacktestReport(
        status="COMPLETE",
        proposal_id=proposal_id,
        expert_ids=tuple(sorted(proposal_id.split(":")[1].split("|"))),
        router=ContextualRouterSpec("BTCUSDT", 48, 48, 96),
        window_start="2024-01-01 00:00:00+00:00",
        window_end="2025-12-31 20:00:00+00:00",
        observation_metrics=metrics,
        observation_gate=gate_obs,
        observation_folds=folds,
        stress_metrics=metrics,
        stress_gate=gate_stress,
        stress_folds=folds,
        promotion=PromotionResult(
            status="OBSERVATION_PASS", observation_verdict="PASS",
            fold_gate_pass=True, stress_verdict="PASS", holdout_verdict=None,
        ),
        allocation_cost_total=0.0,
        stress_allocation_cost_total=0.0,
        execution_workers=1,
    )


PROPOSAL_A = admission_proposal_id(("e0", "e1"))
PROPOSAL_B = admission_proposal_id(("e0", "e2"))


class TestBuildRollingRebalanceSchedule:
    def test_first_five_symbol_window_is_2024_q3_and_earlier_quarters_are_ineligible(
        self,
    ) -> None:
        # RLA-03: the first five-symbol window is 2024-07-01 and earlier
        # quarterly windows are ineligible for lack of common warm-up history.
        config = RollingAdmissionConfig()
        common_start = pd.Timestamp("2022-04-01", tz="UTC")

        earlier = build_rolling_rebalance_schedule(
            common_start, pd.Timestamp("2024-04-01", tz="UTC"), config,
        )
        assert earlier == ()

        windows = build_rolling_rebalance_schedule(
            common_start, pd.Timestamp("2024-07-01", tz="UTC"), config,
        )
        assert windows
        assert windows[0].rebalance_start == pd.Timestamp("2024-07-01", tz="UTC")
        assert windows[0].load_start >= common_start
        assert windows[0].status == "live_or_partial"

    def test_eight_closed_quarters_form_four_equal_six_month_folds(self) -> None:
        # RLA-07: replaying closed 2024-Q3 through 2026-Q2 produces four equal
        # six-month folds, not partial calendar-year folds.
        config = RollingAdmissionConfig()
        windows = build_rolling_rebalance_schedule(
            pd.Timestamp("2022-04-01", tz="UTC"),
            pd.Timestamp("2026-07-07 20:00", tz="UTC"),
            config,
        )
        closed = [w for w in windows if w.status == "closed"]
        assert len(closed) == 8
        assert closed[0].rebalance_start == pd.Timestamp("2024-07-01", tz="UTC")
        assert closed[-1].rebalance_start == pd.Timestamp("2026-04-01", tz="UTC")
        assert windows[-1].status == "live_or_partial"
        assert str(windows[-1].rebalance_start) == "2026-07-01 00:00:00+00:00"

        index = pd.date_range(
            closed[0].deploy_start, closed[-1].deploy_end, freq="4h", tz="UTC",
        )
        equity = pd.Series(np.linspace(1.0, 1.24, len(index)), index=index)
        folds = compute_equal_duration_fold_distribution(equity, fold_duration="6MS")
        assert folds.n_folds == 4
        assert 0.0 < folds.median_fold_cagr < 1.0

    def test_schedule_never_uses_a_moving_latest_data_end(self) -> None:
        config = RollingAdmissionConfig()
        windows = build_rolling_rebalance_schedule(
            pd.Timestamp("2022-04-01", tz="UTC"),
            pd.Timestamp("2026-07-07 20:00", tz="UTC"),
            config,
        )
        for window in windows:
            # observed_end = R - 4h, so no timestamp after R-4h ever enters.
            assert window.observed_end == window.rebalance_start - pd.Timedelta(hours=4)
            assert window.scored_start == (
                window.rebalance_start - pd.DateOffset(months=config.scoring_months)
            )
            assert window.load_start == (
                window.scored_start - config.warmup_period
            )

    def test_equal_duration_fold_rejects_malformed_duration_and_short_span(
        self,
    ) -> None:
        import pytest

        index = pd.date_range("2024-07-01", "2026-06-30 20:00", freq="4h", tz="UTC")
        equity = pd.Series(np.linspace(1.0, 1.24, len(index)), index=index)
        with pytest.raises(ValueError, match="month-start frequency"):
            compute_equal_duration_fold_distribution(equity, fold_duration="6ME")
        short = pd.Series([1.0, 1.01], index=pd.date_range("2024-07-01", periods=2, freq="4h", tz="UTC"))
        with pytest.raises(ValueError, match="does not admit"):
            compute_equal_duration_fold_distribution(short, fold_duration="6MS")

    def test_equal_duration_fold_rejects_invalid_equity_inputs(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="DatetimeIndex"):
            compute_equal_duration_fold_distribution(
                pd.Series([1.0, 1.1]), fold_duration="6MS",
            )
        single = pd.date_range("2024-07-01", periods=1, freq="4h", tz="UTC")
        with pytest.raises(ValueError, match="at least 2 points"):
            compute_equal_duration_fold_distribution(
                pd.Series([1.0], index=single), fold_duration="6MS",
            )
        index = pd.date_range("2024-07-01", periods=3, freq="4h", tz="UTC")
        non_monotonic = pd.Series([1.0, 1.1, 1.05], index=index[[0, 2, 1]])
        with pytest.raises(ValueError, match="monotonic"):
            compute_equal_duration_fold_distribution(non_monotonic, fold_duration="6MS")

    def test_equal_duration_fold_rejects_non_finite_and_non_positive_equity(
        self,
    ) -> None:
        import pytest

        index = pd.date_range("2024-07-01", periods=10, freq="4h", tz="UTC")
        non_finite = pd.Series(np.array([1.0, np.inf, *np.ones(8)]), index=index)
        with pytest.raises(ValueError, match="finite"):
            compute_equal_duration_fold_distribution(non_finite, fold_duration="6MS")
        non_positive = pd.Series(np.array([1.0, 0.0, *np.ones(8)]), index=index)
        with pytest.raises(ValueError, match="strictly positive"):
            compute_equal_duration_fold_distribution(non_positive, fold_duration="6MS")


class TestSelectRebalanceProposal:
    def test_deployable_challenger_replaces_incumbent_on_strictly_better_lcb(
        self,
    ) -> None:
        incumbent = AdmissionProposal(("e0", "e1"), True)
        reports = (
            _canned_report(PROPOSAL_A, lcb_obs=0.10, lcb_stress=0.10),
            _canned_report(PROPOSAL_B, lcb_obs=0.16, lcb_stress=0.16),
        )
        selected = select_rebalance_proposal(reports, incumbent)
        assert selected is not None
        assert selected.proposal_id == PROPOSAL_B

    def test_incumbent_is_retained_on_an_exact_primary_tie(self) -> None:
        # RLA-04: a challenger replaces the incumbent only when its first
        # selection-key term is strictly better; an exact tie retains the
        # incumbent library.
        incumbent = AdmissionProposal(("e0", "e1"), True)
        reports = (
            _canned_report(PROPOSAL_A, lcb_obs=0.10, lcb_stress=0.10),
            _canned_report(PROPOSAL_B, lcb_obs=0.10, lcb_stress=0.10),
        )
        selected = select_rebalance_proposal(reports, incumbent)
        assert selected is incumbent
        assert selected.proposal_id == PROPOSAL_A

    def test_no_deployable_report_yields_cash_not_stale_exposure(self) -> None:
        # RLA-04: no deployable proposal produces CASH (None) rather than
        # keeping a previously active library alive.
        incumbent = AdmissionProposal(("e0", "e1"), True)
        reports = (
            _canned_report(PROPOSAL_A, lcb_obs=0.0, lcb_stress=0.0, verdict="FAIL"),
            _canned_report(
                PROPOSAL_B, lcb_obs=0.05, lcb_stress=0.05, folds_pass=False,
            ),
        )
        assert select_rebalance_proposal(reports, incumbent) is None
        assert select_rebalance_proposal((), incumbent) is None

    def test_without_incumbent_the_best_deployable_proposal_wins(self) -> None:
        reports = (
            _canned_report(PROPOSAL_A, lcb_obs=0.08, lcb_stress=0.08),
            _canned_report(PROPOSAL_B, lcb_obs=0.16, lcb_stress=0.16),
        )
        selected = select_rebalance_proposal(reports, None)
        assert selected is not None
        assert selected.proposal_id == PROPOSAL_B
