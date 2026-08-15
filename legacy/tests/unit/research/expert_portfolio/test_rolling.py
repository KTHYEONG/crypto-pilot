from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.evaluation.metrics import Metrics
from src.research.evaluation.promotion import PromotionResult
from src.research.evaluation.reliability import (
    FoldDistributionResult,
    ReliabilityGateResult,
    compute_equal_duration_fold_distribution,
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
)
from src.research.expert_portfolio.models import ContextualRouterSpec, ExpertDefinition
from src.research.expert_portfolio.rolling import (
    RollingAdmissionConfig,
    RollingCandidateAuditRecord,
    build_rolling_rebalance_schedule,
    resolve_dynamic_shortlist_budget,
    rolling_admission_config_for_profile,
    select_rebalance_proposal,
    select_symbols_for_window,
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


def _canned_selection_report() -> LibraryAdmissionReport:
    experts = tuple(
        ExpertDefinition(
            f"e{i}", f"source_{i}", f"f{i}", (f"S{i}",), "run_technical_expert", "h" * 64,
        )
        for i in range(3)
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


def _canned_window() -> object:
    return build_rolling_rebalance_schedule(
        pd.Timestamp("2022-04-01", tz="UTC"),
        pd.Timestamp("2024-10-01", tz="UTC"),
        RollingAdmissionConfig(),
    )[0]


class TestRollingCandidateAuditRecord:
    def test_equivalent_inputs_produce_byte_stable_payloads(self) -> None:
        # RAP-01: identical audit inputs serialize to byte-identical canonical JSON.
        window = _canned_window()
        selection = _canned_selection_report()
        proposal = AdmissionProposal(("e0", "e1"), True)
        training = (
            _canned_report(PROPOSAL_A, lcb_obs=0.10, lcb_stress=0.10),
            _canned_report(PROPOSAL_B, lcb_obs=0.12, lcb_stress=0.12),
        )
        audit1 = RollingCandidateAuditRecord.from_selection(
            window, selection, (proposal,), training, proposal, "snap-key",
        )
        audit2 = RollingCandidateAuditRecord.from_selection(
            window, selection, (proposal,), training, proposal, "snap-key",
        )
        assert audit1.to_canonical_bytes() == audit2.to_canonical_bytes()
        payload = audit1.to_payload()
        assert payload["profile"] == window.profile
        assert payload["rebalance_start"] == str(window.rebalance_start)
        assert payload["snapshot_key"] == "snap-key"
        assert payload["selected"]["proposal_id"] == proposal.proposal_id
        assert payload["shortlist"] == [proposal.proposal_id]
        assert len(payload["candidates"]) == 3
        assert len(payload["training"]) == 2
        assert "recorded_at" not in payload

    def test_no_deployable_selection_records_explicit_cash_outcome(self) -> None:
        # RAP-01: a no-deployable window is serialized as CASH with a reason.
        audit = RollingCandidateAuditRecord.from_selection(
            _canned_window(), _canned_selection_report(), (), (), None, "snap-key",
        )
        payload = audit.to_payload()
        assert audit.selection_status == "cash"
        assert audit.cash_reason == "no_shortlist_or_incumbent"
        assert payload["selected"] is None
        assert payload["incumbent_kept"] is False

    def test_from_selection_orders_shortlist_and_records_gates(self) -> None:
        # RAP-01: shortlist order, per-proposal base/stress gate verdicts, and
        # the selection key are all captured deterministically.
        window = _canned_window()
        selection = _canned_selection_report()
        proposal_a = AdmissionProposal(("e0", "e1"), True)
        proposal_b = AdmissionProposal(("e0", "e2"), True)
        training = (
            _canned_report(PROPOSAL_B, lcb_obs=0.16, lcb_stress=0.16),
            _canned_report(PROPOSAL_A, lcb_obs=0.10, lcb_stress=0.10),
        )
        audit = RollingCandidateAuditRecord.from_selection(
            window, selection, (proposal_b, proposal_a), training, proposal_b, "snap-key",
        )
        payload = audit.to_payload()
        assert payload["shortlist"] == [proposal_b.proposal_id, proposal_a.proposal_id]
        assert payload["training"][0]["proposal_id"] == PROPOSAL_B
        assert payload["training"][0]["observation_gate_verdict"] == "PASS"
        assert payload["training"][0]["stress_gate_verdict"] == "PASS"
        assert payload["selected"]["proposal_id"] == proposal_b.proposal_id


class TestRollingProfileConfig:
    def test_canonical_config_is_single_priority_path(self) -> None:
        # The rolling path has exactly one canonical config: the priority
        # family-unique search with per-symbol-winner routing hardcoded and the
        # shared one-bar base scenario.
        symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
        config = rolling_admission_config_for_profile("technical-5symbol-rolling", symbols)
        assert config.profile == "technical-5symbol-rolling"
        assert config.base_delay_bars == 1
        assert config.min_shortlist_budget == 8
        assert config.hurdle_cost_multiple == 2.0
        assert config.family_symbol_prefilter_top_k is None
        assert "proposal_search" not in config.fingerprint()
        assert "router_kind" not in config.fingerprint()
        assert "shortlist_budget" not in config.fingerprint()

    def test_unknown_profile_name_fails_closed(self) -> None:
        # An unknown rolling profile name raises ValueError.
        with pytest.raises(ValueError, match="unknown rolling profile"):
            rolling_admission_config_for_profile("technical-5symbol-rolling-v9", ("BTCUSDT",))

    def test_rolling_admission_config_rejects_invalid_new_fields(self) -> None:
        # The new execution-policy knobs fail closed on invalid values.
        with pytest.raises(ValueError, match="min_shortlist_budget"):
            RollingAdmissionConfig(min_shortlist_budget=0)
        with pytest.raises(ValueError, match="max_backtest_wall_seconds_per_window"):
            RollingAdmissionConfig(max_backtest_wall_seconds_per_window=0.0)
        with pytest.raises(ValueError, match="hurdle_cost_multiple"):
            RollingAdmissionConfig(hurdle_cost_multiple=-1.0)

    def test_rolling_admission_config_rejects_prefilter_top_k_below_one(self) -> None:
        # family_symbol_prefilter_top_k must be None or >= 1.
        with pytest.raises(ValueError, match="family_symbol_prefilter_top_k"):
            RollingAdmissionConfig(family_symbol_prefilter_top_k=0)

    def test_canonical_profile_fingerprint_and_resolution_are_stable(self) -> None:
        # The canonical profile's config fingerprint is stable and the resolver
        # matches the builder exactly; legacy v1/v2 profile names no longer
        # resolve.
        from src.research.expert_portfolio.admission_types import (
            ROLLING_LIBRARY_ADMISSION_PROFILES,
            resolve_rolling_library_admission_profile,
            technical_5symbol_rolling_profile,
        )

        symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
        config = rolling_admission_config_for_profile("technical-5symbol-rolling", symbols)
        expected_keys = {
            "profile", "timeframe", "symbols", "scoring_months", "warmup_bars",
            "min_shortlist_budget", "max_backtest_wall_seconds_per_window",
            "min_context_samples", "rebalance_months", "switch_cost",
            "initial_equity", "base_delay_bars", "family_symbol_prefilter_top_k",
            "hurdle_cost_multiple", "dynamic_symbol_selection",
            "symbol_universe", "symbol_top_k",
        }
        assert set(config.fingerprint()) == expected_keys
        assert config.fingerprint()["family_symbol_prefilter_top_k"] is None
        assert config.fingerprint()["timeframe"] == "4h"
        assert config.fingerprint()["dynamic_symbol_selection"] is False
        assert config.fingerprint()["symbol_top_k"] == 5
        assert (
            resolve_rolling_library_admission_profile("technical-5symbol-rolling")
            == technical_5symbol_rolling_profile()
        )
        assert "technical-5symbol-rolling" in ROLLING_LIBRARY_ADMISSION_PROFILES
        assert set(ROLLING_LIBRARY_ADMISSION_PROFILES) == {"technical-5symbol-rolling"}

    def test_rolling_profile_v1_v2_names_no_longer_resolve(self) -> None:
        # Versioned profile names are retired: only the version-less canonical
        # name resolves, and the legacy v1/v2/v3 builder names are gone.
        from src.research.expert_portfolio.admission_types import (
            resolve_rolling_library_admission_profile,
        )
        from src.research.expert_portfolio.contracts import (
            resolve_rolling_library_admission_profile as facade_resolve,
        )

        assert (
            resolve_rolling_library_admission_profile("technical-5symbol-rolling")
            is not None
        )
        for retired in (
            "technical-5symbol-rolling-v1",
            "technical-5symbol-rolling-v2",
            "technical-5symbol-rolling-v3",
        ):
            with pytest.raises(ValueError, match="unknown rolling library admission profile"):
                resolve_rolling_library_admission_profile(retired)
            with pytest.raises(ValueError, match="unknown rolling library admission profile"):
                facade_resolve(retired)

    def test_resolve_dynamic_shortlist_budget_scales_with_measured_wall_time(self) -> None:
        # The effective budget scales with measured probe wall-times: faster
        # probes license more backtests within the same time budget.
        budget = resolve_dynamic_shortlist_budget(
            (1.0, 1.0, 1.0, 1.0), 120.0, 8,
        )
        assert budget == 120

    def test_resolve_dynamic_shortlist_budget_floors_at_min_shortlist_budget(self) -> None:
        # A slow probe never drops the budget below the structural minimum.
        budget = resolve_dynamic_shortlist_budget((100.0,), 120.0, 8)
        assert budget == 8

    def test_resolve_dynamic_shortlist_budget_rejects_empty_probe(self) -> None:
        # An empty probe cannot extrapolate a budget and fails closed.
        with pytest.raises(ValueError, match="must not be empty"):
            resolve_dynamic_shortlist_budget((), 120.0, 8)

    def test_resolve_dynamic_shortlist_budget_rejects_invalid_budget_knobs(self) -> None:
        with pytest.raises(ValueError, match="min_shortlist_budget"):
            resolve_dynamic_shortlist_budget((1.0,), 120.0, 0)
        with pytest.raises(ValueError, match="max_backtest_wall_seconds_per_window"):
            resolve_dynamic_shortlist_budget((1.0,), 0.0, 8)
        with pytest.raises(ValueError, match="must all be positive"):
            resolve_dynamic_shortlist_budget((0.0,), 120.0, 8)

    def test_rolling_admission_config_rejects_invalid_timeframe(self) -> None:
        # timeframe must be a canonical research bucket; anything else fails closed.
        with pytest.raises(ValueError, match="timeframe"):
            RollingAdmissionConfig(timeframe="3h")
        with pytest.raises(ValueError, match="timeframe"):
            rolling_admission_config_for_profile(
                "technical-5symbol-rolling", ("BTCUSDT",), timeframe="5h",
            )

    def test_rolling_admission_config_rejects_invalid_dynamic_symbol_fields(self) -> None:
        # symbol_top_k must be >= 1 and symbol_universe non-empty/unique.
        with pytest.raises(ValueError, match="symbol_top_k"):
            RollingAdmissionConfig(symbol_top_k=0)
        with pytest.raises(ValueError, match="symbol_universe must not be empty"):
            RollingAdmissionConfig(symbol_universe=())
        with pytest.raises(ValueError, match="duplicates"):
            RollingAdmissionConfig(symbol_universe=("BTCUSDT", "BTCUSDT"))


def _write_1h_volume_parquet(root, symbol: str, n: int, quote_vol: float, start: str) -> None:
    import pandas as pd

    from pathlib import Path

    index = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    ms = (index - epoch) // pd.Timedelta(milliseconds=1)
    df = pd.DataFrame(
        {
            "timestamp": ms,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1.0,
            "quote_vol": quote_vol,
        }
    )
    path = Path(root) / "ohlcv" / "1h" / f"{symbol}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


class TestSelectSymbolsForWindow:
    def test_top_k_by_trailing_volume(self, tmp_path) -> None:
        # top_k_by_trailing_volume: the top-5 highest trailing-90d quote-volume
        # symbols are selected in deterministic order using no bar at/after as_of.
        universe = (
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
            "ADAUSDT", "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT",
        )
        for rank, symbol in enumerate(universe):
            volume = 1_000_000.0 * (len(universe) - rank)
            _write_1h_volume_parquet(
                tmp_path, symbol, n=24 * 700, quote_vol=volume, start="2023-01-01",
            )
        as_of = pd.Timestamp("2024-07-01", tz="UTC")
        selected = select_symbols_for_window(as_of, universe, 5, tmp_path)
        assert selected == ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT")

    def test_insufficient_history_excluded(self, tmp_path) -> None:
        # insufficient_history_excluded: a symbol with <90 days of history before
        # as_of is excluded even when its partial-window volume would top the rank.
        universe = ("LOWVOL", "HOTNEW")
        _write_1h_volume_parquet(
            tmp_path, "LOWVOL", n=24 * 700, quote_vol=10.0, start="2023-01-01",
        )
        _write_1h_volume_parquet(
            tmp_path, "HOTNEW", n=24 * 10, quote_vol=1e12, start="2024-06-01",
        )
        as_of = pd.Timestamp("2024-07-01", tz="UTC")
        selected = select_symbols_for_window(as_of, universe, 1, tmp_path)
        assert selected == ("LOWVOL",)

    def test_missing_parquet_and_duplicate_universe_fail_closed(self, tmp_path) -> None:
        # A symbol with no data parquet is treated as insufficient history and
        # excluded; an empty result is returned, never fabricated.
        assert select_symbols_for_window(
            pd.Timestamp("2024-07-01", tz="UTC"), ("NODATAUSDT",), 1, tmp_path,
        ) == ()
        with pytest.raises(ValueError, match="duplicates"):
            select_symbols_for_window(
                pd.Timestamp("2024-07-01", tz="UTC"), ("A", "A"), 1, tmp_path,
            )
        with pytest.raises(ValueError, match="top_k"):
            select_symbols_for_window(
                pd.Timestamp("2024-07-01", tz="UTC"), ("A",), 0, tmp_path,
            )
        with pytest.raises(ValueError, match="universe must not be empty"):
            select_symbols_for_window(pd.Timestamp("2024-07-01", tz="UTC"), (), 1, tmp_path)
        with pytest.raises(ValueError, match="lookback_days"):
            select_symbols_for_window(
                pd.Timestamp("2024-07-01", tz="UTC"), ("A",), 1, tmp_path,
                lookback_days=0,
            )

    def test_stale_symbol_ending_before_window_is_excluded(self, tmp_path) -> None:
        # A symbol whose history ends before the trailing window is treated as
        # insufficiently covered (prior.index[-1] < window_start) and excluded
        # even though its partial volume would otherwise rank top.
        _write_1h_volume_parquet(
            tmp_path, "OLDUSDT", n=24 * 60, quote_vol=1e9, start="2023-01-01",
        )
        _write_1h_volume_parquet(
            tmp_path, "FRESHUSDT", n=24 * 700, quote_vol=10.0, start="2023-01-01",
        )
        selected = select_symbols_for_window(
            pd.Timestamp("2024-07-01", tz="UTC"), ("OLDUSDT", "FRESHUSDT"), 1, tmp_path,
        )
        assert selected == ("FRESHUSDT",)

    def test_gappy_parquet_is_excluded_not_crashing(self, tmp_path) -> None:
        # A parquet whose source has gaps fails the loader's fail-closed
        # integrity gate and is excluded (insufficient history) rather than
        # back-filled, zero-substituted, or crashing the selection.
        good_dir = tmp_path / "ohlcv" / "1h"
        good_dir.mkdir(parents=True, exist_ok=True)
        epoch = pd.Timestamp("1970-01-01", tz="UTC")

        index = pd.date_range("2023-01-01", periods=24 * 700, freq="1h", tz="UTC")
        ms = (index - epoch) // pd.Timedelta(milliseconds=1)
        pd.DataFrame(
            {
                "timestamp": ms, "open": 100.0, "high": 101.0, "low": 99.0,
                "close": 100.5, "volume": 1.0, "quote_vol": 10.0,
            }
        ).to_parquet(good_dir / "GOODUSDT.parquet")

        gappy = pd.DatetimeIndex(
            [pd.Timestamp("2023-01-01", tz="UTC") + pd.Timedelta(hours=h) for h in range(900)]
            + [pd.Timestamp("2023-01-01", tz="UTC") + pd.Timedelta(hours=1001)]
        )
        ms_gappy = (gappy - epoch) // pd.Timedelta(milliseconds=1)
        pd.DataFrame(
            {
                "timestamp": ms_gappy, "open": 100.0, "high": 101.0, "low": 99.0,
                "close": 100.5, "volume": 1.0, "quote_vol": 1e9,
            }
        ).to_parquet(good_dir / "GAPPYUSDT.parquet")

        selected = select_symbols_for_window(
            pd.Timestamp("2024-07-01", tz="UTC"), ("GAPPYUSDT", "GOODUSDT"), 1, tmp_path,
        )
        assert selected == ("GOODUSDT",)

    def test_dynamic_off_schedule_carries_fixed_symbols(self) -> None:
        # flag_off_reproduces_fixed_baseline: with dynamic_symbol_selection off
        # (the default) every window freezes the fixed profile symbols and the
        # schedule is byte-identical to the pre-dynamic baseline.
        config = RollingAdmissionConfig()
        assert config.dynamic_symbol_selection is False
        windows = build_rolling_rebalance_schedule(
            pd.Timestamp("2022-04-01", tz="UTC"),
            pd.Timestamp("2026-07-07 20:00", tz="UTC"),
            config,
        )
        assert windows
        for window in windows:
            assert window.symbols == config.symbols
            assert window.observed_end == window.rebalance_start - pd.Timedelta(hours=4)
        baseline = build_rolling_rebalance_schedule(
            pd.Timestamp("2022-04-01", tz="UTC"),
            pd.Timestamp("2024-10-01", tz="UTC"),
            config,
        )
        assert all(
            w.rebalance_start == b.rebalance_start
            for w, b in zip(windows, baseline, strict=False)
        )

    def test_dynamic_symbol_selection_freezes_top_k_per_window(self, tmp_path) -> None:
        # dynamic_symbol_selection=True ranks the universe by trailing volume
        # at each rebalance boundary and freezes the top-k on the window.
        universe = (
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
            "ADAUSDT", "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT",
        )
        for rank, symbol in enumerate(universe):
            _write_1h_volume_parquet(
                tmp_path, symbol, n=24 * 1400,
                quote_vol=1_000_000.0 * (len(universe) - rank), start="2023-01-01",
            )
        config = RollingAdmissionConfig(
            dynamic_symbol_selection=True,
            symbol_universe=universe,
            symbol_top_k=5,
        )
        windows = build_rolling_rebalance_schedule(
            pd.Timestamp("2022-04-01", tz="UTC"),
            pd.Timestamp("2026-07-07 20:00", tz="UTC"),
            config,
            data_root=tmp_path,
        )
        assert windows
        for window in windows:
            assert window.symbols == ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT")
