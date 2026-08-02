from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.application.research.expert import admission as app
from src.application.research.expert import rolling_admission as ra
from src.application.research.expert.admission import run_technical_library_admission
from src.research.expert_portfolio.admission import (
    priority_shortlist_family_unique_proposals,
)
from src.research.expert_portfolio.admission_types import (
    LibraryAdmissionConfig,
    TechnicalLibraryAdmissionRequest,
    resolve_rolling_library_admission_profile,
)
from src.research.expert_portfolio.models import ContextualRouterSpec, ExpertDefinition
from src.research.expert_portfolio.rolling import (
    build_rolling_rebalance_schedule,
    rolling_admission_config_for_profile,
)


def _synthetic_universe(
    n_families: int, per_family: int, seed: int,
) -> tuple[tuple[ExpertDefinition, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Random family/symbol universe with the rolling-v3 pair-screen gates."""
    rng = np.random.default_rng(seed)
    n = n_families * per_family
    experts = tuple(
        ExpertDefinition(
            f"e{i}", f"s{i}", f"f{i // per_family}", (f"SYM{i}",), "run", "h",
        )
        for i in range(n)
    )
    correlation = np.zeros((n, n))
    joint_negative = np.zeros((n, n))
    compatibility = np.ones((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            c = rng.uniform(-0.6, 0.6)
            v = rng.uniform(0.0, 0.3)
            correlation[i, j] = correlation[j, i] = c
            joint_negative[i, j] = joint_negative[j, i] = v
            compatibility[i, j] = compatibility[j, i] = abs(c) <= 0.5 and v <= 0.15
    np.fill_diagonal(compatibility, False)
    return experts, correlation, joint_negative, compatibility


def _synthetic_evidence(profile, window, definitions) -> ra.WindowScenarioEvidence:
    rows = 520
    index = pd.date_range("2022-06-20", periods=rows, freq="4h", tz="UTC")
    rng = np.random.default_rng(3)
    panel = pd.DataFrame(np.nan, index=index, columns=[d.expert_id for d in definitions])
    panel.iloc[1:] = np.clip(
        rng.normal(0.002, 0.003, (rows - 1, len(definitions))), 0.0, None,
    )
    context = pd.Series("up_low_vol", index=index)
    trades = pd.DataFrame()
    return ra.WindowScenarioEvidence(
        profile=profile,
        window=window,
        code_hash="h" * 64,
        definitions=definitions,
        base_panel=panel,
        base_trade_counts={d.expert_id: 25 for d in definitions},
        base_component_trades=trades,
        base_context=context,
        stress_panel=panel.copy(),
        stress_component_trades=trades,
        base_candidate_runner_calls=0,
        stress_candidate_runner_calls=0,
        base_wall_seconds=0.0,
        stress_wall_seconds=0.0,
        base_workers=0,
        stress_workers=0,
    )


def _training_report(proposal_id: str) -> ra.LibraryAdmissionBacktestReport:
    from src.research.evaluation.metrics import Metrics
    from src.research.evaluation.promotion import PromotionResult
    from src.research.evaluation.reliability import (
        FoldDistributionResult,
        ReliabilityGateResult,
    )
    from src.research.expert_portfolio.models import ContextualRouterSpec

    expert_ids = tuple(sorted(proposal_id.split(":")[1].split("|")))
    gate = ReliabilityGateResult(
        lcb90_cagr=0.20, lcb95_cagr=0.15, p_negative=0.0, point_cagr=0.22,
        t_stat=2.5, trade_count=40, block_size_used=1, verdict="PASS",
    )
    folds = FoldDistributionResult(
        n_folds=4, median_fold_cagr=0.01, worst_fold_cagr=0.0,
        median_fold_calmar=0.5, max_period_contribution=0.2, gate_pass=True,
    )
    metrics = Metrics(
        cagr=0.02, mdd=-0.01, sharpe=0.5, sortino=0.6, calmar=2.0,
        profit_factor=1.5, expectancy=0.001, win_rate=0.5, payoff_ratio=1.1,
        trade_count=40, exposure=0.5, turnover=0.1,
        trades_per_year={"2024": 20, "2025": 20},
    )
    return ra.LibraryAdmissionBacktestReport(
        status="COMPLETE",
        proposal_id=proposal_id,
        expert_ids=expert_ids,
        router=ContextualRouterSpec("BTCUSDT", 48, 48, 96),
        window_start="2022-07-01 00:00:00+00:00",
        window_end="2024-06-30 20:00:00+00:00",
        observation_metrics=metrics,
        observation_gate=gate,
        observation_folds=folds,
        stress_metrics=metrics,
        stress_gate=gate,
        stress_folds=folds,
        promotion=PromotionResult(
            status="OBSERVATION_PASS", observation_verdict="PASS",
            fold_gate_pass=True, stress_verdict="PASS", holdout_verdict=None,
        ),
        allocation_cost_total=0.0,
        stress_allocation_cost_total=0.0,
        execution_workers=1,
        # Deliberately not near-zero: an unrealistically tiny measured
        # wall-time (e.g. 0.001s) extrapolates an effective budget so large
        # (~120000) that the best-first search legitimately exhausts its real
        # node budget on this dense synthetic 90-candidate panel and falls
        # back to the probe-only shortlist. 0.1s keeps the dynamic-budget
        # expansion meaningfully exercised (COMPLETE, hundreds of proposals)
        # while staying fast.
        wall_seconds=0.1,
    )


def test_expert_portfolio_rolling_as_of_2026_07_07_reports_nonzero_observation_pass_rate_or_explains_why_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # RAP-V3: at the 2026-07-07 snapshot the canonical profile dispatches to the
    # exact best-first search over the full 90-candidate universe and produces
    # backtests. The observation gate pass rate is reported (via synthetic
    # PASS-gated training evidence) rather than being structurally zero.
    profile = resolve_rolling_library_admission_profile("technical-5symbol-rolling")
    config = rolling_admission_config_for_profile(
        "technical-5symbol-rolling", profile.symbols,
    )
    window = build_rolling_rebalance_schedule(
        pd.Timestamp("2022-04-01", tz="UTC"),
        pd.Timestamp("2026-07-07 20:00", tz="UTC"),
        config,
    )[0]
    definitions = ra._materialize_universe_definitions(profile, "h" * 64)
    assert len(definitions) == 90
    evidence = _synthetic_evidence(profile, window, definitions)
    monkeypatch.setattr(ra, "build_window_scenario_evidence", lambda *a, **k: evidence)
    monkeypatch.setattr(
        ra, "_run_proposal_from_evidence",
        lambda proposal, evidence, profile, window, config: _training_report(
            proposal.proposal_id,
        ),
    )
    selection, shortlist, reports = ra._select_for_window(profile, window, config, None)
    assert selection.generation_status == "COMPLETE"
    assert shortlist
    assert len(reports) == len(shortlist)
    assert {r.proposal_id for r in reports} == {p.proposal_id for p in shortlist}
    observation_pass_rate = sum(
        1 for r in reports if r.observation_gate.verdict == "PASS"
    ) / max(len(reports), 1)
    assert observation_pass_rate > 0.0


def test_expert_portfolio_rolling_v3_growth_simulation_stays_within_perf_budget_at_300_candidates() -> None:
    # RAP-V3: a future 30-family x 10-member 300-candidate universe (3x the real
    # profile) stays inside the node budget and completes within the perf guard,
    # so the operator's "keep adding strategies" plan stays computationally
    # feasible for the best-first search.
    experts, correlation, joint_negative, compatibility = _synthetic_universe(
        30, 10, 11,
    )
    assert len(experts) == 300
    config = LibraryAdmissionConfig(2, 5, 1, 1, 0.5, 0.15, 1, 1_000_000)
    started = time.perf_counter()
    result = priority_shortlist_family_unique_proposals(
        experts, correlation, joint_negative, compatibility, config, 24,
    )
    elapsed = time.perf_counter() - started
    assert result.generation_status == "COMPLETE"
    assert result.proposals
    assert result.generated_nodes < config.max_combinations
    assert elapsed < 20.0


def test_flag_off_reproduces_fixed_baseline() -> None:
    # flag_off_reproduces_fixed_baseline: RollingAdmissionConfig with
    # dynamic_symbol_selection=False (the default) produces byte-identical
    # rebalance schedules/symbols to the current technical-5symbol-rolling
    # profile -- zero behavior change unless explicitly opted in.
    import dataclasses

    from src.research.expert_portfolio.rolling import RollingAdmissionConfig

    common_start = pd.Timestamp("2022-04-01", tz="UTC")
    as_of = pd.Timestamp("2026-07-07 20:00", tz="UTC")
    baseline = RollingAdmissionConfig()
    assert baseline.dynamic_symbol_selection is False

    default_schedule = build_rolling_rebalance_schedule(common_start, as_of, baseline)
    explicit_off = dataclasses.replace(baseline, dynamic_symbol_selection=False)
    explicit_schedule = build_rolling_rebalance_schedule(
        common_start, as_of, explicit_off,
    )
    assert len(default_schedule) == len(explicit_schedule)
    for left, right in zip(default_schedule, explicit_schedule, strict=True):
        assert left.to_report_dict() == right.to_report_dict()
        assert right.symbols == baseline.symbols
    assert default_schedule[0].symbols == ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")


def test_library_admission_4h_default_is_byte_identical_to_unscaled_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # TIS-04: at the default timeframe='4h' the scaling call sites in
    # admission.py are no-ops, so the report's router and admission are
    # byte-identical to the request (the pre-scaling behavior) and the
    # fingerprint stays stable.
    index = pd.date_range("2024-01-01", periods=4400, freq="4h", tz="UTC")
    t = np.arange(len(index), dtype=np.float64)
    close = 100.0 + 0.01 * t + 5.0 * np.sin(t / 20.0)
    frame = pd.DataFrame({
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": 1000.0,
    }, index=index)
    funding = pd.Series(0.0, index=index, dtype=float)
    monkeypatch.setattr(
        app, "_load_technical_market_data",
        lambda symbol, start, end, *, timeframe="4h": (frame, funding),
    )
    monkeypatch.setattr(
        app, "load_ohlcv_1h_as",
        lambda path, timeframe, *, start=None, end=None: frame,
    )
    monkeypatch.setattr(app, "compute_code_hash", lambda *args, **kwargs: "c" * 64)
    monkeypatch.setattr(
        app, "technical_data_hashes",
        lambda symbol: {"perp_ohlcv": "a" * 64, "funding": "b" * 64},
    )

    request = TechnicalLibraryAdmissionRequest(
        candidate_sources=(
            "technical_macd_histogram_regime_long_v1",
            "technical_rsi_trend_pullback_long_v1",
        ),
        symbols=("BTCUSDT", "ETHUSDT"),
        router=ContextualRouterSpec("BTCUSDT", 60, 20, 30),
        admission=LibraryAdmissionConfig(
            min_experts=1,
            max_experts=2,
            min_closed_trades=0,
            min_active_return_bars=200,
            max_abs_pairwise_log_return_correlation=0.8,
            max_joint_negative_return_rate=0.5,
            min_context_covered_states=1,
            max_combinations=100,
            max_workers=1,
        ),
        start="2024-01-01",
    )
    report = run_technical_library_admission(request)

    assert report.status == "COMPLETE"
    assert report.router == request.router
    assert report.admission == request.admission
    assert report.fingerprint()["router"] == {
        "context_symbol": "BTCUSDT",
        "trend_lookback_bars": 60,
        "volatility_lookback_bars": 20,
        "min_context_history_bars": 30,
        "confidence": 0.90,
    }
    assert report.fingerprint()["admission"]["min_active_return_bars"] == 200
