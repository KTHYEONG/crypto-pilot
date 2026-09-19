"""Invariant guards for continuous dependence-aware inference."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.backtest.certification import ValidationInferenceSpec
from src.mhs.deploy_gate import evaluate_continuous_growth_survival
from src.mhs.params import GrowthRiskEnvelope
from src.mhs.resources import MhsMemoryBudget


def _envelope() -> GrowthRiskEnvelope:
    return GrowthRiskEnvelope(
        name="test",
        max_drawdown=0.6,
        max_drawdown_prob=0.5,
        ruin_fraction=0.6,
        max_ruin_prob=0.5,
        horizon_years=0.25,
        leverage_ceiling=1.0,
    )


def _returns(days: int = 90, daily: float = 0.002, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.002, size=days)
    values = daily + noise
    index = pd.date_range("2024-01-01", periods=days, freq="D", tz="UTC")
    return pd.Series(values, index=index, dtype="float64")


def test_continuous_inference_ignores_quarter_partitioning() -> None:
    base = _returns()
    stress = _returns(seed=11) * 0.99
    first = evaluate_continuous_growth_survival(
        base,
        stress,
        envelope=_envelope(),
        alpha=0.05,
        n_paths=400,
        seed=7,
        batch_paths=200,
        minimum_block_days=1,
        memory_budget=MhsMemoryBudget(),
    )
    second = evaluate_continuous_growth_survival(
        base,
        stress,
        envelope=_envelope(),
        alpha=0.05,
        n_paths=400,
        seed=7,
        batch_paths=200,
        minimum_block_days=1,
        memory_budget=MhsMemoryBudget(),
    )
    assert first.go == second.go
    assert first.metrics["base_ann_log_growth_lcb"] == pytest.approx(second.metrics["base_ann_log_growth_lcb"])
    assert first.metrics["p_mdd_breach_ucb"] == pytest.approx(second.metrics["p_mdd_breach_ucb"])


def test_paired_bootstrap_preserves_same_day_shocks() -> None:
    base = _returns()
    stress = base.copy()
    verdict = evaluate_continuous_growth_survival(
        base,
        stress,
        envelope=_envelope(),
        alpha=0.05,
        n_paths=400,
        seed=7,
        batch_paths=200,
        minimum_block_days=1,
        memory_budget=MhsMemoryBudget(),
    )
    assert verdict.metrics["base_ann_log_growth_lcb"] == pytest.approx(verdict.metrics["stress_ann_log_growth_lcb"])
    assert verdict.metrics["base_ann_log_growth"] == pytest.approx(verdict.metrics["stress_ann_log_growth"])


def test_family_wise_spending_uses_union_bound() -> None:
    spec = ValidationInferenceSpec(
        family_alpha=0.2,
        procedure_budget=3,
        look_budget=2,
        endpoint_budget=4,
        bootstrap_paths=400,
        seed=7,
        resample_batch_paths=200,
        minimum_block_days=5,
    )
    expected = 0.2 / (3 * 2 * 4)
    assert spec.local_alpha == pytest.approx(expected)
    sidak = 1.0 - (1.0 - 0.2) ** (1.0 / 24.0)
    assert spec.local_alpha != pytest.approx(sidak)
    assert spec.required_paths == math.ceil(20.0 / expected)


def test_unresolved_tail_never_passes_as_zero_risk() -> None:
    base = _returns()
    with pytest.raises(DataIntegrityError, match="TAIL"):
        evaluate_continuous_growth_survival(
            base,
            base,
            envelope=_envelope(),
            alpha=0.05,
            n_paths=10,
            seed=7,
            batch_paths=10,
            minimum_block_days=1,
            memory_budget=MhsMemoryBudget(),
        )


def test_batched_draws_reproduce_identical_verdict() -> None:
    base = _returns()
    stress = _returns(seed=11) * 0.99
    small_batch = evaluate_continuous_growth_survival(
        base,
        stress,
        envelope=_envelope(),
        alpha=0.05,
        n_paths=400,
        seed=7,
        batch_paths=37,
        minimum_block_days=1,
        memory_budget=MhsMemoryBudget(),
    )
    large_batch = evaluate_continuous_growth_survival(
        base,
        stress,
        envelope=_envelope(),
        alpha=0.05,
        n_paths=400,
        seed=7,
        batch_paths=200,
        minimum_block_days=1,
        memory_budget=MhsMemoryBudget(),
    )
    assert small_batch.go == large_batch.go
    assert small_batch.metrics["base_ann_log_growth_lcb"] == pytest.approx(large_batch.metrics["base_ann_log_growth_lcb"])
    assert small_batch.metrics["p_ruin_ucb"] == pytest.approx(large_batch.metrics["p_ruin_ucb"])


def test_continuous_inference_rejects_invalid_inputs() -> None:
    base = _returns()
    budget = MhsMemoryBudget()
    with pytest.raises(DataIntegrityError):
        evaluate_continuous_growth_survival(
            base, base, envelope=_envelope(), alpha=0.0, n_paths=400, seed=7,
            batch_paths=200, minimum_block_days=1, memory_budget=budget,
        )
    with pytest.raises(DataIntegrityError):
        evaluate_continuous_growth_survival(
            base, base, envelope=_envelope(), alpha=0.05, n_paths=0, seed=7,
            batch_paths=200, minimum_block_days=1, memory_budget=budget,
        )
    with pytest.raises(DataIntegrityError):
        evaluate_continuous_growth_survival(
            base, base, envelope=_envelope(), alpha=0.05, n_paths=400, seed=7,
            batch_paths=0, minimum_block_days=1, memory_budget=budget,
        )
    with pytest.raises(DataIntegrityError):
        evaluate_continuous_growth_survival(
            base, base, envelope=_envelope(), alpha=0.05, n_paths=400, seed=7,
            batch_paths=200, minimum_block_days=0, memory_budget=budget,
        )
    with pytest.raises(DataIntegrityError):
        evaluate_continuous_growth_survival(
            base, base, envelope=_envelope(), alpha=0.05, n_paths=400, seed=7,
            batch_paths=200, minimum_block_days=1, memory_budget="bad",  # type: ignore[arg-type]
        )
    naive = pd.Series(base.to_numpy(), index=pd.DatetimeIndex(base.index.tz_convert(None)), dtype="float64")
    with pytest.raises(DataIntegrityError):
        evaluate_continuous_growth_survival(
            naive, naive, envelope=_envelope(), alpha=0.05, n_paths=400, seed=7,
            batch_paths=200, minimum_block_days=1, memory_budget=budget,
        )
    non_utc = pd.Series(base.to_numpy(), index=base.index.tz_convert("America/New_York"), dtype="float64")
    with pytest.raises(DataIntegrityError):
        evaluate_continuous_growth_survival(
            non_utc, non_utc, envelope=_envelope(), alpha=0.05, n_paths=400, seed=7,
            batch_paths=200, minimum_block_days=1, memory_budget=budget,
        )
    dup = pd.concat([base, base.iloc[:1]])
    with pytest.raises(DataIntegrityError):
        evaluate_continuous_growth_survival(
            dup, dup, envelope=_envelope(), alpha=0.05, n_paths=400, seed=7,
            batch_paths=200, minimum_block_days=1, memory_budget=budget,
        )
    bad = base.copy()
    bad.iloc[0] = float("nan")
    with pytest.raises(DataIntegrityError):
        evaluate_continuous_growth_survival(
            bad, bad, envelope=_envelope(), alpha=0.05, n_paths=400, seed=7,
            batch_paths=200, minimum_block_days=1, memory_budget=budget,
        )
    ruin = base.copy()
    ruin.iloc[0] = -1.5
    with pytest.raises(DataIntegrityError):
        evaluate_continuous_growth_survival(
            ruin, ruin, envelope=_envelope(), alpha=0.05, n_paths=400, seed=7,
            batch_paths=200, minimum_block_days=1, memory_budget=budget,
        )
    shifted = pd.Series(base.to_numpy(), index=base.index + pd.Timedelta(days=1), dtype="float64")
    with pytest.raises(DataIntegrityError):
        evaluate_continuous_growth_survival(
            base, shifted, envelope=_envelope(), alpha=0.05, n_paths=400, seed=7,
            batch_paths=200, minimum_block_days=1, memory_budget=budget,
        )
    short = _returns(days=2)
    verdict = evaluate_continuous_growth_survival(
        short, short, envelope=_envelope(), alpha=0.05, n_paths=400, seed=7,
        batch_paths=200, minimum_block_days=5, memory_budget=budget,
    )
    assert verdict.go is False
    assert verdict.reason_codes == ("INFERENCE_INSUFFICIENT_BLOCKS",)
    from src.mhs.deploy_gate import _binomial_upper_bound, _require_continuous_returns

    with pytest.raises(DataIntegrityError):
        _binomial_upper_bound(0, 0, 0.05)
    assert _binomial_upper_bound(0, 400, 0.05) > 0.0
    assert _binomial_upper_bound(400, 400, 0.05) == 1.0
    with pytest.raises(DataIntegrityError):
        _require_continuous_returns(pd.Series([0.01]), "x")
    nat_index = pd.DatetimeIndex([pd.NaT], tz="UTC")
    with pytest.raises(DataIntegrityError):
        _require_continuous_returns(pd.Series([0.01], index=nat_index), "x")
    with pytest.raises(DataIntegrityError):
        evaluate_continuous_growth_survival(
            base, base, envelope=_envelope(), alpha=0.05, n_paths=400, seed=True,  # type: ignore[arg-type]
            batch_paths=200, minimum_block_days=1, memory_budget=budget,
        )
    single = _returns(days=1)
    with pytest.raises(DataIntegrityError):
        evaluate_continuous_growth_survival(
            single, single, envelope=_envelope(), alpha=0.05, n_paths=400, seed=7,
            batch_paths=200, minimum_block_days=1, memory_budget=budget,
        )
    assert _binomial_upper_bound(5, 400, 0.05) > 0.0
    falling = _returns(days=90, daily=-0.01)
    tight = GrowthRiskEnvelope(
        name="tight", max_drawdown=0.01, max_drawdown_prob=0.01,
        ruin_fraction=0.99, max_ruin_prob=0.01, horizon_years=0.25, leverage_ceiling=1.0,
    )
    failed = evaluate_continuous_growth_survival(
        falling, falling, envelope=tight, alpha=0.05, n_paths=400, seed=7,
        batch_paths=200, minimum_block_days=1, memory_budget=budget,
    )
    assert failed.go is False
    assert "CONTINUOUS_BASE_GROWTH" in failed.reason_codes
    assert "CONTINUOUS_STRESS_GROWTH" in failed.reason_codes
    assert "CONTINUOUS_MDD_BUDGET" in failed.reason_codes
    assert "CONTINUOUS_RUIN_BUDGET" in failed.reason_codes


def test_certification_contracts_reject_invalid_shapes() -> None:
    from src.mhs.backtest.certification import (
        DailyPortfolioEvidence,
        EvaluationContext,
        EvidenceCheck,
        ValidationInferenceSpec,
        assess_process_validation,
        inventory_daily_evidence,
    )

    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-02-01", tz="UTC")
    with pytest.raises(DataIntegrityError):
        EvidenceCheck(
            requirement="nope", status="passed", procedure_digest="p", input_manifest_digest=None,
            code_digest="c", interval_start=start, interval_end=end, artifact_digest="a", reason_codes=(),
        )
    with pytest.raises(DataIntegrityError):
        EvidenceCheck(
            requirement="input_seal", status="bogus",  # type: ignore[arg-type]
            procedure_digest="p", input_manifest_digest=None, code_digest="c",
            interval_start=start, interval_end=end, artifact_digest="a", reason_codes=(),
        )
    with pytest.raises(DataIntegrityError):
        EvidenceCheck(
            requirement="input_seal", status="passed", procedure_digest="p", input_manifest_digest=None,
            code_digest="c", interval_start=end, interval_end=start, artifact_digest="a", reason_codes=(),
        )
    naive_start = pd.Timestamp("2024-01-01")
    with pytest.raises(DataIntegrityError):
        EvidenceCheck(
            requirement="input_seal", status="passed", procedure_digest="p", input_manifest_digest=None,
            code_digest="c", interval_start=naive_start, interval_end=end, artifact_digest="a", reason_codes=(),
        )
    with pytest.raises(DataIntegrityError):
        ValidationInferenceSpec(
            family_alpha=0.0, procedure_budget=1, look_budget=1, endpoint_budget=4,
            bootstrap_paths=400, seed=7, resample_batch_paths=200, minimum_block_days=1,
        )
    with pytest.raises(DataIntegrityError):
        ValidationInferenceSpec(
            family_alpha=0.2, procedure_budget=0, look_budget=1, endpoint_budget=4,
            bootstrap_paths=400, seed=7, resample_batch_paths=200, minimum_block_days=1,
        )
    with pytest.raises(DataIntegrityError):
        ValidationInferenceSpec(
            family_alpha=0.2, procedure_budget=1, look_budget=1, endpoint_budget=3,
            bootstrap_paths=400, seed=7, resample_batch_paths=200, minimum_block_days=1,
        )
    with pytest.raises(DataIntegrityError):
        EvaluationContext(
            role="bogus",  # type: ignore[arg-type]
            procedure_digest="p", code_digest="c", input_manifest_digest=None,
            interval_start=start, interval_end=end, registered_at=None, consulted_through=None,
            family_id=None, look_ordinal=None, inference_spec=None,
            journal_complete=True, observed_through=end,
        )
    with pytest.raises(DataIntegrityError):
        EvaluationContext(
            role="forward", procedure_digest="p", code_digest="c", input_manifest_digest=None,
            interval_start=end, interval_end=start, registered_at=None, consulted_through=None,
            family_id=None, look_ordinal=None, inference_spec=None,
            journal_complete=True, observed_through=end,
        )
    with pytest.raises(DataIntegrityError):
        EvaluationContext(
            role="forward", procedure_digest="p", code_digest="c", input_manifest_digest=None,
            interval_start=start, interval_end=end, registered_at=None, consulted_through=None,
            family_id=None, look_ordinal=0, inference_spec=None,
            journal_complete=True, observed_through=end,
        )
    with pytest.raises(DataIntegrityError):
        ValidationInferenceSpec(
            family_alpha=0.2, procedure_budget=1, look_budget=1, endpoint_budget=4,
            bootstrap_paths=400, seed=True,  # type: ignore[arg-type]
            resample_batch_paths=200, minimum_block_days=1,
        )
    with pytest.raises(DataIntegrityError):
        EvaluationContext(
            role="forward", procedure_digest="p", code_digest="c", input_manifest_digest=None,
            interval_start=naive_start, interval_end=end, registered_at=None, consulted_through=None,
            family_id=None, look_ordinal=None, inference_spec=None,
            journal_complete=True, observed_through=end,
        )
    with pytest.raises(DataIntegrityError):
        EvaluationContext(
            role="forward", procedure_digest="p", code_digest="c", input_manifest_digest=None,
            interval_start=start, interval_end=end, registered_at=None, consulted_through=None,
            family_id=None, look_ordinal=None, inference_spec=None,
            journal_complete=True, observed_through=naive_start,
        )
    with pytest.raises(DataIntegrityError):
        EvaluationContext(
            role="forward", procedure_digest="p", code_digest="c", input_manifest_digest=None,
            interval_start=start, interval_end=end, registered_at=None, consulted_through=None,
            family_id=None, look_ordinal=None, inference_spec=None,
            journal_complete="yes",  # type: ignore[arg-type]
            observed_through=end,
        )
    plain = pd.Series([0.01, 0.02])
    with pytest.raises(DataIntegrityError):
        inventory_daily_evidence(_replay_like(plain))
    nat_eq = pd.Series(
        [1.0, 1.01],
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC"), pd.NaT], tz="UTC"),
    )
    with pytest.raises(DataIntegrityError):
        inventory_daily_evidence(_replay_like(nat_eq))
    naive_eq = pd.Series(
        [1.0, 1.01], index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"]),
    )
    with pytest.raises(DataIntegrityError):
        inventory_daily_evidence(_replay_like(naive_eq))
    dup_eq = pd.Series(
        [1.0, 1.01],
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-01"], tz="UTC"),
    )
    with pytest.raises(DataIntegrityError):
        inventory_daily_evidence(_replay_like(dup_eq))
    nan_eq = pd.Series(
        [1.0, float("nan")],
        index=pd.date_range("2024-01-01", periods=2, freq="30min", tz="UTC"),
    )
    with pytest.raises(DataIntegrityError):
        inventory_daily_evidence(_replay_like(nan_eq))
    import dataclasses

    no_avail = dataclasses.replace(
        _replay_like(
            pd.Series([1.0, 1.01], index=pd.date_range("2024-01-01", periods=2, freq="30min", tz="UTC"))
        ),
        ledger_available_at=None,
    )
    with pytest.raises(DataIntegrityError):
        inventory_daily_evidence(no_avail)
    with pytest.raises(DataIntegrityError):
        inventory_daily_evidence(object())  # type: ignore[arg-type]
    two_day = pd.Series([1.0, 1.01], index=pd.date_range("2024-01-01", periods=2, freq="30min", tz="UTC"))
    bad_avail = dataclasses.replace(_replay_like(two_day), ledger_available_at=[start, end])  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError):
        inventory_daily_evidence(bad_avail)
    short_avail = dataclasses.replace(
        _replay_like(two_day),
        ledger_available_at=pd.DatetimeIndex([two_day.index[0]], tz="UTC"),
    )
    with pytest.raises(DataIntegrityError):
        inventory_daily_evidence(short_avail)
    nat_avail = dataclasses.replace(
        _replay_like(two_day),
        ledger_available_at=pd.DatetimeIndex([two_day.index[0], pd.NaT], tz="UTC"),
    )
    with pytest.raises(DataIntegrityError):
        inventory_daily_evidence(nat_avail)
    naive_avail = dataclasses.replace(
        _replay_like(two_day),
        ledger_available_at=pd.DatetimeIndex([two_day.index[0].tz_convert(None), two_day.index[1].tz_convert(None)]),
    )
    with pytest.raises(DataIntegrityError):
        inventory_daily_evidence(naive_avail)
    early_avail = dataclasses.replace(
        _replay_like(two_day),
        ledger_available_at=pd.DatetimeIndex(
            [two_day.index[0] - pd.Timedelta(hours=1), two_day.index[1] + pd.Timedelta(minutes=30)], tz="UTC"
        ),
    )
    with pytest.raises(DataIntegrityError):
        inventory_daily_evidence(early_avail)
    one_day = pd.Series([1.0, 1.01], index=pd.date_range("2024-01-01", periods=2, freq="30min", tz="UTC"))
    with pytest.raises(DataIntegrityError):
        inventory_daily_evidence(_replay_like(one_day.iloc[:1]))
    zero_eq = pd.Series(
        [0.0, 0.0, 1.0, 1.01],
        index=pd.date_range("2024-01-01", periods=4, freq="12h", tz="UTC"),
    )
    with pytest.raises(DataIntegrityError):
        inventory_daily_evidence(_replay_like(zero_eq))
    from src.mhs.backtest.certification import _select_formal_returns

    grid = pd.date_range("2024-01-01", periods=96, freq="30min", tz="UTC")
    eq = pd.Series(1.0 + 0.0001 * np.arange(len(grid)), index=grid, dtype="float64")
    good = _replay_like(eq)
    import dataclasses as _dc

    good = _dc.replace(good, ledger_available_at=pd.DatetimeIndex(grid + pd.Timedelta(minutes=30), tz="UTC"))
    good_ev = inventory_daily_evidence(good)
    far_ctx = EvaluationContext(
        role="historical", procedure_digest="p", code_digest="c", input_manifest_digest=None,
        interval_start=pd.Timestamp("2030-01-01", tz="UTC"), interval_end=pd.Timestamp("2030-02-01", tz="UTC"),
        registered_at=None, consulted_through=None, family_id=None, look_ordinal=None,
        inference_spec=None, journal_complete=True,
        observed_through=pd.Timestamp("2030-02-02", tz="UTC"),
    )
    with pytest.raises(DataIntegrityError):
        _select_formal_returns(good_ev, good_ev, far_ctx)
    grid_b = pd.date_range("2024-01-02", periods=96, freq="30min", tz="UTC")
    eq_b = pd.Series(1.0 + 0.0001 * np.arange(len(grid_b)), index=grid_b, dtype="float64")
    other = _dc.replace(
        _replay_like(eq_b),
        ledger_available_at=pd.DatetimeIndex(grid_b + pd.Timedelta(minutes=30), tz="UTC"),
    )
    other_ev = inventory_daily_evidence(other)
    near_ctx = EvaluationContext(
        role="historical", procedure_digest="p", code_digest="c", input_manifest_digest=None,
        interval_start=pd.Timestamp("2024-01-01", tz="UTC"), interval_end=pd.Timestamp("2024-01-05", tz="UTC"),
        registered_at=None, consulted_through=None, family_id=None, look_ordinal=None,
        inference_spec=None, journal_complete=True,
        observed_through=pd.Timestamp("2024-01-06", tz="UTC"),
    )
    with pytest.raises(DataIntegrityError):
        _select_formal_returns(good_ev, other_ev, near_ctx)
    with pytest.raises(DataIntegrityError):
        DailyPortfolioEvidence(
            returns=pd.Series([0.01, 0.02]),
            label_start=pd.DatetimeIndex([start], tz="UTC"),
            label_end=pd.DatetimeIndex([start, end], tz="UTC"),
            available_at=pd.DatetimeIndex([start, end], tz="UTC"),
        )
    empty = pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"))
    with pytest.raises(DataIntegrityError):
        inventory_daily_evidence(_replay_like(empty))
    unknown = EvidenceCheck(
        requirement="input_seal", status="passed", procedure_digest="p", input_manifest_digest=None,
        code_digest="c", interval_start=start, interval_end=end, artifact_digest="a", reason_codes=(),
    )
    ctx = EvaluationContext(
        role="historical", procedure_digest="p", code_digest="c", input_manifest_digest=None,
        interval_start=start, interval_end=end, registered_at=None, consulted_through=None,
        family_id=None, look_ordinal=None, inference_spec=None,
        journal_complete=True, observed_through=end,
    )
    with pytest.raises(DataIntegrityError):
        assess_process_validation(
            _replay_like(empty), _replay_like(empty), context=ctx, checks=(unknown, unknown),
            envelope=_envelope(), memory_budget=MhsMemoryBudget(),
        )
    with pytest.raises(DataIntegrityError):
        assess_process_validation(
            _replay_like(empty), _replay_like(empty), context=ctx, checks=(),
            envelope=_envelope(), memory_budget="bad",  # type: ignore[arg-type]
        )


def _replay_like(equity: pd.Series):  # type: ignore[no-untyped-def]
    from src.mhs.execution.contracts import SimulatedInventoryLedgerResult, StrategyExecutionReplayResult

    zeros = pd.Series(dtype="float64", index=equity.index)
    ledger = SimulatedInventoryLedgerResult(
        equity=equity, net_returns=zeros, simulated_units=None, mark_to_market_pnl=zeros,
        funding_charge=zeros, fee_charge=zeros, fill_turnover=zeros,
        fill_source="OHLCV_IMMEDIATE_TAKER", mark_source="OHLCV_CLOSE_FALLBACK",
        primary_valid=True, invalid_reasons=(), data_gaps=(),
    )
    avail = None
    if len(equity) and isinstance(equity.index, pd.DatetimeIndex) and equity.index.tz is not None:
        avail = pd.DatetimeIndex(equity.index + pd.Timedelta(minutes=30), tz="UTC")
    return StrategyExecutionReplayResult(
        simulated_fills=pd.DataFrame(), ledger=ledger, simulated_units=pd.DataFrame(),
        simulated_notional_weights=pd.DataFrame(), fill_source="OHLCV_IMMEDIATE_TAKER",
        mark_source="OHLCV_CLOSE_FALLBACK", submit_times=pd.Series(dtype="datetime64[ns, UTC]"),
        fill_times=pd.Series(dtype="datetime64[ns, UTC]"), fill_count=0, unfilled_count=0,
        fallback_count=0, all_intent_shortfall_bps=0.0, forced_exit_count=0,
        forced_exit_notional=0.0, termination_counts={}, unsupported_assumptions=(),
        elapsed_seconds=0.0, ledger_available_at=avail,
    )
