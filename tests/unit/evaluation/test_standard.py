"""Spec 38 part 6: one local evaluation standard over the book that trades.

Unit bootstrap budget: the stationary block bootstrap runs 200 paths here
(production uses 2000); seeds stay fixed so verdicts are deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.evaluation.standard as standard
from src.common.errors import DataIntegrityError
from src.evaluation.standard import EvaluationInputs, Verdict, evaluate_strategy
from src.evaluation.statistics import sharpe_sampling_variance
from src.evaluation.trials import TrialPopulation
from src.strategy.release import EvaluationCriteria

@pytest.fixture(autouse=True)
def _bootstrap_budget(monkeypatch):
    monkeypatch.setattr(standard, "REPORT_BOOTSTRAP_PATHS", 200)

N_DISCOVERY = 1826
N_HOLDOUT = 92
DISCOVERY_START = pd.Timestamp("2021-04-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-07-01", tz="UTC")
CUTOFF = pd.Timestamp("2026-07-01", tz="UTC")


def _series(values: np.ndarray, start: pd.Timestamp) -> pd.Series:
    index = pd.date_range(start=start, periods=len(values), freq="D", tz="UTC")
    return pd.Series(np.asarray(values, dtype="float64"), index=index, dtype="float64")


def _drift_returns(n: int, seed: int, drift_scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_t(5, size=n) * np.sqrt(3.0 / 5.0)
    return 0.01 * 2.0 / np.sqrt(365.0) * drift_scale + 0.01 * noise


def _population(values: np.ndarray) -> TrialPopulation:
    observed = float(values.mean() / np.std(values, ddof=1))
    skew = float(pd.Series(values).skew())
    kurtosis = float(pd.Series(values).kurt() + 3.0)
    floor = sharpe_sampling_variance(observed, len(values), skew, kurtosis)
    return TrialPopulation(
        family="flow_mom", prior_trials=96, sharpes=(), n_trials=97, sharpe_variance=floor,
    )


def _good_inputs(**overrides) -> EvaluationInputs:
    base = _drift_returns(N_DISCOVERY, seed=0)
    stress = base - 0.0002
    deployed = _drift_returns(N_DISCOVERY, seed=1000)
    holdout = _drift_returns(N_HOLDOUT, seed=11)
    base_s = _series(base, DISCOVERY_START)
    stress_s = _series(stress, DISCOVERY_START)
    neighbors = tuple(_series(_drift_returns(N_DISCOVERY, 500 + i, 0.8), DISCOVERY_START) for i in range(7))
    params = {
        "strategy_id": "flow_mom_top20",
        "spec_digest": "abc",
        "discovery": (DISCOVERY_START, DISCOVERY_START + pd.Timedelta(days=N_DISCOVERY - 1)),
        "holdout": (HOLDOUT_START, HOLDOUT_START + pd.Timedelta(days=N_HOLDOUT - 1)),
        "design_data_cutoff": CUTOFF,
        "base_returns": base_s,
        "stress_returns": stress_s,
        "funding_by_symbol": {f"S{i}USDT": 0.02 for i in range(7)},
        "funding_income_daily": pd.DataFrame({f"S{i}USDT": np.full(N_DISCOVERY, 0.02 / N_DISCOVERY) for i in range(7)}, index=base_s.index),
        "participation": pd.Series(np.zeros(N_DISCOVERY), index=base_s.index, dtype="float64"),
        "ledger_certified": True,
        "lake_coverage_ok": True,
        "causality_ok": True,
        "deployed_returns": _series(deployed, DISCOVERY_START),
        "deployed_stress_returns": _series(deployed - 0.0001, DISCOVERY_START),
        "deployed_liquidated": False,
        "deployed_stress_liquidated": False,
        "deployed_margin_breaches": 0,
        "deployed_max_leverage": 1.0,
        "neighbors": neighbors,
        "trial_population": _population(base),
        "holdout_returns": _series(holdout, HOLDOUT_START),
    }
    params.update(overrides)
    if "deployed_returns" in overrides and "deployed_stress_returns" not in overrides:
        params["deployed_stress_returns"] = overrides["deployed_returns"].copy()
    if "funding_by_symbol" in overrides:
        params["funding_income_daily"] = pd.DataFrame({symbol: np.full(len(params["stress_returns"]), income / len(params["stress_returns"])) for symbol, income in overrides["funding_by_symbol"].items()}, index=params["stress_returns"].index)
    else:
        params["funding_income_daily"] = params["funding_income_daily"].reindex(params["stress_returns"].index).fillna(0)
    return EvaluationInputs(**params)


def _code(result, code: str):
    return next(c for c in result.checks if c.code == code)


def test_good_synthetic_strategy_accepted() -> None:
    """Five Sharpe-2 years, passing neighbors, compatible book, same-process holdout: ACCEPT."""
    result = evaluate_strategy(_good_inputs(), EvaluationCriteria())
    assert result.verdict == Verdict.ACCEPT
    assert all(c.passed is True for c in result.checks)


def test_short_discovery_is_inconclusive_not_rejected() -> None:
    """400 discovery days: INCONCLUSIVE with computable checks still reported."""
    n = 400
    base = _drift_returns(n, seed=0)
    base_s = _series(base, DISCOVERY_START)
    result = evaluate_strategy(
        _good_inputs(
            base_returns=base_s,
            stress_returns=_series(base - 0.0002, DISCOVERY_START),
            participation=pd.Series(np.zeros(n), index=base_s.index, dtype="float64"),
            discovery=(DISCOVERY_START, DISCOVERY_START + pd.Timedelta(days=n - 1)),
            neighbors=tuple(_series(_drift_returns(n, 500 + i, 0.8), DISCOVERY_START) for i in range(7)),
            trial_population=_population(base),
        ),
        EvaluationCriteria(),
    )
    assert result.verdict == Verdict.INCONCLUSIVE
    assert _code(result, "E1_DSR").passed is not None


def test_missing_holdout_is_inconclusive() -> None:
    """No holdout observed yet: INCONCLUSIVE, other checks still reported."""
    result = evaluate_strategy(
        _good_inputs(holdout=None, holdout_returns=None), EvaluationCriteria(),
    )
    assert result.verdict == Verdict.INCONCLUSIVE
    assert _code(result, "E1_DSR").passed is True
    assert _code(result, "H1_HOLDOUT_GROWTH").passed is None


def test_luck_only_strategy_rejected_by_dsr() -> None:
    """Zero-drift returns with a lucky Sharpe: E1_DSR fails under N=96 priors."""
    base = _drift_returns(N_DISCOVERY, seed=0)
    flat = base - base.mean() + 0.01 / np.sqrt(365.0)
    flat_s = _series(flat, DISCOVERY_START)
    result = evaluate_strategy(
        _good_inputs(
            base_returns=flat_s,
            stress_returns=_series(flat - 0.0002, DISCOVERY_START),
            trial_population=_population(flat),
        ),
        EvaluationCriteria(),
    )
    assert _code(result, "E1_DSR").passed is False
    assert result.verdict == Verdict.REJECT


def test_single_year_dependence_rejected() -> None:
    """All growth from one year: leaving it out leaves nothing."""
    base = _drift_returns(N_DISCOVERY, seed=0)
    stress = base - 0.0002
    stress_s = _series(stress, DISCOVERY_START)
    concentrated = stress_s.copy()
    keep = concentrated.index.year == 2024
    concentrated.loc[~keep] = 0.0
    result = evaluate_strategy(_good_inputs(stress_returns=concentrated), EvaluationCriteria())
    assert _code(result, "R1_LEAVE_ONE_YEAR_OUT").passed is False
    assert result.verdict == Verdict.REJECT


def test_funding_squeeze_dependence_rejected() -> None:
    """Growth that turns negative without top-5 funding income: R2 fails."""
    funding = {f"S{i}USDT": 0.5 for i in range(7)}
    result = evaluate_strategy(_good_inputs(funding_by_symbol=funding), EvaluationCriteria())
    assert _code(result, "R2_FUNDING_DEPENDENCE").passed is False
    assert result.verdict == Verdict.REJECT


def test_knife_edge_rejected() -> None:
    """One neighbor with negative growth: the plateau is a knife edge."""
    base = _drift_returns(N_DISCOVERY, seed=0)
    neighbors = list(_good_inputs().neighbors)
    neighbors[0] = _series(-(base - 0.0002), DISCOVERY_START)
    result = evaluate_strategy(_good_inputs(neighbors=tuple(neighbors)), EvaluationCriteria())
    assert _code(result, "R3_PLATEAU").passed is False
    assert result.verdict == Verdict.REJECT


def test_integrity_failure_short_circuits() -> None:
    """Uncertified ledger: INVALID, no performance check evaluated."""
    result = evaluate_strategy(_good_inputs(ledger_certified=False), EvaluationCriteria())
    assert result.verdict == Verdict.INVALID
    assert _code(result, "E1_DSR").passed is None
    assert _code(result, "G1_GROWTH_LCB").passed is None


def test_holdout_break_rejected() -> None:
    """Holdout growth below the bootstrap 5% quantile: the strategy broke."""
    broken = np.full(N_HOLDOUT, -0.005) + _drift_returns(N_HOLDOUT, seed=13) * 0.2
    result = evaluate_strategy(
        _good_inputs(holdout_returns=_series(broken, HOLDOUT_START)), EvaluationCriteria(),
    )
    assert _code(result, "H1_HOLDOUT_GROWTH").passed is False
    assert result.verdict == Verdict.REJECT


def test_deterministic_digest() -> None:
    """Same inputs produce the same digest."""
    first = evaluate_strategy(_good_inputs(), EvaluationCriteria())
    second = evaluate_strategy(_good_inputs(), EvaluationCriteria())
    assert first.digest == second.digest


def test_nan_inputs_raise() -> None:
    """NaN returns fail closed instead of producing a verdict."""
    bad = _series(_drift_returns(N_DISCOVERY, seed=0), DISCOVERY_START).copy()
    bad.iloc[10] = float("nan")
    with pytest.raises(DataIntegrityError):
        evaluate_strategy(_good_inputs(base_returns=bad), EvaluationCriteria())


def test_no_paper_inputs_in_cli_builder() -> None:
    """The CLI builder never touches paper/live state: no src.live import."""
    from pathlib import Path

    source = Path("src/cli/commands/evaluate.py").read_text(encoding="utf-8")
    assert "src.live" not in source
    assert "data/state" not in source


def test_unknown_capacity_and_funding_are_inconclusive() -> None:
    import dataclasses

    inputs = dataclasses.replace(_good_inputs(), participation=pd.Series(dtype=float), funding_income_daily=None)
    result = evaluate_strategy(inputs, EvaluationCriteria())
    assert result.verdict == Verdict.INCONCLUSIVE
    assert _code(result, "R4_CAPACITY").passed is False
    assert _code(result, "R2_FUNDING_DEPENDENCE").passed is None
    result = evaluate_strategy(inputs, dataclasses.replace(EvaluationCriteria(), max_top5_funding_dependence=False))
    assert _code(result, "R2_FUNDING_DEPENDENCE").passed is True


def test_funding_removed_before_log_transform() -> None:
    import dataclasses

    inputs = _good_inputs()
    funding = pd.DataFrame({"X": 0.005}, index=inputs.stress_returns.index)
    adjusted = inputs.stress_returns.to_numpy() - 0.005
    expected = float(np.log1p(adjusted).sum())
    result = evaluate_strategy(dataclasses.replace(inputs, funding_income_daily=funding), EvaluationCriteria())
    assert _code(result, "R2_FUNDING_DEPENDENCE").value == pytest.approx(expected)


def test_bootstrap_different_horizon_and_initial_loss() -> None:
    values = np.full(90, -0.01)
    paths = standard._bootstrap_paths(values, 1095, 10, 5)
    assert paths.shape == (10, 1095)
    assert np.array_equal(paths, np.full((10, 1095), -0.01))
    result = evaluate_strategy(_good_inputs(deployed_returns=_series(values, DISCOVERY_START)), EvaluationCriteria())
    assert _code(result, "G1_GROWTH_LCB").passed is False


def test_constant_returns_rejected_without_sharpe() -> None:
    """Zero-variance returns carry no measurable edge: DSR is NaN, verdict REJECT."""
    n = N_DISCOVERY
    flat = _series(np.full(n, 0.001), DISCOVERY_START)
    result = evaluate_strategy(
        _good_inputs(
            base_returns=flat,
            stress_returns=_series(np.full(n, 0.0008), DISCOVERY_START),
            trial_population=TrialPopulation(
                family="flow_mom", prior_trials=96, sharpes=(), n_trials=97,
                sharpe_variance=1e-6,
            ),
        ),
        EvaluationCriteria(),
    )
    assert _code(result, "E1_DSR").passed is False
    assert _code(result, "E1_DSR").value is None
    assert result.verdict == Verdict.REJECT


def test_total_loss_day_rejected() -> None:
    """A -100% day wipes log growth: robustness fails, verdict REJECT."""
    stress = _series(_drift_returns(N_DISCOVERY, seed=0) - 0.0002, DISCOVERY_START)
    stress.iloc[100] = -1.0
    result = evaluate_strategy(_good_inputs(stress_returns=stress), EvaluationCriteria())
    assert _code(result, "E2_STRESS_GROWTH_LCB").passed is False
    assert _code(result, "E2_STRESS_GROWTH_LCB").value is None
    assert _code(result, "R2_FUNDING_DEPENDENCE").passed is False
    assert result.verdict == Verdict.REJECT




def test_single_year_discovery_reports_no_margin() -> None:
    """One calendar year of data: leave-one-year-out reports None, still INCONCLUSIVE."""
    n = 300
    start = pd.Timestamp("2025-01-01", tz="UTC")
    base = _series(_drift_returns(n, seed=0), start)
    result = evaluate_strategy(
        _good_inputs(
            base_returns=base,
            stress_returns=_series(base.to_numpy(dtype="float64") - 0.0002, start),
            participation=pd.Series(np.zeros(n), index=base.index, dtype="float64"),
            discovery=(start, start + pd.Timedelta(days=n - 1)),
            neighbors=tuple(_series(_drift_returns(n, 500 + i, 0.8), start) for i in range(7)),
            trial_population=_population(base.to_numpy(dtype="float64")),
        ),
        EvaluationCriteria(),
    )
    assert _code(result, "R1_LEAVE_ONE_YEAR_OUT").passed is False
    assert _code(result, "R1_LEAVE_ONE_YEAR_OUT").value is None
    assert result.verdict == Verdict.INCONCLUSIVE


def test_missing_neighbors_rejected() -> None:
    """No plateau evidence: R3 fails, verdict REJECT."""
    result = evaluate_strategy(_good_inputs(neighbors=()), EvaluationCriteria())
    assert _code(result, "R3_PLATEAU").passed is False
    assert result.verdict == Verdict.REJECT


def test_empty_trial_population_raises() -> None:
    """A population counting zero trials fails closed."""
    with pytest.raises(DataIntegrityError):
        evaluate_strategy(
            _good_inputs(
                trial_population=TrialPopulation(
                    family="flow_mom", prior_trials=0, sharpes=(),
                    n_trials=0, sharpe_variance=1e-6,
                )
            ),
            EvaluationCriteria(),
        )


def test_unknown_risk_envelope_raises() -> None:
    """Retired criteria keys fail closed instead of silently passing risk."""
    import dataclasses

    from src.strategy.release import _parse_criteria

    with pytest.raises(DataIntegrityError, match="retired key"):
        _parse_criteria({**dataclasses.asdict(EvaluationCriteria()), "risk_envelope": "growth"})
    with pytest.raises(DataIntegrityError, match="retired key"):
        _parse_criteria({**dataclasses.asdict(EvaluationCriteria()), "holdout_drawdown_max_quantile": 0.95})


def test_input_guards_reject_malformed_inputs() -> None:
    """Malformed windows, series, funding, and leverage fail closed."""
    import dataclasses

    good = _good_inputs()
    cases = [
        dataclasses.replace(good, trial_population=dataclasses.replace(good.trial_population, sharpe_variance=float("nan"))),
        dataclasses.replace(good, ledger_certified="false"),
        dataclasses.replace(good, participation=pd.Series([float("nan")])),
        dataclasses.replace(good, funding_income_daily=pd.DataFrame({"X": [0.1]})),
        dataclasses.replace(good, discovery="2021-04-01"),  # type: ignore[arg-type]
        dataclasses.replace(
            good,
            discovery=(pd.Timestamp("NaT", tz="UTC"), DISCOVERY_START + pd.Timedelta(days=10)),
        ),
        dataclasses.replace(good, base_returns=pd.Series(dtype="float64")),
        dataclasses.replace(good, strategy_id=""),
        dataclasses.replace(good, spec_digest=""),
        dataclasses.replace(
            good, discovery=(DISCOVERY_START, DISCOVERY_START - pd.Timedelta(days=1))
        ),
        dataclasses.replace(
            good,
            discovery=(pd.Timestamp("2021-04-01"), DISCOVERY_START + pd.Timedelta(days=10)),
        ),
        dataclasses.replace(
            good, design_data_cutoff=pd.Timestamp("2026-07-01")
        ),
        dataclasses.replace(good, base_returns=good.base_returns.iloc[:100]),
        dataclasses.replace(
            good,
            participation=pd.Series(
                np.full(N_DISCOVERY, -0.1), index=good.base_returns.index, dtype="float64"
            ),
        ),
        dataclasses.replace(good, funding_by_symbol=["x"]),  # type: ignore[arg-type]
        dataclasses.replace(good, funding_by_symbol={"": 0.1}),
        dataclasses.replace(good, funding_by_symbol={"A": float("nan")}),
        dataclasses.replace(good, deployed_max_leverage=float("inf")),
        dataclasses.replace(good, withdrawn_seat_fraction=-0.001),
        dataclasses.replace(good, participation_scale_to_deployed=0.0),
        dataclasses.replace(
            good, base_returns=pd.Series([0.01, 0.02], index=[0, 1], dtype="float64")
        ),
        dataclasses.replace(
            good,
            base_returns=pd.Series(
                np.ones(N_DISCOVERY) * 0.001,
                index=pd.date_range("2021-04-01", periods=N_DISCOVERY, freq="D"),
                dtype="float64",
            ),
        ),
        dataclasses.replace(
            good,
            base_returns=good.base_returns.iloc[::-1],
        ),
        dataclasses.replace(
            good,
            base_returns=_series(np.append(good.base_returns.to_numpy(dtype="float64"), [-2.0]),
                                 DISCOVERY_START).iloc[: N_DISCOVERY + 1],
        ),
        dataclasses.replace(good, deployed_stress_returns=good.deployed_stress_returns.iloc[::-1]),
        dataclasses.replace(good, deployed_stress_returns=good.deployed_returns.iloc[:100]),
        dataclasses.replace(good, deployed_liquidated=1),  # type: ignore[arg-type]
        dataclasses.replace(good, deployed_stress_liquidated="false"),  # type: ignore[arg-type]
        dataclasses.replace(good, deployed_margin_breaches=True),  # type: ignore[arg-type]
        dataclasses.replace(good, deployed_margin_breaches=-1),
    ]
    for inputs in cases:
        with pytest.raises(DataIntegrityError, match=r".+"):
            evaluate_strategy(inputs, EvaluationCriteria())


def test_i2_reports_measured_fraction_against_criteria() -> None:
    """I2 carries the withdrawn-seat fraction as value and the criteria cap as threshold."""
    result = evaluate_strategy(
        _good_inputs(lake_coverage_ok=True, withdrawn_seat_fraction=0.00025),
        EvaluationCriteria(),
    )
    check = _code(result, "I2_LAKE_COVERAGE")
    assert check.passed is True
    assert check.value == pytest.approx(0.00025)
    assert check.threshold == f"<= {EvaluationCriteria().max_withdrawn_seat_fraction}"


def test_i2_material_withdrawal_invalidates() -> None:
    """A 0.2 % withdrawal fails I2 and forces verdict INVALID."""
    result = evaluate_strategy(
        _good_inputs(lake_coverage_ok=True, withdrawn_seat_fraction=0.002),
        EvaluationCriteria(),
    )
    assert _code(result, "I2_LAKE_COVERAGE").passed is False
    assert result.verdict == Verdict.INVALID


def test_r4_scales_unit_participation_to_deployed_capital() -> None:
    """R4 values p95 x scale; 0.0002 x 2100x3/100000 passes the 0.01 line."""
    import dataclasses

    base = _good_inputs()
    participation = pd.Series(np.full(N_DISCOVERY, 0.0002), index=base.base_returns.index, dtype="float64")
    scale = 2100.0 * 3.0 / 100000.0
    result = evaluate_strategy(
        dataclasses.replace(base, participation=participation, participation_scale_to_deployed=scale),
        EvaluationCriteria(),
    )
    check = _code(result, "R4_CAPACITY")
    assert check.passed is True
    assert check.value == pytest.approx(0.0002 * scale)


def test_r4_missing_participation_fails_closed() -> None:
    """An empty participation series fails R4 instead of skipping it."""
    import dataclasses

    result = evaluate_strategy(
        dataclasses.replace(_good_inputs(), participation=pd.Series(dtype="float64")),
        EvaluationCriteria(),
    )
    check = _code(result, "R4_CAPACITY")
    assert check.passed is False
    assert "participation not recorded" in check.reason


@pytest.mark.parametrize(("daily", "passed"), [(0.002, True), (-0.002, False)])
def test_deployed_growth_floor_and_sizing(daily, passed) -> None:
    returns = _series(np.full(N_DISCOVERY, daily), DISCOVERY_START)
    result = evaluate_strategy(_good_inputs(deployed_stress_returns=returns), EvaluationCriteria())
    growth = _code(result, "G1_GROWTH_LCB")
    assert growth.value == pytest.approx(365 * np.log1p(daily))
    assert growth.passed is passed
    assert _code(result, "G2_NOT_OVERBET").passed is passed
    assert result.verdict == (Verdict.ACCEPT if passed else Verdict.REJECT)


def test_deployed_wipeout_stays_in_growth_distribution() -> None:
    returns = _series(np.full(N_DISCOVERY, 0.001), DISCOVERY_START)
    returns.iloc[100] = -1.0
    result = evaluate_strategy(_good_inputs(deployed_stress_returns=returns), EvaluationCriteria())
    assert _code(result, "G1_GROWTH_LCB").value == float("-inf")
    assert _code(result, "G1_GROWTH_LCB").passed is False
    assert _code(result, "G2_NOT_OVERBET").value == float("-inf")
    assert result.verdict == Verdict.REJECT


def test_high_variance_overbet_fails_growth_comparison() -> None:
    returns = _series(np.tile([0.32, -0.28], N_DISCOVERY // 2), DISCOVERY_START)
    result = evaluate_strategy(_good_inputs(deployed_stress_returns=returns), EvaluationCriteria())
    assert _code(result, "G2_NOT_OVERBET").value < 0
    assert _code(result, "G2_NOT_OVERBET").passed is False
    assert result.verdict == Verdict.REJECT


@pytest.mark.parametrize("overrides", [
    {"deployed_liquidated": True}, {"deployed_stress_liquidated": True},
    {"deployed_margin_breaches": 1},
])
def test_survival_fails_despite_positive_growth(overrides) -> None:
    result = evaluate_strategy(_good_inputs(**overrides), EvaluationCriteria())
    assert _code(result, "G1_GROWTH_LCB").passed is True
    assert _code(result, "G2_NOT_OVERBET").passed is True
    assert _code(result, "S1_SURVIVAL").passed is False
    assert result.verdict == Verdict.REJECT


def test_high_drawdown_and_leverage_are_not_verdict_gates() -> None:
    returns = _series(np.full(N_DISCOVERY, 0.01), DISCOVERY_START)
    returns.iloc[100] = -0.65
    paths = standard._bootstrap_paths(returns.to_numpy(), 1095, 200, standard.REPORT_BOOTSTRAP_SEED)
    equity = np.cumprod(1 + paths, axis=1)
    peak = np.maximum.accumulate(np.maximum(equity, 1), axis=1)
    probability = np.mean((1 - equity / peak).max(axis=1) > 0.6)
    assert 0.3 < probability < 0.7
    result = evaluate_strategy(
        _good_inputs(deployed_stress_returns=returns, deployed_max_leverage=10), EvaluationCriteria(),
    )
    assert result.verdict == Verdict.ACCEPT
    assert {c.code for c in result.checks}.isdisjoint({"K1_ENVELOPE", "H2_HOLDOUT_DRAWDOWN"})
    invalid = evaluate_strategy(_good_inputs(ledger_certified=False), EvaluationCriteria())
    for code in ("G1_GROWTH_LCB", "G2_NOT_OVERBET", "S1_SURVIVAL"):
        assert _code(invalid, code).passed is None
