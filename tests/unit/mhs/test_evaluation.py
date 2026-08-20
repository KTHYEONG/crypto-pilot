from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from src.mhs.evaluation import (
    AnchoredPurgedFold,
    DeploymentReadinessResult,
    autocorrelation_adjusted_sharpe,
    book_evidence,
    compute_deployment_readiness,
    cost_response_curve,
    deflated_sharpe_ratio,
    effective_breadth,
    phase_1_anchored_purged_folds,
    phase_diagnostic_metrics,
    probabilistic_sharpe_ratio,
    synthetic_stress_scenarios,
    tail_sensitivity_curve,
    year_restricted_correlation,
)
from src.mhs.execution import mhs_ledger_pnl
from src.mhs.execution import simulated_inventory_ledger


def test_mhs_5m_02_pit_roster_uses_only_eligible_trailing_volume() -> None:
    """MHS-5M-02-PIT-ROSTER: ranking is causal and eligibility masked."""
    from src.application.research.mhs.evaluation import _pit_execution_mask

    idx = pd.date_range("2025-01-01", periods=720, freq="1h", tz="UTC")
    volume = pd.DataFrame({"A": 10.0, "B": 20.0, "C": 30.0}, index=idx)
    eligible = pd.DataFrame(True, index=idx, columns=volume.columns)
    eligible.loc[idx[-1], "C"] = False
    mask = _pit_execution_mask(volume, eligible, 2)
    assert bool(mask.loc[idx[-1], "A"])
    assert bool(mask.loc[idx[-1], "B"])
    assert not bool(mask.loc[idx[-1], "C"])


def test_mhs_roster_hysteresis_enter_unchanged_from_baseline() -> None:
    """SCENARIO_MHS_HYSTERESIS_01_ENTER_UNCHANGED_FROM_BASELINE: with a
    720-row constant-volume fixture and ``universe_size=2``, the last row's
    mask is exactly ``{A: True, B: True, C: False}`` as before hysteresis was
    added -- entry via the top ``universe_size`` trailing-volume rank is
    unchanged."""
    from src.application.research.mhs.evaluation import _pit_execution_mask

    idx = pd.date_range("2025-01-01", periods=720, freq="1h", tz="UTC")
    volume = pd.DataFrame({"A": 10.0, "B": 20.0, "C": 30.0}, index=idx)
    eligible = pd.DataFrame(True, index=idx, columns=volume.columns)
    eligible.loc[idx[-1], "C"] = False
    mask = _pit_execution_mask(volume, eligible, 2)
    assert list(mask.loc[idx[-1], ["A", "B", "C"]]) == [True, True, False]


def test_mhs_roster_hysteresis_member_survives_rank_dip_within_exit_band() -> None:
    """SCENARIO_MHS_HYSTERESIS_02_MEMBER_SURVIVES_RANK_DIP_WITHIN_EXIT_BAND:
    a member whose rank worsens past ``universe_size`` but stays within the
    ``universe_size * MHS_EXECUTION_ROSTER_EXIT_MULTIPLIER`` band is retained,
    unlike the pre-fix hard cutoff which dropped it."""
    from src.application.research.mhs.evaluation import _pit_execution_mask
    from src.mhs.params import MHS_EXECUTION_ROSTER_EXIT_MULTIPLIER

    universe_size = 2
    exit_size = universe_size * MHS_EXECUTION_ROSTER_EXIT_MULTIPLIER
    idx = pd.date_range("2025-01-01", periods=721, freq="1h", tz="UTC")
    volume = pd.DataFrame({"A": 100.0, "B": 200.0, "C": 300.0}, index=idx)
    volume.loc[idx[720], "A"] = 200_000.0
    volume.loc[idx[720], "B"] = 100_000.0
    volume.loc[idx[720], "C"] = 100.0
    eligible = pd.DataFrame(True, index=idx, columns=volume.columns)
    # C is a member on bar 719 (rank 1); on bar 720 its trailing-volume rank
    # dips to 3 -- past universe_size but still inside the exit band.
    trailing = volume.rolling(720, min_periods=720).mean()
    ranked = trailing.where(eligible).rank(axis=1, ascending=False, method="first")
    assert ranked.loc[idx[719], "C"] == 1.0
    assert ranked.loc[idx[720], "C"] == 3.0
    assert exit_size >= 3.0

    mask = _pit_execution_mask(volume, eligible, universe_size)
    # Retained despite the rank dip (hysteresis); pre-fix rank<=2 would have dropped it.
    assert bool(mask.loc[idx[720], "C"])
    assert bool(mask.loc[idx[720], "A"])
    assert bool(mask.loc[idx[720], "B"])


def test_mhs_roster_hysteresis_member_exits_past_exit_band() -> None:
    """SCENARIO_MHS_HYSTERESIS_03_MEMBER_EXITS_PAST_EXIT_BAND: a member whose
    rank worsens past ``universe_size * MHS_EXECUTION_ROSTER_EXIT_MULTIPLIER``
    is dropped on that bar."""
    from src.application.research.mhs.evaluation import _pit_execution_mask
    from src.mhs.params import MHS_EXECUTION_ROSTER_EXIT_MULTIPLIER

    universe_size = 2
    exit_size = universe_size * MHS_EXECUTION_ROSTER_EXIT_MULTIPLIER
    idx = pd.date_range("2025-01-01", periods=721, freq="1h", tz="UTC")
    volume = pd.DataFrame(
        {"A": 100.0, "B": 200.0, "C": 300.0, "D": 400.0, "E": 500.0}, index=idx,
    )
    volume.loc[idx[720]] = [600_000.0, 500_000.0, 400_000.0, 100.0, 700_000.0]
    eligible = pd.DataFrame(True, index=idx, columns=volume.columns)
    trailing = volume.rolling(720, min_periods=720).mean()
    ranked = trailing.where(eligible).rank(axis=1, ascending=False, method="first")
    # D is a member on bar 719 (rank 2); on bar 720 its rank drops to 5, beyond exit_size=4.
    assert ranked.loc[idx[719], "D"] == 2.0
    assert ranked.loc[idx[720], "D"] == 5.0
    assert exit_size < 5.0

    mask = _pit_execution_mask(volume, eligible, universe_size)
    assert not bool(mask.loc[idx[720], "D"])
    assert bool(mask.loc[idx[720], "E"])


def test_mhs_roster_hysteresis_ineligible_exits_immediately() -> None:
    """SCENARIO_MHS_HYSTERESIS_04_INELIGIBLE_EXITS_IMMEDIATELY_REGARDLESS_OF_HYSTERESIS:
    a held member whose eligible flag becomes False is excluded on that same bar
    even though its raw trailing-volume rank is still inside the exit band."""
    from src.application.research.mhs.evaluation import _pit_execution_mask

    idx = pd.date_range("2025-01-01", periods=721, freq="1h", tz="UTC")
    volume = pd.DataFrame(
        {"A": 100.0, "B": 200.0, "C": 300.0, "D": 400.0, "E": 500.0}, index=idx,
    )
    eligible = pd.DataFrame(True, index=idx, columns=volume.columns)
    eligible.loc[idx[720], "D"] = False
    # D's raw trailing-volume rank is still 2 (well inside the exit band) -- only
    # the eligibility flip changes membership.
    raw_ranked = volume.rolling(720, min_periods=720).mean().rank(
        axis=1, ascending=False, method="first",
    )
    assert raw_ranked.loc[idx[720], "D"] == 2.0

    mask = _pit_execution_mask(volume, eligible, 2)
    assert not bool(mask.loc[idx[720], "D"])
    assert bool(mask.loc[idx[720], "E"])


def _wsf() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weights = pd.DataFrame({"A": [0.5, 0.5, -0.5], "B": [-0.5, -0.5, 0.5]})
    opens = pd.DataFrame({"A": [100.0, 101.0, 102.0], "B": [50.0, 49.0, 48.0]})
    funding = pd.DataFrame({"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]})
    return weights, opens, funding


def test_year_restricted_correlation_restricts_to_years() -> None:
    """SCENARIO_MHS_YEAR_RESTRICTED_CORRELATION_02: ``year_restricted_correlation``
    returns the correlation of the two series restricted to the requested
    calendar years (hand-computable with pandas directly), not the full-series
    correlation; an intersection with fewer than 3 points returns NaN rather
    than a spurious +/-1.0 from a 2-point intersection."""
    idx = pd.date_range("2021-01-01", periods=4 * 365, freq="1D", tz="UTC")
    rng = np.random.default_rng(5)
    a = pd.Series(rng.standard_normal(len(idx)), index=idx)
    b = a + rng.standard_normal(len(idx)) * 0.01
    expected = pd.concat(
        [a[a.index.year == 2023], b[b.index.year == 2023]], axis=1, join="inner",
    ).corr().iloc[0, 1]
    assert year_restricted_correlation(a, b, (2023,)) == pytest.approx(expected)
    assert year_restricted_correlation(a, b, (2021, 2022, 2023, 2024)) != pytest.approx(expected)

    short = pd.date_range("2024-01-01", periods=2, freq="1D", tz="UTC")
    c = pd.Series([1.0, 2.0], index=short)
    d = pd.Series([2.0, 1.0], index=short)
    assert math.isnan(year_restricted_correlation(c, d, (2024,)))


class TestCostResponseCurve:
    """MHS-08-COST-RESPONSE-MONOTONE: friction, not signal, is the binding constraint."""

    def test_non_increasing_in_rate_and_gross_at_zero(self) -> None:
        weights, opens, funding = _wsf()
        curve = cost_response_curve(weights, opens, funding, (0.0, 2.0, 4.0, 8.0), 365.0 * 24)
        rates = sorted(curve)
        for low, high in itertools.pairwise(rates):
            assert curve[high].net_ann <= curve[low].net_ann + 1e-12
        assert set(rates) == {0.0, 2.0, 4.0, 8.0}

    def test_must_include_measured_tiers(self) -> None:
        from src.mhs.contracts import MEASURED_EXECUTION_COST_TIERS_BPS

        weights, opens, funding = _wsf()
        grid = tuple(MEASURED_EXECUTION_COST_TIERS_BPS.values())
        curve = cost_response_curve(weights, opens, funding, grid, 365.0 * 24)
        assert set(curve) == set(grid)
        assert curve[grid[0]].net_ann > curve[grid[-1]].net_ann

    def test_fails_closed_on_empty_or_negative_grid(self) -> None:
        weights, opens, funding = _wsf()
        with pytest.raises(ValueError, match="must not be empty"):
            cost_response_curve(weights, opens, funding, (), 365.0 * 24)
        with pytest.raises(ValueError, match="must be >= 0"):
            cost_response_curve(weights, opens, funding, (-1.0,), 365.0 * 24)


class TestPhaseDiagnosticMetrics:
    """MHS-09-PHASE-DEGENERACY-FLAG: spread beyond the mean marks degenerate."""

    def test_degenerate_flag_and_inf_sharpe(self) -> None:
        a = pd.Series([0.01] * 4)
        b = pd.Series([0.05] * 4)
        result = phase_diagnostic_metrics({0: a, 1: b}, 4.0)
        assert result.n_phases == 2
        assert abs(result.min_phase_ann - 0.04) < 1e-12
        assert abs(result.max_phase_ann - 0.20) < 1e-12
        assert abs(result.mean_phase_ann - 0.12) < 1e-12
        assert abs(result.ensemble_ann - 0.12) < 1e-12
        assert abs(result.phase_spread_ann - 0.16) < 1e-12
        assert result.degenerate is True
        assert result.ensemble_sharpe == float("inf")

    def test_negative_inf_for_non_positive_zero_variance_ensemble(self) -> None:
        a = pd.Series([-0.01] * 4)
        b = pd.Series([-0.05] * 4)
        result = phase_diagnostic_metrics({0: a, 1: b}, 4.0)
        assert result.ensemble_sharpe == float("-inf")

    def test_fails_closed_on_empty(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            phase_diagnostic_metrics({}, 4.0)


class TestTailSensitivityCurve:
    """MHS-11-TAIL-CURVE-NOT-SINGLE-GATE: winsorize per symbol, events coalesce."""

    def test_contract_sandbox(self) -> None:
        weights = pd.DataFrame(
            {"A": [0.5, 0.5, 0.5, 0.5], "B": [-0.5, -0.5, -0.5, -0.5]},
        )
        fwd = pd.DataFrame(
            {"A": [0.02, 0.60, 0.50, 0.03], "B": [-0.01, -0.02, -0.02, -0.01]},
        )
        turnover = pd.Series([1.0, 0.0, 0.0, 0.0])
        result = tail_sensitivity_curve(weights, fwd, turnover, 8.0, 365.0, 1)
        assert result.winsor_curve[50][0] < result.base_net_ann
        assert result.event_count == 1
        assert result.top1_event_share > 0.5

    def test_adjacent_top_bars_coalesce_into_one_event(self) -> None:
        n = 200
        fwd = pd.DataFrame({"A": [0.001] * n}, dtype=float)
        fwd.loc[100, "A"] = 0.5
        fwd.loc[101, "A"] = 0.6
        weights = pd.DataFrame({"A": [1.0] * n}, dtype=float)
        turnover = pd.Series(0.0, index=weights.index)
        result = tail_sensitivity_curve(weights, fwd, turnover, 8.0, 365.0, 1)
        assert result.event_count == 1
        assert result.top1_event_share >= 0.8

    def test_winsor_curve_changes_across_caps(self) -> None:
        n = 200
        fwd = pd.DataFrame({"A": [0.001] * n}, dtype=float)
        fwd.loc[100, "A"] = 0.5
        fwd.loc[101, "A"] = 0.6
        weights = pd.DataFrame({"A": [1.0] * n}, dtype=float)
        turnover = pd.Series(0.0, index=weights.index)
        result = tail_sensitivity_curve(weights, fwd, turnover, 8.0, 365.0, 1)
        assert set(result.winsor_curve) == {10, 20, 30, 50}
        assert result.winsor_curve[10][0] < result.winsor_curve[50][0]

    def test_leave_worst_event_out_removes_full_window(self) -> None:
        n = 100
        fwd = pd.DataFrame({"A": [0.001] * n}, dtype=float)
        fwd.loc[50, "A"] = -0.9  # worst bar
        fwd.loc[49, "A"] = 0.0
        weights = pd.DataFrame({"A": [1.0] * n}, dtype=float)
        turnover = pd.Series(0.0, index=weights.index)
        result = tail_sensitivity_curve(weights, fwd, turnover, 8.0, 365.0, 1)
        assert not np.isnan(result.leave_worst_event_out_sharpe)

    def test_fails_closed_on_misaligned_frames(self) -> None:
        weights = pd.DataFrame({"A": [0.5]}, index=pd.RangeIndex(1))
        fwd = pd.DataFrame({"A": [0.01]}, index=pd.RangeIndex(2))
        turnover = pd.Series([0.0])
        with pytest.raises(ValueError, match="identically indexed"):
            tail_sensitivity_curve(weights, fwd, turnover, 8.0, 365.0, 1)


class TestAnchoredPurgedFolds:
    """MHS-16-PURGED-ANCHOR-BOUNDARY: embargo is 168h, derived from the forward dependency."""

    def test_three_preregistered_folds(self) -> None:
        folds = phase_1_anchored_purged_folds()
        assert len(folds) == 4
        assert [f.purge_hours for f in folds] == [168, 168, 168, 168]
        assert folds[0].train_end == pd.Timestamp("2021-12-31", tz="UTC")
        assert folds[0].validation_start == pd.Timestamp("2022-01-08", tz="UTC")
        assert folds[1].train_end == pd.Timestamp("2022-12-31", tz="UTC")
        assert folds[1].validation_start == pd.Timestamp("2023-01-08", tz="UTC")
        assert folds[2].train_end == pd.Timestamp("2023-12-31", tz="UTC")
        assert folds[2].validation_start == pd.Timestamp("2024-01-08", tz="UTC")
        assert folds[3].train_end == pd.Timestamp("2024-12-31", tz="UTC")
        assert folds[3].validation_start == pd.Timestamp("2025-01-08", tz="UTC")
        for fold in folds:
            assert fold.purge_hours >= fold.forward_dependency_hours
            embargo = fold.validation_start - fold.train_end
            assert embargo >= pd.Timedelta(hours=fold.purge_hours)
            assert embargo <= pd.Timedelta(hours=fold.purge_hours + 24)

    def test_embargo_must_be_positive(self) -> None:
        start = pd.Timestamp("2021-01-01", tz="UTC")
        end = pd.Timestamp("2022-12-31", tz="UTC")
        validation = pd.Timestamp("2023-12-31", tz="UTC")
        with pytest.raises(ValueError, match=r"ascending|embargo"):
            AnchoredPurgedFold(start, end, end, validation, 168, 168)

    def test_training_label_boundary_rule(self) -> None:
        folds = phase_1_anchored_purged_folds()
        # The first fold's training labels must end at or before train_end.
        assert folds[0].train_end == pd.Timestamp("2021-12-31", tz="UTC")
        assert folds[1].train_end == pd.Timestamp("2022-12-31", tz="UTC")


class TestCompoundingAlphaAxesFolds:
    """SCENARIO_MHS_COMPOUNDING_ALPHA_AXES_05: 4-fold coverage."""

    def test_four_folds_with_correct_dates(self) -> None:
        folds = phase_1_anchored_purged_folds()
        assert len(folds) == 4
        assert [f.validation_start.strftime("%Y-%m-%d") for f in folds] == [
            "2022-01-08", "2023-01-08", "2024-01-08", "2025-01-08",
        ]
        assert [f.validation_end.strftime("%Y-%m-%d") for f in folds] == [
            "2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31",
        ]

    def test_all_folds_have_168h_purge(self) -> None:
        folds = phase_1_anchored_purged_folds()
        for fold in folds:
            assert fold.purge_hours == 168
            assert fold.forward_dependency_hours == 168

    def test_all_folds_share_train_start_identity(self) -> None:
        from src.mhs.contracts import MHS_DISCOVERY_START
        folds = phase_1_anchored_purged_folds()
        for fold in folds:
            assert fold.train_start is MHS_DISCOVERY_START

    def test_embargo_bounds(self) -> None:
        folds = phase_1_anchored_purged_folds()
        for fold in folds:
            embargo = fold.validation_start - fold.train_end
            assert embargo >= pd.Timedelta(hours=168)
            assert embargo <= pd.Timedelta(hours=192)


class TestDeploymentReadiness:
    """MHS-17-CAPITAL-GO-DATA-BOUNDARY: Research GO can stand without forward data."""

    def test_readiness_from_strict_proxy_equity(self) -> None:
        marks = pd.DataFrame(
            {"A": [100.0, 110.0, 121.0, 133.1]},
            index=pd.date_range("2021-01-01", periods=4, freq="1D", tz="UTC"),
        )
        fills = pd.DataFrame(
            [{"timestamp": marks.index[0], "symbol": "A", "quantity_delta": 1.0,
              "fill_price": 100.0, "fee_bps": 0.0, "reason": "passive_fill"}],
        )
        ledger = simulated_inventory_ledger(
            fills, marks, pd.DataFrame(0.0, index=marks.index, columns=["A"]),
            1.0, "OHLCV_STRICT_PROXY", "MARK_PRICE",
        )
        result = compute_deployment_readiness(
            ledger.equity, 365.0, n_bootstrap=50, mean_block_bars=2,
        )
        assert result.geometric_cagr > 0
        assert result.max_drawdown <= 0
        assert result.research_go_eligible is True
        assert result.execution_go_eligible is False
        assert result.pilot_go_eligible is False
        assert result.scale_go_eligible is False

    def test_eligibility_flags_are_booleans(self) -> None:
        assert DeploymentReadinessResult.__dataclass_fields__["research_go_eligible"].type is bool
        assert DeploymentReadinessResult.__dataclass_fields__["scale_go_eligible"].type is bool


class TestMarkPriceGoValidity:
    """MHS-MARK-05-GO-VALIDITY: a primary_valid=False replay never yields a Research GO."""

    def test_invalid_primary_blocks_research_go_but_preserves_metrics(self) -> None:
        idx = pd.date_range("2021-01-01", periods=10, freq="1h", tz="UTC")
        equity = pd.Series(np.cumprod(1.0 + np.full(10, 0.001)), index=idx)
        invalid = compute_deployment_readiness(equity, 8760.0, primary_valid=False, n_bootstrap=2, mean_block_bars=1)
        assert invalid.research_go_eligible is False
        assert invalid.execution_go_eligible is False
        assert invalid.pilot_go_eligible is False
        assert invalid.scale_go_eligible is False
        valid = compute_deployment_readiness(equity, 8760.0, primary_valid=True, n_bootstrap=2, mean_block_bars=1)
        assert valid.research_go_eligible is True
        assert invalid.geometric_cagr == pytest.approx(valid.geometric_cagr)
        assert invalid.max_drawdown == valid.max_drawdown
        assert invalid.expected_shortfall == pytest.approx(valid.expected_shortfall)

    def test_rejects_non_bool_primary_valid(self) -> None:
        idx = pd.date_range("2021-01-01", periods=3, freq="1h", tz="UTC")
        equity = pd.Series([1.0, 1.01, 1.02], index=idx)
        with pytest.raises(ValueError, match="bool"):
            compute_deployment_readiness(equity, 8760.0, primary_valid=1, n_bootstrap=2)


class TestSyntheticStressScenarios:
    """MHS-18-SYNTHETIC-STRESS-AND-TERMINATION-CLASSES: nine deterministic shocks."""

    EXPECTED_NAMES = frozenset({
        "BTC_DOWN_10",
        "BTC_DOWN_20",
        "ALT_BETA_UP",
        "XS_CORRELATION_ONE",
        "SPREAD_AND_COST_X3",
        "PASSIVE_FILL_DEGRADATION",
        "FUNDING_EXTREME",
        "LIQUIDITY_DETERIORATION_50PCT",
        "VENUE_API_OUTAGE_30M",
    })

    def test_all_nine_scenarios_present(self) -> None:
        scenarios = synthetic_stress_scenarios()
        assert len(scenarios) == 9
        assert {s.name for s in scenarios} == self.EXPECTED_NAMES
        assert all(s.description for s in scenarios)


class TestAutocorrelationAdjustedSharpe:
    """MHS-22-PRIMARY-SHARPE-AUTOCORRELATION-ADJUSTED: daily compounding, 7-day adjustment."""

    def test_constant_series_is_inf(self) -> None:
        r = pd.Series(
            [0.01] * 10,
            index=pd.date_range("2021-01-01", periods=10, freq="1D", tz="UTC"),
        )
        assert autocorrelation_adjusted_sharpe(r, 365, 7) == float("inf")

    def test_positive_autocorrelation_reduces_adjusted_sharpe(self) -> None:
        rng = np.random.default_rng(0)
        idx = pd.date_range("2021-01-01", periods=120, freq="1D", tz="UTC")
        eps = rng.normal(0.0, 0.01, len(idx))
        returns = pd.Series(0.0005 + eps, index=idx)
        returns = pd.Series(np.cumsum(0.4 * np.r_[0.0, returns.to_numpy()[:-1]]) * 0.0 + returns.to_numpy(), index=idx)
        naive = returns.mean() / returns.std(ddof=1) * np.sqrt(365)
        adjusted = autocorrelation_adjusted_sharpe(returns, 365, 7)
        assert adjusted < naive

    def test_fails_closed_on_tz_naive(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            autocorrelation_adjusted_sharpe(
                pd.Series([0.01] * 10, index=pd.date_range("2021-01-01", periods=10)),
                365, 7,
            )

def _scalar_bootstrap_reference(
    net_returns: np.ndarray, n_replicates: int, mean_block: int, seed: int, mdd: bool,
) -> np.ndarray:
    """Original scalar while-loop block bootstrap (pre-refactor reference)."""
    rng = np.random.default_rng(seed + (1 if mdd else 0))
    n = len(net_returns)
    p_block = 1.0 / mean_block if mean_block > 0 else 0.0
    out = np.empty(n_replicates, dtype="float64")
    for r in range(n_replicates):
        blocks: list[float] = []
        while len(blocks) < n:
            start = int(rng.integers(0, n))
            length = 1
            while length < n and rng.random() > p_block:
                length += 1
            length = min(length, n - len(blocks))
            blocks.extend(net_returns[start : start + length].tolist())
        path = np.array(blocks[:n], dtype="float64")
        if mdd:
            eq = np.cumprod(1.0 + path)
            out[r] = float((eq / np.maximum.accumulate(eq) - 1.0).min())
        else:
            out[r] = float(np.prod(1.0 + path))
    return out


class TestMhsPerfOptimizationO1Bootstrap:
    """SCENARIO_O1_STAT_EQUIV_BOOTSTRAP: the vectorized block bootstrap is
    statistically equivalent to the scalar while-loop reference (the RNG draw
    order differs by design, so exact reproduction is neither required nor
    possible -- matching the MHS_PERF_OPT_003 precedent for _bootstrap_ci)."""

    @staticmethod
    def _fixture_returns() -> np.ndarray:
        rng = np.random.default_rng(5)
        prices = np.cumsum(rng.normal(0.0, 0.001, 6000))
        return np.diff(prices).astype("float64")

    @pytest.mark.slow
    def test_wealth_paths_statistically_equivalent(self) -> None:
        from src.mhs.evaluation import _stationary_block_bootstrap_paths

        arr = self._fixture_returns()
        ref = _scalar_bootstrap_reference(arr, 800, 168, 20260807, mdd=False)
        new = _stationary_block_bootstrap_paths(arr, 800, 168, 20260807)
        assert new.shape == (800,)
        assert np.isfinite(new).all()
        # Statistical equivalence: distribution location/scale within tolerance.
        assert abs(new.mean() - ref.mean()) / abs(ref.mean()) < 0.05
        assert abs(np.percentile(new, 5) - np.percentile(ref, 5)) / abs(np.percentile(ref, 5)) < 0.10

    @pytest.mark.slow
    def test_mdd_paths_statistically_equivalent(self) -> None:
        from src.mhs.evaluation import _bootstrap_mdd_paths

        arr = self._fixture_returns()
        ref = _scalar_bootstrap_reference(arr, 800, 168, 20260807, mdd=True)
        new = _bootstrap_mdd_paths(arr, 800, 168, 20260807)
        assert new.shape == (800,)
        assert np.isfinite(new).all()
        assert new.max() <= 0.0
        # MDD is a tail statistic extremely sensitive to block boundaries; an
        # absolute 0.01 tolerance on the mean is the statistically-equivalent
        # gate (matching the MHS_PERF_OPT_003 absolute-tolerance precedent).
        assert abs(new.mean() - ref.mean()) < 0.01

    def test_rejects_bad_arguments(self) -> None:
        from src.mhs.evaluation import _bootstrap_mdd_paths, _stationary_block_bootstrap_paths

        arr = np.array([0.001, -0.002, 0.0005], dtype="float64")
        with pytest.raises(ValueError, match="n_replicates"):
            _stationary_block_bootstrap_paths(arr, 0, 2, 1)
        with pytest.raises(ValueError, match="mean_block"):
            _bootstrap_mdd_paths(arr, 5, -1, 1)

    def test_empty_returns_shape(self) -> None:
        from src.mhs.evaluation import _bootstrap_mdd_paths, _stationary_block_bootstrap_paths

        empty = np.array([], dtype="float64")
        assert np.array_equal(_stationary_block_bootstrap_paths(empty, 5, 168, 1), np.ones(5))
        assert np.array_equal(_bootstrap_mdd_paths(empty, 5, 168, 1), np.zeros(5))


class TestMhsPerfOptimizationO1DeploymentReadiness:
    """SCENARIO_O1_STAT_EQUIV_DEPLOYMENT_READINESS: non-bootstrap scalar fields
    are bit-identical to the pre-refactor golden; bootstrap-derived
    probabilities are statistically equivalent; ruin_probs is exactly {0.0, 1.0}
    (single deterministic cumprod check per leverage)."""

    @staticmethod
    def _fixture_equity() -> pd.Series:
        rng = np.random.default_rng(7)
        n = 43_830
        net = rng.normal(0.00001, 0.01, n)
        return pd.Series(
            np.cumprod(1.0 + net),
            index=pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC"),
        )

    def test_non_bootstrap_fields_bit_identical_to_golden(self) -> None:
        rng = np.random.default_rng(7)
        n = 1000
        net = rng.normal(0.00001, 0.01, n)
        equity = pd.Series(
            np.cumprod(1.0 + net),
            index=pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC"),
        )
        res = compute_deployment_readiness(
            equity, 8760.0, n_bootstrap=20, mean_block_bars=24, seed=20260807,
        )
        # Deterministic arithmetic independent of the bootstrap RNG stream.
        net = equity.pct_change().dropna()
        final, initial = float(equity.iloc[-1]), float(equity.iloc[0])
        expected_cagr = (final / initial) ** (8760.0 / len(net)) - 1.0
        assert res.geometric_cagr == expected_cagr
        assert res.max_drawdown <= 0.0
        assert res.worst_1d == float(net.min())
        assert res.worst_7d <= res.worst_1d or res.worst_7d == float(net.min())
        assert res.recovery_bars is None or isinstance(res.recovery_bars, int)
        assert res.time_under_water_bars >= 0

    def test_ruin_probs_are_zero_or_one(self) -> None:
        res = compute_deployment_readiness(
            self._fixture_equity(), 8760.0, n_bootstrap=20, mean_block_bars=168,
        )
        for prob in res.leverage_ruin_probabilities.values():
            assert prob in (0.0, 1.0)

    def test_probabilities_finite_bounded(self) -> None:
        res = compute_deployment_readiness(
            self._fixture_equity(), 8760.0, n_bootstrap=20, mean_block_bars=168,
        )
        assert 0.0 <= res.probability_final_wealth_below_initial <= 1.0
        assert 0.0 <= res.probability_mdd_over_20pct <= 1.0
        assert 0.0 <= res.probability_mdd_over_30pct <= 1.0


def _book_evidence_panel(
    n: int = 400, cols: tuple[str, ...] = ("A", "B", "C", "D"), seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministic hourly open/funding panel plus an equal-weight book."""
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)
    opens = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.standard_normal((n, len(cols))) * 0.01, axis=0)),
        index=idx, columns=cols,
    )
    funding = pd.DataFrame(0.0, index=idx, columns=cols)
    weights = pd.DataFrame(np.full((n, len(cols)), 1.0 / len(cols)), index=idx, columns=cols)
    return weights, opens, funding


def _inline_book_evidence(
    weights_1h: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    cost_grid: tuple[float, ...],
    periods_per_year: float,
    event_window_bars: int,
) -> tuple[dict[float, object], object]:
    """The pre-change inline construction from ``_book_outcome`` (the reference
    book only): the exact block the ``book_evidence`` extraction replaced."""
    prescreen = cost_response_curve(
        weights_1h, opens, bar_funding, cost_grid, periods_per_year,
    )
    effective_weights = weights_1h.shift(2).fillna(0.0)
    fwd = opens.pct_change()
    _net, turnover = mhs_ledger_pnl(weights_1h, opens, bar_funding, 8.0)
    tail = tail_sensitivity_curve(
        effective_weights, fwd, turnover, 8.0, periods_per_year, event_window_bars,
    )
    return prescreen, tail


class TestBookEvidence:
    """SCENARIO_MHS_BOOK_EVIDENCE_MATCHES_INLINE_BLOCK_01: ``book_evidence`` is
    a pure, behavior-preserving extraction of the reference-book significance
    block -- both instruments must reproduce the inline construction exactly."""

    def test_matches_inline_block(self) -> None:
        weights, opens, funding = _book_evidence_panel()
        cost_grid = (0.0, 2.64, 4.18, 8.0)
        ev = book_evidence(weights, opens, funding, cost_grid, 8760.0, 168)
        expected_prescreen, expected_tail = _inline_book_evidence(
            weights, opens, funding, cost_grid, 8760.0, 168,
        )
        assert ev.prescreen == expected_prescreen
        assert ev.tail == expected_tail
        assert set(ev.prescreen) == set(cost_grid)

    def test_net_t_monotone_in_cost(self) -> None:
        weights, opens, funding = _book_evidence_panel()
        ev = book_evidence(weights, opens, funding, (0.0, 2.64, 4.18, 8.0), 8760.0, 168)
        for low, high in itertools.pairwise(sorted(ev.prescreen)):
            assert ev.prescreen[high].net_ann <= ev.prescreen[low].net_ann + 1e-12

    def test_distinguishes_reference_from_executed(self) -> None:
        """SCENARIO_MHS_BOOK_EVIDENCE_DISTINGUISHES_REFERENCE_FROM_EXECUTED_02:
        two different weight books over the same panel produce different
        prescreen net_t values -- the helper measures the book it is handed,
        not any ambient state (the unit-level RC-1 guard)."""
        weights, opens, funding = _book_evidence_panel()
        concentrated = weights.copy()
        concentrated[["A", "B"]] = 0.0
        concentrated[["C", "D"]] = 0.5
        cost_grid = (0.0, 4.18)
        wide = book_evidence(weights, opens, funding, cost_grid, 8760.0, 168)
        narrow = book_evidence(concentrated, opens, funding, cost_grid, 8760.0, 168)
        assert wide.prescreen[4.18].net_t != narrow.prescreen[4.18].net_t
        assert wide.tail.base_net_ann != narrow.tail.base_net_ann

    def test_fails_closed(self) -> None:
        """SCENARIO_MHS_BOOK_EVIDENCE_FAILS_CLOSED_03: an empty cost grid, a
        negative cost rate, periods_per_year <= 0, event_window_bars < 1, and a
        negative tail one-way rate each raise ValueError -- never a degenerate
        or NaN-filled result."""
        weights, opens, funding = _book_evidence_panel()
        with pytest.raises(ValueError, match="must not be empty"):
            book_evidence(weights, opens, funding, (), 8760.0, 168)
        with pytest.raises(ValueError, match=">= 0"):
            book_evidence(weights, opens, funding, (-1.0,), 8760.0, 168)
        with pytest.raises(ValueError, match="> 0"):
            book_evidence(weights, opens, funding, (0.0,), 0.0, 168)
        with pytest.raises(ValueError, match=">= 1"):
            book_evidence(weights, opens, funding, (0.0,), 8760.0, 0)
        with pytest.raises(ValueError, match=">= 0"):
            book_evidence(weights, opens, funding, (0.0,), 8760.0, 168, -1.0)


class TestDeflatedSharpeRatio:
    """SCENARIO_MHS_DEFLATED_SHARPE_DEFLATES_WITH_TRIALS_06: the Bailey-LdP
    multiple-testing adjustment deflates harder as the number of trials grows,
    collapses to the plain PSR at zero trial dispersion, and stays in [0, 1]."""

    def test_strictly_decreasing_in_trials(self) -> None:
        few = deflated_sharpe_ratio(0.12, 0.0025, 5, 1200, 0.0, 3.0)
        mid = deflated_sharpe_ratio(0.12, 0.0025, 100, 1200, 0.0, 3.0)
        many = deflated_sharpe_ratio(0.12, 0.0025, 500, 1200, 0.0, 3.0)
        assert few > mid > many
        assert 0.0 <= many <= few <= 1.0

    def test_zero_trial_variance_equals_psr_against_zero_benchmark(self) -> None:
        d = deflated_sharpe_ratio(0.12, 0.0, 50, 1200, 0.0, 3.0)
        p = probabilistic_sharpe_ratio(0.12, 0.0, 1200, 0.0, 3.0)
        assert d == pytest.approx(p, rel=1e-12)
        assert 0.0 <= d <= 1.0

    def test_observed_sr_greater_than_benchmark_gives_high_psr(self) -> None:
        v = probabilistic_sharpe_ratio(0.1, 0.0, 500, 0.0, 3.0)
        assert 0.5 < v <= 1.0
        assert probabilistic_sharpe_ratio(0.0, 0.0, 500, 0.0, 3.0) == 0.5

    def test_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="n_trials"):
            deflated_sharpe_ratio(0.1, 0.0, 0, 100, 0.0, 3.0)
        with pytest.raises(ValueError, match="n_obs"):
            deflated_sharpe_ratio(0.1, 0.0, 5, 1, 0.0, 3.0)
        with pytest.raises(ValueError, match="trial_sr_variance"):
            deflated_sharpe_ratio(0.1, -0.1, 5, 100, 0.0, 3.0)
        with pytest.raises(ValueError, match="n_obs"):
            probabilistic_sharpe_ratio(0.1, 0.0, 1, 0.0, 3.0)

    def test_degenerate_radicand_returns_nan_not_inf(self) -> None:
        # A radicand <= 0 (e.g. extreme skew/observed Sharpe) must yield NaN,
        # never an inf or complex value.
        assert np.isnan(probabilistic_sharpe_ratio(0.3, 0.0, 500, 10.0, 3.0))


class TestEffectiveBreadth:
    """SCENARIO_MHS_EFFECTIVE_BREADTH_MATCHES_KNOWN_CASES_03."""

    def test_independent_columns_match_nominal(self) -> None:
        # N independent random return columns yield n_eff within sampling error
        # of N, with near-zero mean pairwise correlation.
        rng = np.random.default_rng(0)
        indep = pd.DataFrame(rng.standard_normal((300, 5)), columns=list("abcde"))
        n_eff, mean_corr = effective_breadth(indep)
        assert 3.5 <= n_eff <= 5.0
        assert -0.2 <= mean_corr <= 0.2

    def test_perfectly_correlated_columns_collapse_to_one(self) -> None:
        # N perfectly correlated (identical) columns yield n_eff == 1.0.
        rng = np.random.default_rng(0)
        base = rng.standard_normal(300)
        dup = pd.DataFrame(dict.fromkeys(list("abcd"), base))
        n_eff, _ = effective_breadth(dup)
        assert abs(n_eff - 1.0) < 0.05

    def test_mixed_panel_strictly_between(self) -> None:
        # A mixed panel (two correlated columns, two independent) yields n_eff
        # strictly between 1.0 and N -- the direct correctness check behind the
        # n_eff=1.76/19 and n_eff=3.05/28 saturation claims in the spec.
        rng = np.random.default_rng(0)
        base = rng.standard_normal(300)
        frame = pd.DataFrame(
            {
                "a": base,
                "b": base + 0.1 * rng.standard_normal(300),
                "c": rng.standard_normal(300),
                "d": rng.standard_normal(300),
            }
        )
        n_eff, _ = effective_breadth(frame)
        assert 1.0 < n_eff < 4.0

    def test_zero_variance_column_is_absorbed_not_nan(self) -> None:
        # A constant (zero-variance) column must not poison the correlation
        # matrix with NaNs; the non-finite entries become 0.0 and the diagonal
        # is forced to 1.0, so n_eff stays finite and within [1.0, N].
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            {
                "a": rng.standard_normal(300),
                "b": rng.standard_normal(300),
                "c": 3.0,
            }
        )
        n_eff, _ = effective_breadth(frame)
        assert 1.0 <= n_eff <= 3.0
        assert np.isfinite(n_eff)

    def test_fails_closed_on_too_few_rows_or_columns(self) -> None:
        with pytest.raises(ValueError, match="effective_breadth"):
            effective_breadth(pd.DataFrame({"a": [1.0, 2.0]}))
        with pytest.raises(ValueError, match="effective_breadth"):
            effective_breadth(pd.DataFrame({"a": [1.0], "b": [2.0]}))


