"""Evidence-layer unit contract: selection-window overlap disclosure (I1).

SCENARIO_MHS_UNIT_REGRESSION_UNCHANGED: 아래 신설 케이스 포함, 이 디렉터리의
기존 MHS 유닛 스위트 전체는 두 수정이 가법적임을 확인하는 회귀 게이트다.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.mhs.evidence import (
    MIN_TRIAL_SHARPE_OUTCOMES,
    TRIAL_SHARPE_DEDUP_DECIMALS,
    causal_regime_labels,
    deflated_sharpe_decomposition,
    deflated_sharpe_ratio,
    distinct_trial_sr_variance,
    effective_observation_count,
    holdout_tail_evidence,
    regime_conditional_sharpe_blocks,
    selection_overlap_fraction,
)
from src.mhs.params import (
    DEFAULT_SELECTION_WINDOW,
    PERIODS_PER_YEAR_1H,
    PNL_VOL_TARGET_BURN_IN_DAYS,
)
from src.research.evaluation.policy import HOLDOUT_CUTOFF

_UTC = "UTC"


def test_selection_window_is_the_registered_defaults_span() -> None:
    """The disclosure denominator window matches the span the CLI defaults
    (growth_extreme, committee_kelly_sizing, breadth 60) were measured on."""
    registered = DEFAULT_SELECTION_WINDOW
    assert registered == (
        pd.Timestamp("2021-01-01", tz=_UTC),
        pd.Timestamp("2025-12-31", tz=_UTC),
    )


def test_full_containment_reports_exact_one() -> None:
    fraction = selection_overlap_fraction(
        pd.Timestamp("2021-01-01", tz=_UTC), pd.Timestamp("2025-12-31", tz=_UTC)
    )
    assert fraction == 1.0


def test_disjoint_and_zero_length_windows_report_zero() -> None:
    after = selection_overlap_fraction(
        pd.Timestamp("2026-01-01", tz=_UTC), pd.Timestamp("2026-12-31", tz=_UTC)
    )
    before = selection_overlap_fraction(
        pd.Timestamp("2019-01-01", tz=_UTC), pd.Timestamp("2020-12-31", tz=_UTC)
    )
    zero_length = selection_overlap_fraction(
        pd.Timestamp("2024-06-01", tz=_UTC), pd.Timestamp("2024-06-01", tz=_UTC)
    )
    assert after == 0.0
    assert before == 0.0
    assert zero_length == 0.0


def test_partial_overlap_is_fractional_and_clipped() -> None:
    start, end = DEFAULT_SELECTION_WINDOW
    half = selection_overlap_fraction(start, end + (end - start))
    assert half == pytest.approx(0.5)


def test_inverted_window_fails_closed() -> None:
    with pytest.raises(ValueError, match="report_end"):
        selection_overlap_fraction(
            pd.Timestamp("2026-01-01", tz=_UTC), pd.Timestamp("2025-01-01", tz=_UTC)
        )


def _ar1_series(phi: float, n: int, seed: int = 20260807) -> pd.Series:
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0, 1.0, n)
    values = np.empty(n)
    values[0] = innovations[0]
    for i in range(1, n):
        values[i] = phi * values[i - 1] + innovations[i]
    return pd.Series(values)


# SCENARIO_MHS_DSR_01_EFFECTIVE_N_SHRINKS_UNDER_POSITIVE_AUTOCORR
def test_SCENARIO_MHS_DSR_01_EFFECTIVE_N_SHRINKS_UNDER_POSITIVE_AUTOCORR() -> None:
    n = 5000
    positive = effective_observation_count(_ar1_series(0.6, n), 24)
    assert 1 <= positive < n

    iid = effective_observation_count(
        pd.Series(np.random.default_rng(1).normal(0.0, 1.0, n)), 24
    )
    assert abs(iid - n) <= 0.15 * n

    negative = effective_observation_count(_ar1_series(-0.6, n), 24)
    assert negative == n


# SCENARIO_MHS_DSR_01_EFFECTIVE_N_SHRINKS_UNDER_POSITIVE_AUTOCORR (I1/FAIL-CLOSED)
def test_effective_observation_count_fails_closed_and_stays_bounded() -> None:
    series = _ar1_series(0.3, 100)
    with pytest.raises(ValueError, match="max_lag"):
        effective_observation_count(series, 0)
    with pytest.raises(ValueError, match="observations"):
        effective_observation_count(series.head(10), 24)
    # Negatively autocorrelated input never inflates past the raw count.
    for phi in (-0.2, -0.4, -0.6, -0.9):
        count = effective_observation_count(_ar1_series(phi, 2000), 24)
        assert 1 <= count <= 2000


# SCENARIO_MHS_DSR_02_AUTOCORR_CORRECTION_LOWERS_DSR
def test_SCENARIO_MHS_DSR_02_AUTOCORR_CORRECTION_LOWERS_DSR() -> None:
    kwargs = {
        "observed_sr": 0.03083118,
        "trial_sr_variance": 1.2570356685e-04,
        "n_trials": 70,
        "skew": 0.8579,
        "kurtosis": 78.46,
    }
    raw = deflated_sharpe_ratio(n_obs=43823, **kwargs)
    assert raw == pytest.approx(0.793534, abs=1e-5)
    corrected = deflated_sharpe_ratio(n_obs=29250, **kwargs)
    assert corrected == pytest.approx(0.748218, abs=1e-5)
    # A2 monotonicity: the autocorrelation correction can only lower the DSR.
    assert corrected < raw


# SCENARIO_MHS_DSR_03_DECOMPOSITION_REPRODUCES_DSR
def test_SCENARIO_MHS_DSR_03_DECOMPOSITION_REPRODUCES_DSR() -> None:
    decomp = deflated_sharpe_decomposition(
        observed_sr=0.03083118,
        trial_sr_variance=1.2570356685e-04,
        n_trials=70,
        n_obs_raw=43823,
        n_obs_effective=29250,
        skew=0.8579,
        kurtosis=78.46,
        fold_sharpes=(0.019426, 0.024174, 0.014090, 0.040085),
    )
    assert decomp.benchmark_sr == pytest.approx(0.02693581, abs=1e-8)
    assert decomp.margin == pytest.approx(0.00389537, abs=1e-8)
    assert decomp.radicand == pytest.approx(0.991958, abs=1e-6)
    assert decomp.n_obs_raw == 43823
    assert decomp.n_obs_effective == 29250
    assert decomp.trial_sr_sqrt_variance == pytest.approx(math.sqrt(1.2570356685e-04))
    reproduced = norm.cdf(
        decomp.margin * math.sqrt(decomp.n_obs_effective - 1) / math.sqrt(decomp.radicand)
    )
    expected = deflated_sharpe_ratio(
        0.03083118, 1.2570356685e-04, 70, 29250, 0.8579, 78.46
    )
    assert reproduced == pytest.approx(expected, abs=1e-9)


# SCENARIO_MHS_DSR_03_DECOMPOSITION_REPRODUCES_DSR (FAIL-CLOSED validation)
def test_decomposition_propagates_dsr_validation() -> None:
    base = {
        "observed_sr": 0.03,
        "trial_sr_variance": 1.2570356685e-04,
        "n_trials": 70,
        "n_obs_raw": 43823,
        "n_obs_effective": 29250,
        "skew": 0.8579,
        "kurtosis": 78.46,
        "fold_sharpes": (),
    }
    with pytest.raises(ValueError, match="n_trials"):
        deflated_sharpe_decomposition(**{**base, "n_trials": 0})
    with pytest.raises(ValueError, match="n_obs"):
        deflated_sharpe_decomposition(**{**base, "n_obs_raw": 1})
    with pytest.raises(ValueError, match="n_obs"):
        deflated_sharpe_decomposition(**{**base, "n_obs_effective": 1})
    with pytest.raises(ValueError, match="trial_sr_variance"):
        deflated_sharpe_decomposition(**{**base, "trial_sr_variance": -1.0})


# SCENARIO_MHS_DSR_PASSAGE_DEDUP_VARIANCE_01
def test_SCENARIO_MHS_DSR_PASSAGE_DEDUP_VARIANCE_01() -> None:
    trials = [0.010, 0.010, 0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070]
    expected = float(np.var([0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070], ddof=1))
    # The fixture pool holds only 7 distinct outcomes, below the registered
    # minimum, so the dedup arithmetic is exercised with min_outcomes=7.
    variance = distinct_trial_sr_variance(
        trials, observed_sr=0.010, min_outcomes=7
    )
    # The four duplicate 0.010 entries (three trials plus observed_sr) collapse
    # to one pooled member.
    assert variance == expected
    # Duplicate re-runs cannot shrink V: appending 50 more copies of 0.010
    # returns the identical float.
    inflated = trials + [0.010] * 50
    assert distinct_trial_sr_variance(
        inflated, observed_sr=0.010, min_outcomes=7
    ) == expected


# SCENARIO_MHS_DSR_PASSAGE_OBSERVED_ALWAYS_POOLED_02
def test_SCENARIO_MHS_DSR_PASSAGE_OBSERVED_ALWAYS_POOLED_02() -> None:
    trials = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    # observed_sr=0.99 is an 8th distinct value: all 8 pool members survive and
    # the ddof=1 variance over exactly those 8 values is returned.
    variance = distinct_trial_sr_variance(trials, observed_sr=0.99)
    assert variance == float(np.var(sorted([0.99, *trials]), ddof=1))
    assert MIN_TRIAL_SHARPE_OUTCOMES == 8
    # observed_sr=0.01 duplicates a trial outcome: only 7 distinct outcomes
    # remain and the failure names both the count and the minimum.
    with pytest.raises(ValueError, match="distinct trial Sharpe outcomes") as excinfo:
        distinct_trial_sr_variance(trials, observed_sr=0.01)
    message = str(excinfo.value)
    assert "7" in message
    assert "8" in message


# SCENARIO_MHS_DSR_PASSAGE_DECOMPOSITION_REPRODUCES_05
def test_SCENARIO_MHS_DSR_PASSAGE_DECOMPOSITION_REPRODUCES_05() -> None:
    annualized_history = (-0.670, 0.412, 0.905, 1.204, 1.618, 1.877, 2.101, 2.456, 2.924)
    scale = math.sqrt(PERIODS_PER_YEAR_1H)
    fixture_pool = tuple(value / scale for value in annualized_history)
    observed_sr = 0.03083117885474785
    trial_sr_variance = distinct_trial_sr_variance(fixture_pool, observed_sr)
    pooled_distinct = len(
        {round(value, TRIAL_SHARPE_DEDUP_DECIMALS) for value in (observed_sr, *fixture_pool)}
    )
    decomposition = deflated_sharpe_decomposition(
        observed_sr=observed_sr,
        trial_sr_variance=trial_sr_variance,
        n_trials=130,
        n_obs_raw=43823,
        n_obs_effective=39031,
        skew=0.8579,
        kurtosis=78.4613,
        fold_sharpes=(0.019426, 0.024174, 0.014090, 0.040085),
        distinct_trial_outcomes=pooled_distinct,
    )
    reproduced = norm.cdf(
        decomposition.margin * math.sqrt(decomposition.n_obs_effective - 1)
        / math.sqrt(decomposition.radicand)
    )
    expected = deflated_sharpe_ratio(
        observed_sr, trial_sr_variance, 130, 39031, 0.8579, 78.4613
    )
    assert reproduced == pytest.approx(expected, abs=1e-12)
    assert decomposition.trial_sr_source == "distinct_trial_outcomes"
    assert decomposition.distinct_trial_outcomes == len(
        {round(value, TRIAL_SHARPE_DEDUP_DECIMALS) for value in (observed_sr, *fixture_pool)}
    )


# SCENARIO_MHS_DSR_PASSAGE_REGIME_LABELS_CAUSAL_08
def test_SCENARIO_MHS_DSR_PASSAGE_REGIME_LABELS_CAUSAL_08() -> None:
    rng = np.random.default_rng(20260825)
    # Deterministic cycle so all three BTC drawdown states occur: calm
    # uptrend, noisy bull, a ~-62% crash (bear), a volatile base, then a sharp
    # recovery. This is the INDEPENDENT reference price series, never the
    # book's own equity (that would make "Sharpe conditional on our own
    # drawdown" close to tautological).
    btc_segments = (
        np.full(800, 0.0005),
        rng.normal(0.0008, 0.008, 600),
        np.full(120, -0.008),
        rng.normal(0.0, 0.02, 500),
        np.full(80, 0.010),
        rng.normal(0.001, 0.01, 900),
    )
    index = pd.date_range("2021-01-01", periods=3000, freq="1h", tz="UTC")
    btc_returns = pd.Series(np.concatenate(btc_segments), index=index)
    btc_close = (1.0 + btc_returns).cumprod()
    # A DIFFERENT synthetic book return series (its own low/mid/high vol
    # blocks) so the two regime axes are independent, as they are in
    # production (book's own returns vs. an exogenous BTC price).
    book_segments = (
        rng.normal(0.0003, 0.003, 1000),
        rng.normal(0.0002, 0.012, 1000),
        rng.normal(0.0004, 0.006, 1000),
    )
    returns = pd.Series(np.concatenate(book_segments), index=index)
    n_hours = len(returns)
    labels_full = causal_regime_labels(returns, btc_close)
    # No future bar enters any label: the label at index t is bit-identical
    # whether computed on the prefix [:t+1] of both series or the full series.
    for cut in (400, 1200, n_hours - 1):
        labels_prefix = causal_regime_labels(
            returns.iloc[: cut + 1], btc_close.iloc[: cut + 1]
        )
        for column in labels_full.columns:
            assert (
                labels_prefix[column].to_numpy() == labels_full[column].to_numpy()[: cut + 1]
            ).all(), f"label {column} at cut={cut} is not causal"
    blocks = regime_conditional_sharpe_blocks(returns, btc_close)
    assert set(blocks) >= {
        "btc_drawdown_bull", "btc_drawdown_correction", "btc_drawdown_bear",
        "book_vol_low", "book_vol_mid", "book_vol_high",
    }
    for block in blocks.values():
        assert {"n_hours", "sharpe", "ann_vol"} <= set(block)
    for family in ("btc_drawdown_", "book_vol_"):
        total = sum(
            int(block["n_hours"])
            for name, block in blocks.items()
            if name.startswith(family)
        )
        assert total <= n_hours


# SCENARIO_MHS_SELECTION_EXEC_OVERLAP_REDUCED_03
def test_SCENARIO_MHS_SELECTION_EXEC_OVERLAP_REDUCED_03() -> None:
    """The final-OOS window (through 2026-06-30) discloses strictly less
    selection-window overlap than the default window (through 2025-12-31),
    via the unmodified selection_overlap_fraction."""
    final_oos_fraction = selection_overlap_fraction(
        pd.Timestamp("2021-01-01", tz=_UTC), pd.Timestamp("2026-06-30 23:59:59", tz=_UTC)
    )
    default_fraction = selection_overlap_fraction(
        pd.Timestamp("2021-01-01", tz=_UTC), pd.Timestamp("2025-12-31 23:59:59", tz=_UTC)
    )
    assert final_oos_fraction < default_fraction


# SCENARIO_HOLDOUT_TAIL_EVIDENCE_ISOLATES_POST_CUTOFF_PERFORMANCE
def test_holdout_tail_evidence_isolates_post_cutoff_performance() -> None:
    """봉인 경계(HOLDOUT_CUTOFF) 이후 구간만의 성과를 분리 계산하고, 미달
    표본/미통과 입력은 None으로 정직하게 신호하며 입력을 변이하지 않는다."""
    rng = np.random.default_rng(20260825)
    hours = pd.date_range("2021-01-01 00:00", "2026-06-30 23:00", freq="1h", tz="UTC")
    post = hours > HOLDOUT_CUTOFF
    hourly_pre = rng.normal(0.0, 0.003, int((~post).sum()))
    # cutoff 이후: 알려진 양의 일간 드리프트 + 소량 노이즈(std>0 보장).
    daily_drift = 0.002
    hourly_post = (
        (1.0 + daily_drift) ** (1.0 / 24.0) - 1.0
        + rng.normal(0.0, 0.0005, int(post.sum()))
    )
    hourly = np.empty(len(hours))
    hourly[~post] = hourly_pre
    hourly[post] = hourly_post
    equity = pd.Series(np.cumprod(1.0 + hourly), index=hours)
    snapshot = equity.copy(deep=True)

    tail = holdout_tail_evidence(equity, HOLDOUT_CUTOFF)

    assert tail is not None
    expected_days = len(pd.date_range("2026-01-01", "2026-06-30", freq="1D", tz="UTC"))
    assert tail["n_days"] == expected_days == 181
    assert set(tail) >= {
        "start", "end", "n_days", "total_return",
        "geometric_cagr", "max_drawdown", "naive_sharpe",
    }
    # 통계는 주입된 post-cutoff 드리프트만 반영(pre-cutoff 국면과 무관):
    # 일간 (1+d) 복리의 기하 CAGR와 총수익은 해석해 닫힌형으로 검증.
    exact_cagr = (1.0 + daily_drift) ** 365.25 - 1.0
    assert tail["geometric_cagr"] == pytest.approx(exact_cagr, rel=0.3)
    assert tail["total_return"] == pytest.approx((1.0 + daily_drift) ** 181 - 1.0, rel=0.3)
    assert tail["naive_sharpe"] > 10.0
    assert -0.05 <= tail["max_drawdown"] <= 0.0
    assert str(pd.Timestamp("2026-01-01", tz="UTC")) in str(tail["start"])

    # (b) 봉인을 아예 넘지 않는 시리즈: None.
    equity_pre_only = equity.loc[hours <= HOLDOUT_CUTOFF]
    assert holdout_tail_evidence(equity_pre_only, HOLDOUT_CUTOFF) is None

    # (c) 꼬리 표본이 min_days 미만(종료 10일 전 컷): None.
    short_cutoff = hours[-1] - pd.Timedelta(days=10)
    assert holdout_tail_evidence(equity, short_cutoff) is None
    assert PNL_VOL_TARGET_BURN_IN_DAYS == 90

    # (d) 입력 Series 불변.
    pd.testing.assert_series_equal(equity, snapshot)
