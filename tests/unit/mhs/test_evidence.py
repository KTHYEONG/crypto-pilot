"""Evidence-layer unit contract: selection-window overlap disclosure (I1)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.mhs.evidence import (
    deflated_sharpe_decomposition,
    deflated_sharpe_ratio,
    effective_observation_count,
    selection_overlap_fraction,
)
from src.mhs.params import DEFAULT_SELECTION_WINDOW

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
