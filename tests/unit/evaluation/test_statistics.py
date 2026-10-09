"""Spec 38 part 6: growth sampling statistics always computable, never lenient."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.evaluation.statistics import deflated_sharpe_ratio, sharpe_sampling_variance


def _utc_series(values: np.ndarray, start: str = "2021-04-01") -> pd.Series:
    index = pd.date_range(start=start, periods=len(values), freq="D", tz="UTC")
    return pd.Series(np.asarray(values, dtype="float64"), index=index, dtype="float64")


def _fat_tail_drift(n: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    noise = rng.standard_t(5, size=n) * np.sqrt(3.0 / 5.0)
    return _utc_series(0.01 * 2.0 / np.sqrt(365.0) + 0.01 * noise)


def test_dsr_always_computable_with_prior_search_only() -> None:
    """One recorded trial plus the documented 96 priors: DSR is finite via the floor."""
    from src.evaluation.trials import TrialPopulation

    series = _fat_tail_drift(1826, seed=0)
    values = series.to_numpy(dtype="float64")
    observed = float(values.mean() / np.std(values, ddof=1))
    skew = float(pd.Series(values).skew())
    kurtosis = float(pd.Series(values).kurt() + 3.0)
    floor = sharpe_sampling_variance(observed, len(values), skew, kurtosis)
    population = TrialPopulation(
        family="flow_mom", prior_trials=96, sharpes=(), n_trials=97, sharpe_variance=floor,
    )
    dsr = deflated_sharpe_ratio(
        observed, population.sharpe_variance, population.n_trials, len(values), skew, kurtosis,
    )
    assert math.isfinite(dsr)


def test_sampling_variance_floor_never_more_lenient() -> None:
    """Tiny cross-trial dispersion: the floor, not the cross variance, is used."""
    from src.evaluation.trials import trial_population

    from src.evaluation.trials import TrialRecord
    from src.evaluation.trials import append_trial

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "flow_mom.trials.jsonl"
        for digest in ("aaa", "bbb"):
            append_trial(
                TrialRecord(
                    family="flow_mom", spec_digest=digest,
                    window_start=pd.Timestamp("2021-01-01", tz="UTC"),
                    window_end=pd.Timestamp("2026-07-01", tz="UTC"),
                    daily_sharpe=0.05 if digest == "aaa" else 0.0500001,
                    n_obs=1826, recorded_at=pd.Timestamp("2026-10-08", tz="UTC"), source="cli",
                ),
                path=path,
            )
        population = trial_population(
            "flow_mom", path=path,
            candidate_sharpe=0.06, candidate_n_obs=1826, candidate_skew=0.0, candidate_kurtosis=9.0,
        )
    floor = sharpe_sampling_variance(0.06, 1826, 0.0, 9.0)
    assert population.sharpe_variance == floor


def test_growth_lcb_orders_with_alpha() -> None:
    """Wider confidence bands sit lower: LCB(0.05) <= LCB(0.25) <= mean."""
    import src.evaluation.statistics as statistics

    series = _fat_tail_drift(1826, seed=0)
    lcb_05 = statistics.log_growth_lcb(series, alpha=0.05, n_paths=200, seed=20261008)
    lcb_25 = statistics.log_growth_lcb(series, alpha=0.25, n_paths=200, seed=20261008)
    mean = float(np.log1p(series.to_numpy(dtype="float64")).mean() * 365.0)
    assert lcb_05 <= lcb_25 <= mean
    assert math.isfinite(lcb_05)
    assert math.isfinite(lcb_25)


def test_renamed_helpers_keep_values() -> None:
    import pytest

    from src.evaluation.statistics import expected_max_trial_sr, psr_radicand

    assert psr_radicand(0.05, 0.2, 3.5) == pytest.approx(0.9915625)
    assert expected_max_trial_sr(0.0025, 100) == pytest.approx(0.12653014466008425)
    assert psr_radicand(-0.1, -0.5, 4.0) == pytest.approx(0.9575)
    assert expected_max_trial_sr(0.01, 50) == pytest.approx(0.22763030934203485)


def test_psr_and_dsr_reject_degenerate_inputs() -> None:
    """Degenerate moments fail closed (NaN or ValueError), never silently pass."""
    import pytest

    from src.evaluation.statistics import probabilistic_sharpe_ratio

    with pytest.raises(ValueError, match="n_obs"):
        probabilistic_sharpe_ratio(0.1, 0.0, 1, 0.0, 3.0)
    assert math.isnan(probabilistic_sharpe_ratio(1.0, 0.0, 10, 0.0, -3.0))
    with pytest.raises(ValueError, match="n_trials"):
        deflated_sharpe_ratio(0.1, 0.01, 0, 100, 0.0, 3.0)
    with pytest.raises(ValueError, match="n_obs"):
        deflated_sharpe_ratio(0.1, 0.01, 5, 1, 0.0, 3.0)
    with pytest.raises(ValueError, match="trial_sr_variance"):
        deflated_sharpe_ratio(0.1, -0.01, 5, 100, 0.0, 3.0)
    assert math.isfinite(deflated_sharpe_ratio(0.1, 0.0, 96, 500, 0.0, 3.0))
    assert math.isfinite(deflated_sharpe_ratio(0.1, 0.01, 1, 500, 0.0, 3.0))


def test_statistics_guards_reject_bad_inputs() -> None:
    """Sampling-variance and growth-LCB guards fail closed on bad inputs."""
    import pytest

    from src.common.errors import DataIntegrityError
    from src.evaluation.statistics import log_growth_lcb

    with pytest.raises(DataIntegrityError):
        sharpe_sampling_variance(float("nan"), 100, 0.0, 3.0)
    with pytest.raises(DataIntegrityError):
        sharpe_sampling_variance(0.05, 1, 0.0, 3.0)
    with pytest.raises(DataIntegrityError):
        sharpe_sampling_variance(1.0, 100, 0.0, -3.0)
    series = _fat_tail_drift(200, seed=3)
    with pytest.raises(DataIntegrityError):
        log_growth_lcb(series, alpha=1.5, n_paths=50, seed=1)
    with pytest.raises(DataIntegrityError):
        log_growth_lcb(series, alpha=0.05, n_paths=0, seed=1)
    with pytest.raises(DataIntegrityError):
        log_growth_lcb(series, alpha=0.05, n_paths=50, seed=-1)
    with pytest.raises(DataIntegrityError):
        log_growth_lcb(pd.Series([0.01, 0.02]), alpha=0.05, n_paths=50, seed=1)
    bad = series.copy()
    bad.iloc[5] = float("nan")
    with pytest.raises(DataIntegrityError):
        log_growth_lcb(bad, alpha=0.05, n_paths=50, seed=1)
    naive = pd.Series(np.ones(50) * 0.001, index=pd.date_range("2021-01-01", periods=50, freq="D"))
    with pytest.raises(DataIntegrityError):
        log_growth_lcb(naive, alpha=0.05, n_paths=50, seed=1)
    dup = pd.Series(
        np.ones(50) * 0.001,
        index=pd.DatetimeIndex([pd.Timestamp("2021-01-01", tz="UTC")] * 50),
    )
    with pytest.raises(DataIntegrityError):
        log_growth_lcb(dup, alpha=0.05, n_paths=50, seed=1)
    rev = _fat_tail_drift(50, seed=4).iloc[::-1]
    with pytest.raises(DataIntegrityError):
        log_growth_lcb(rev, alpha=0.05, n_paths=50, seed=1)
    ruin = _utc_series(np.full(50, 0.001))
    ruin.iloc[3] = -1.5
    with pytest.raises(DataIntegrityError):
        log_growth_lcb(ruin, alpha=0.05, n_paths=50, seed=1)
    wiped = _utc_series(np.full(50, 0.001))
    wiped.iloc[3] = -1.0
    with pytest.raises(DataIntegrityError):
        log_growth_lcb(wiped, alpha=0.05, n_paths=50, seed=1)
    with pytest.raises(DataIntegrityError):
        log_growth_lcb(pd.Series(dtype="float64"), alpha=0.05, n_paths=50, seed=1)
