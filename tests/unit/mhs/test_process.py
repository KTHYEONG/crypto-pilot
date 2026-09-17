"""Invariant scenarios for the causal MHS process estimators."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.process import (
    ema_smoothing_rate,
    estimation_adjusted_kelly_exposure,
    estimation_adjusted_mean,
    ledoit_wolf_covariance,
    long_only_growth_weights,
    monthly_refit_schedule,
    select_smoothing_halflife,
    smoothed_book_path,
    step_proxy_net_returns,
)


def test_monthly_refit_schedule_first_point() -> None:
    schedule = monthly_refit_schedule(
        pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2022-06-30", tz="UTC")
    )
    assert schedule[0].effective_from == pd.Timestamp("2022-02-01", tz="UTC")
    assert schedule[0].train_end == schedule[0].effective_from - pd.Timedelta(hours=720)
    for prev, nxt in itertools.pairwise(schedule):
        assert nxt.effective_from == prev.effective_to
    assert schedule[-1].effective_to == pd.Timestamp("2022-07-01", tz="UTC")


def test_monthly_refit_schedule_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match=r".+"):
        monthly_refit_schedule(
            pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2021-03-01", tz="UTC")
        )
    with pytest.raises(ValueError, match=r".+"):
        monthly_refit_schedule(
            pd.Timestamp("2021-01-01"), pd.Timestamp("2022-06-30", tz="UTC")
        )


def test_estimation_adjusted_mean_shrinkage() -> None:
    rng = np.random.default_rng(0)
    noise = pd.DataFrame({"a": rng.normal(0, 1, 50)})
    strong = pd.DataFrame({"b": rng.normal(0.5, 1, 5000)})
    assert estimation_adjusted_mean(noise)["a"] == 0.0
    col = strong["b"].to_numpy()
    n, m, s = len(col), col.mean(), col.std(ddof=1)
    t2 = n * m * m / (s * s)
    assert estimation_adjusted_mean(strong)["b"] == pytest.approx(m * (1 - 1 / t2), rel=1e-12)
    assert estimation_adjusted_mean(pd.DataFrame({"c": np.ones(10)}))["c"] == 0.0
    with pytest.raises(ValueError, match=r".+"):
        estimation_adjusted_mean(pd.DataFrame({"d": [1.0, np.nan]}))


def test_ledoit_wolf_covariance_pd() -> None:
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.normal(0, 1, (200, 4)), columns=list("abcd"))
    cov = ledoit_wolf_covariance(returns)
    assert np.allclose(cov.to_numpy(), cov.to_numpy().T)
    assert np.all(np.linalg.eigvalsh(cov.to_numpy()) > 0)
    sample = returns.cov().to_numpy() * (len(returns) - 1) / len(returns)
    assert abs(np.trace(cov.to_numpy()) - np.trace(sample)) < 1e-8
    twin = returns.copy()
    twin["b"] = twin["a"]
    dup = ledoit_wolf_covariance(twin)
    assert np.all(np.linalg.eigvalsh(dup.to_numpy()) > 0)
    with pytest.raises(ValueError, match=r".+"):
        ledoit_wolf_covariance(returns.iloc[:1])


def test_long_only_growth_weights_cases() -> None:
    expected = pd.Series([-1.0, -0.5], index=["a", "b"])
    cov = pd.DataFrame(np.eye(2), index=["a", "b"], columns=["a", "b"])
    assert (long_only_growth_weights(expected, cov) == 0.0).all()
    mu = pd.Series([0.1, 0.2], index=["a", "b"])
    sigma = pd.DataFrame([[2.0, 0.5], [0.5, 1.0]], index=["a", "b"], columns=["a", "b"])
    weights = long_only_growth_weights(mu, sigma)
    assert np.allclose(weights.to_numpy(), np.linalg.solve(sigma.to_numpy(), mu.to_numpy()), atol=1e-6)
    twin_mu = pd.Series([0.1, 0.1], index=["a", "b"])
    twin_sigma = pd.DataFrame([[1.0, 1.0 - 1e-6], [1.0 - 1e-6, 1.0]], index=["a", "b"], columns=["a", "b"])
    twin = long_only_growth_weights(twin_mu, twin_sigma)
    assert bool((twin >= 0).all())
    single = long_only_growth_weights(
        pd.Series([0.1], index=["a"]), pd.DataFrame([[1.0]], index=["a"], columns=["a"])
    )
    assert twin.sum() == pytest.approx(single.sum(), rel=1e-3)
    with pytest.raises(ValueError, match=r".+"):
        long_only_growth_weights(mu, pd.DataFrame(np.eye(2), index=["a", "c"], columns=["a", "c"]))


def test_ema_smoothing_rate_values() -> None:
    assert ema_smoothing_rate(0.0) == 1.0
    assert ema_smoothing_rate(1.0) == 0.5
    with pytest.raises(ValueError, match=r".+"):
        ema_smoothing_rate(-1.0)


def test_smoothed_book_path_rates() -> None:
    index = pd.date_range("2022-01-01", periods=3, freq="24h", tz="UTC")
    targets = pd.DataFrame({"a": [0.0, 1.0, 1.0], "b": [0.0, -1.0, -1.0]}, index=index)
    full = smoothed_book_path(targets, pd.Series(1.0, index=index))
    assert np.allclose(full.to_numpy(), targets.to_numpy())
    half = smoothed_book_path(targets, pd.Series(0.5, index=index))
    assert half.iloc[0]["a"] == pytest.approx(0.0)
    stepped = pd.DataFrame({"a": [0.0, 1.0, 1.0]}, index=index)
    path = smoothed_book_path(stepped, pd.Series(0.5, index=index))
    assert path["a"].tolist() == pytest.approx([0.0, 0.5, 0.75])
    initial = pd.Series({"a": 2.0, "b": 1.0})
    half_from_initial = smoothed_book_path(
        targets.iloc[:1], pd.Series(0.5, index=index[:1]), initial
    ).iloc[0]
    assert half_from_initial["a"] == pytest.approx(1.0)
    assert half_from_initial["b"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match=r".+"):
        smoothed_book_path(targets, pd.Series(0.0, index=index))


def test_step_proxy_net_returns_hand_computed() -> None:
    index = pd.date_range("2022-01-01", periods=3, freq="24h", tz="UTC")
    weights = pd.DataFrame({"a": [0.5, 0.5, 0.5], "b": [-0.5, -0.5, -0.5]}, index=index)
    log_close = pd.DataFrame(
        {"a": np.log([100.0, 101.0, 102.0]), "b": np.log([50.0, 50.0, 49.0])}, index=index
    )
    funding = pd.DataFrame({"a": [0.001, 0.001, 0.001], "b": [0.0, 0.0, 0.0]}, index=index)
    rets = step_proxy_net_returns(weights, log_close, funding, 0.0)
    assert len(rets) == 2
    move_a = 101.0 / 100.0 - 1.0
    move_b = 50.0 / 50.0 - 1.0
    assert rets.iloc[0] == pytest.approx(0.5 * move_a - 0.5 * move_b - 0.5 * 0.001)
    nan_close = log_close.copy()
    nan_close.iloc[2, 0] = np.nan
    assert np.isfinite(step_proxy_net_returns(weights, nan_close, funding, 0.0)).all()
    with pytest.raises(ValueError, match=r".+"):
        step_proxy_net_returns(weights, log_close, funding, -1.0)


def test_select_smoothing_halflife_prefers_longer_on_whipsaw() -> None:
    index = pd.date_range("2022-01-01", periods=60, freq="24h", tz="UTC")
    whipsaw = pd.DataFrame({"a": [0.5 if i % 2 == 0 else -0.5 for i in range(60)]}, index=index)
    persistent = pd.DataFrame({"a": [0.5] * 60}, index=index)
    log_close = pd.DataFrame({"a": np.log(np.linspace(100, 160, 60))}, index=index)
    zero_fund = pd.DataFrame({"a": np.zeros(60)}, index=index)
    train_end = index[40] + pd.Timedelta(hours=12)
    whipsaw_pick = select_smoothing_halflife(whipsaw, log_close, zero_fund, 100.0, train_end)
    calm_pick = select_smoothing_halflife(persistent, log_close, zero_fund, 0.0, train_end)
    assert whipsaw_pick > calm_pick
    flat = pd.DataFrame({"a": np.zeros(60)}, index=index)
    assert select_smoothing_halflife(flat, log_close, zero_fund, 10.0, train_end) == 8.0
    with pytest.raises(ValueError, match=r".+"):
        select_smoothing_halflife(whipsaw, log_close, zero_fund, 10.0, index[0])


def test_select_smoothing_halflife_raises_on_ruin() -> None:
    index = pd.date_range("2022-01-01", periods=10, freq="24h", tz="UTC")
    targets = pd.DataFrame({"a": [1.0 if i % 2 == 0 else -1.0 for i in range(10)]}, index=index)
    log_close = pd.DataFrame({"a": np.zeros(10)}, index=index)
    funding = pd.DataFrame({"a": np.zeros(10)}, index=index)
    with pytest.raises(DataIntegrityError, match=r".+"):
        select_smoothing_halflife(targets, log_close, funding, 100000.0, index[6], ladder=(0.0,))


def test_estimation_adjusted_kelly_exposure_properties() -> None:
    index = pd.date_range("2022-01-01", periods=100, freq="24h", tz="UTC")
    active = index[10]
    drift = pd.Series(np.linspace(0.001, 0.01, 100), index=index)
    exposure = estimation_adjusted_kelly_exposure(drift, active_from=active, cap=2.0)
    assert (exposure.loc[index[:10]] == 0.0).all()
    assert bool(((exposure.loc[index[30:]] > 0) & (exposure.loc[index[30:]] <= 2.0)).all())
    assert (estimation_adjusted_kelly_exposure(
        pd.Series(-0.01, index=index), active_from=active, cap=2.0
    ) == 0.0).all()
    altered = drift.copy()
    altered.iloc[80:] = 5.0
    before = estimation_adjusted_kelly_exposure(drift, active_from=active, cap=2.0)
    after = estimation_adjusted_kelly_exposure(altered, active_from=active, cap=2.0)
    assert before.loc[index[50]] == after.loc[index[50]]
    with pytest.raises(ValueError, match=r".+"):
        estimation_adjusted_kelly_exposure(drift, active_from=pd.Timestamp("2022-01-01"), cap=2.0)


def test_monthly_refit_schedule_rejects_order_and_nonpositive() -> None:
    with pytest.raises(ValueError, match=r".+"):
        monthly_refit_schedule(
            pd.Timestamp("2022-06-30", tz="UTC"), pd.Timestamp("2022-06-30", tz="UTC")
        )
    with pytest.raises(ValueError, match=r".+"):
        monthly_refit_schedule(
            pd.Timestamp("2021-01-01", tz="UTC"),
            pd.Timestamp("2022-06-30", tz="UTC"),
            min_train_days=0,
        )
    with pytest.raises(ValueError, match=r".+"):
        monthly_refit_schedule(
            pd.Timestamp("2021-01-01", tz="UTC"),
            pd.Timestamp("2022-06-30", tz="UTC"),
            purge_hours=0,
        )


def test_estimation_adjusted_mean_edge_rows() -> None:
    single = pd.DataFrame({"a": [0.5]})
    assert estimation_adjusted_mean(single)["a"] == 0.0
    zero_mean = pd.DataFrame({"b": [1.0, -1.0]})
    assert estimation_adjusted_mean(zero_mean)["b"] == 0.0


def test_ledoit_wolf_rejects_nonfinite() -> None:
    bad = pd.DataFrame({"a": [1.0, np.nan], "b": [0.5, 0.25]})
    with pytest.raises(ValueError, match=r".+"):
        ledoit_wolf_covariance(bad)


def test_ema_smoothing_rate_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match=r".+"):
        ema_smoothing_rate(float("inf"))


def test_smoothed_book_path_rejects_misaligned() -> None:
    index = pd.date_range("2022-01-01", periods=2, freq="24h", tz="UTC")
    targets = pd.DataFrame({"a": [0.0, 1.0]}, index=index)
    with pytest.raises(ValueError, match=r".+"):
        smoothed_book_path(targets, pd.Series(0.5, index=index[::-1]))
    with pytest.raises(ValueError, match=r".+"):
        smoothed_book_path(
            targets, pd.Series(0.5, index=index), pd.Series({"b": 0.0})
        )


def test_step_proxy_net_returns_edge_cases() -> None:
    index = pd.date_range("2022-01-01", periods=2, freq="24h", tz="UTC")
    weights = pd.DataFrame({"a": [0.5, 0.5]}, index=index)
    log_close = pd.DataFrame({"a": [0.0, 0.01]}, index=index)
    funding = pd.DataFrame({"a": [0.0, 0.0]}, index=index)
    with pytest.raises(ValueError, match=r".+"):
        step_proxy_net_returns(
            weights, log_close.iloc[::-1], funding, 1.0
        )
    with pytest.raises(ValueError, match=r".+"):
        step_proxy_net_returns(
            weights, log_close.rename(columns={"a": "b"}), funding, 1.0
        )
    single_out = step_proxy_net_returns(
        weights.iloc[:1], log_close.iloc[:1], funding.iloc[:1], 1.0
    )
    assert single_out.empty


def test_select_smoothing_halflife_rejects_empty_ladder() -> None:
    index = pd.date_range("2022-01-01", periods=5, freq="24h", tz="UTC")
    targets = pd.DataFrame({"a": np.ones(5)}, index=index)
    log_close = pd.DataFrame({"a": np.zeros(5)}, index=index)
    funding = pd.DataFrame({"a": np.zeros(5)}, index=index)
    with pytest.raises(ValueError, match=r".+"):
        select_smoothing_halflife(targets, log_close, funding, 1.0, index[-1], ladder=())
    with pytest.raises(ValueError, match=r".+"):
        select_smoothing_halflife(
            targets.iloc[:1], log_close.iloc[:1], funding.iloc[:1], 1.0, index[-1]
        )


def test_estimation_adjusted_kelly_exposure_rejects_bad_inputs() -> None:
    index = pd.date_range("2022-01-01", periods=5, freq="24h", tz="UTC")
    good = pd.Series(0.01, index=index)
    active = pd.Timestamp("2022-01-01", tz="UTC")
    with pytest.raises(ValueError, match=r".+"):
        estimation_adjusted_kelly_exposure(good.iloc[::-1], active_from=active, cap=2.0)
    with pytest.raises(ValueError, match=r".+"):
        estimation_adjusted_kelly_exposure(
            pd.Series([1.0, np.nan, 1.0, 1.0, 1.0], index=index),
            active_from=active, cap=2.0,
        )
    with pytest.raises(ValueError, match=r".+"):
        estimation_adjusted_kelly_exposure(good, active_from=active, cap=0.0)
    with pytest.raises(ValueError, match=r".+"):
        estimation_adjusted_kelly_exposure(good, active_from=active, cap=2.0, ewma_lambda=1.0)


def test_estimation_adjusted_kelly_exposure_constant_history_is_zero() -> None:
    index = pd.date_range("2022-01-01", periods=10, freq="24h", tz="UTC")
    constant = pd.Series(0.005, index=index)
    out = estimation_adjusted_kelly_exposure(
        constant, active_from=index[0], cap=2.0
    )
    assert (out == 0.0).all()
