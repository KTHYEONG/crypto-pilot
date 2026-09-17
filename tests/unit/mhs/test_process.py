"""Invariant scenarios for the causal MHS process estimators."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from src.mhs.process import (
    ema_smoothing_rate,
    estimation_adjusted_mean,
    ledoit_wolf_covariance,
    long_only_growth_weights,
    monthly_refit_schedule,
    smoothed_book_path,
    step_proxy_net_returns,
    volatility_scaled_exposure,
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


def test_volatility_scaled_exposure_constant_vol_at_cap() -> None:
    rng = np.random.default_rng(0)
    index = pd.date_range("2022-01-01", periods=60, freq="24h", tz="UTC")
    returns = pd.Series(rng.normal(0, 0.01, 60), index=index)
    exposure = volatility_scaled_exposure(returns, cap=3.0, halflife_days=5)
    assert (exposure.iloc[:5] == 0.0).all()
    assert bool(((exposure >= 0.0) & (exposure <= 3.0)).all())
    assert exposure.index.equals(index)
    vol = returns.ewm(halflife=5, min_periods=5).std().shift(1)
    median = vol.expanding().median()
    expected = (3.0 * median / vol.where(vol > 0)).clip(lower=0.0, upper=3.0).fillna(0.0)
    assert np.allclose(exposure.to_numpy(), expected.to_numpy(), atol=1e-12)
    steady = exposure.loc[vol == median]
    if len(steady):
        assert np.allclose(steady.to_numpy(), 3.0, atol=1e-9)
    alternating = pd.Series(np.where(np.arange(60) % 2 == 0, 0.01, -0.01), index=index)
    alt_exposure = volatility_scaled_exposure(alternating, cap=3.0, halflife_days=5)
    assert np.allclose(alt_exposure.iloc[10:].to_numpy(), 3.0, atol=1e-9)


def test_volatility_scaled_exposure_derisks_when_vol_triples() -> None:
    rng = np.random.default_rng(1)
    index = pd.date_range("2022-01-01", periods=80, freq="24h", tz="UTC")
    calm = rng.normal(0, 0.01, 60)
    storm = rng.normal(0, 0.03, 20)
    returns = pd.Series(np.concatenate([calm, storm]), index=index)
    exposure = volatility_scaled_exposure(returns, cap=3.0, halflife_days=5)
    tail = exposure.iloc[-10:]
    assert bool(((tail < 3.0) & (tail > 0.0)).all())


def test_volatility_scaled_exposure_all_zero_is_zero() -> None:
    index = pd.date_range("2022-01-01", periods=30, freq="24h", tz="UTC")
    exposure = volatility_scaled_exposure(pd.Series(0.0, index=index), cap=3.0, halflife_days=5)
    assert (exposure == 0.0).all()


def test_volatility_scaled_exposure_causal() -> None:
    rng = np.random.default_rng(2)
    index = pd.date_range("2022-01-01", periods=60, freq="24h", tz="UTC")
    returns = pd.Series(rng.normal(0, 0.01, 60), index=index)
    shocked = returns.copy()
    shocked.iloc[30:] = 0.5
    before = volatility_scaled_exposure(returns, cap=3.0, halflife_days=5)
    after = volatility_scaled_exposure(shocked, cap=3.0, halflife_days=5)
    assert np.allclose(before.iloc[:31].to_numpy(), after.iloc[:31].to_numpy())


def test_volatility_scaled_exposure_rejects_bad_inputs() -> None:
    index = pd.date_range("2022-01-01", periods=10, freq="24h", tz="UTC")
    good = pd.Series(0.01, index=index)
    with pytest.raises(ValueError, match=r".+"):
        volatility_scaled_exposure(good, cap=0.0, halflife_days=5)
    with pytest.raises(ValueError, match=r".+"):
        volatility_scaled_exposure(good, cap=-1.0, halflife_days=5)
    with pytest.raises(ValueError, match=r".+"):
        volatility_scaled_exposure(good, cap=2.0, halflife_days=0)
    with pytest.raises(ValueError, match=r".+"):
        volatility_scaled_exposure(good.iloc[::-1], cap=2.0, halflife_days=5)
    with pytest.raises(ValueError, match=r".+"):
        volatility_scaled_exposure(
            pd.Series([1.0, np.nan] + [1.0] * 8, index=index), cap=2.0, halflife_days=5
        )


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


def _policy_frame(rows: list[list[float]]) -> pd.DataFrame:
    index = pd.date_range("2022-01-01", periods=len(rows), freq="24h", tz="UTC")
    return pd.DataFrame(rows, index=index, columns=["a", "b"])


def test_policy_config_boundary() -> None:
    from src.mhs.process import ProcessExecutionPolicy

    assert ProcessExecutionPolicy(None).tracking_error_threshold is None
    assert ProcessExecutionPolicy(0.0).tracking_error_threshold == 0.0
    assert ProcessExecutionPolicy(0.10).tracking_error_threshold == 0.10
    assert ProcessExecutionPolicy(0.20).tracking_error_threshold == 0.20
    with pytest.raises(ValueError, match=r".+"):
        ProcessExecutionPolicy(-0.1)
    with pytest.raises(ValueError, match=r".+"):
        ProcessExecutionPolicy(float("nan"))
    with pytest.raises(ValueError, match=r".+"):
        ProcessExecutionPolicy(float("inf"))
    with pytest.raises(ValueError, match=r".+"):
        ProcessExecutionPolicy(float("-inf"))
    with pytest.raises(ValueError, match=r".+"):
        ProcessExecutionPolicy("not-a-number")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r".+"):
        ProcessExecutionPolicy("0.2")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r".+"):
        ProcessExecutionPolicy(True)  # type: ignore[arg-type]


def test_policy_identity_and_empty() -> None:
    from src.mhs.process import ProcessExecutionPolicy, apply_process_execution_policy

    frame = _policy_frame([[0.5, -0.5], [0.25, -0.25]])
    for policy in (ProcessExecutionPolicy(None), ProcessExecutionPolicy(0.0)):
        out = apply_process_execution_policy(frame, policy)
        assert out.values.tolist() == frame.values.tolist()
        assert list(out.columns) == ["a", "b"]
        assert str(out.dtypes.iloc[0]) == "float64"
        out.iloc[0, 0] = 999.0
        assert frame.iloc[0, 0] != 999.0
    empty = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"), columns=["a", "b"])
    out_empty = apply_process_execution_policy(empty, ProcessExecutionPolicy(None))
    assert out_empty.shape == (0, 2)
    fully_empty = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    out_fully = apply_process_execution_policy(fully_empty, ProcessExecutionPolicy(None))
    assert out_fully.shape == (0, 0)


def test_policy_adoption_boundary() -> None:
    from src.mhs.process import ProcessExecutionPolicy, apply_process_execution_policy

    frame = _policy_frame([[0.5, -0.5], [0.5625, -0.5625], [0.625, -0.625]])
    out = apply_process_execution_policy(frame, ProcessExecutionPolicy(0.25))
    assert out.values.tolist() == [[0.5, -0.5], [0.5, -0.5], [0.625, -0.625]]


def test_policy_rejects_labels_and_anomalies() -> None:
    from src.mhs.process import ProcessExecutionPolicy, apply_process_execution_policy

    good = _policy_frame([[0.5, -0.5], [0.5, -0.5]])
    policies = (ProcessExecutionPolicy(None), ProcessExecutionPolicy(0.25))
    bad_index = pd.DataFrame([[0.5]], index=[0], columns=["a"])
    dup_index = pd.DataFrame(
        [[0.5, -0.5], [0.5, -0.5]],
        index=[good.index[0], good.index[0]],
        columns=["a", "b"],
    )
    unordered = good.iloc[::-1]
    naive = pd.DataFrame(
        [[0.5, -0.5]], index=pd.DatetimeIndex(["2022-01-01"]), columns=["a", "b"]
    )
    non_utc = pd.DataFrame(
        [[0.5, -0.5]],
        index=pd.DatetimeIndex(["2022-01-01"], tz="America/New_York"),
        columns=["a", "b"],
    )
    nat_index = good.copy()
    nat_index.index = pd.DatetimeIndex([pd.NaT, good.index[1]])
    dup_cols = pd.DataFrame(
        [[0.5, 0.5]], index=good.index[:1], columns=["a", "a"]
    )
    zero_cols = pd.DataFrame(index=good.index)
    non_numeric = pd.DataFrame(
        [["x", "y"], ["z", "w"]], index=good.index, columns=["a", "b"]
    )
    non_finite = good.copy()
    non_finite.iloc[0, 0] = float("nan")
    inf_frame = good.copy()
    inf_frame.iloc[0, 0] = float("inf")
    cases = [
        bad_index,
        dup_index,
        unordered,
        naive,
        non_utc,
        nat_index,
        dup_cols,
        zero_cols,
        non_numeric,
        non_finite,
        inf_frame,
    ]
    for bad in cases:
        for policy in policies:
            with pytest.raises(ValueError, match=r".+"):
                apply_process_execution_policy(bad, policy)
