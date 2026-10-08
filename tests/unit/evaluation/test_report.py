"""Decision-grade statistics of one daily return path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.evaluation.report import strategy_statistics

_CUTOFF = pd.Timestamp("2026-07-01T00:00:00Z")


def test_initial_loss_and_liquidation_drawdown() -> None:
    for values, expected in (([-0.2, 0.1], 0.2), ([-1.0, 0.0], 1.0)):
        index = pd.date_range("2025-12-31", periods=2, freq="D", tz="UTC")
        returns = pd.Series(values, index=index)
        stats = strategy_statistics(returns, daily_funding_share=returns * 0, funding_by_symbol={}, initial_equity=100, design_data_cutoff=_CUTOFF, seed=1, n_paths=8)
        assert stats.max_drawdown == pytest.approx(expected)
        if expected == 1:
            assert stats.cagr == -1
            assert np.isnan(stats.years[1].funding_share)


@pytest.mark.parametrize(("values", "index"), [
    ([-1.01, 0], pd.date_range("2025-01-01", periods=2, tz="UTC")),
    ([0, 0], pd.date_range("2025-01-01", periods=2)),
    ([0, 0], pd.DatetimeIndex(["2025-01-01", "2025-01-01"], tz="UTC")),
    ([0, 0], pd.date_range("2025-01-01", periods=2, tz="UTC")[::-1]),
])
def test_invalid_chronology_and_negative_wealth_fail(values, index) -> None:
    returns = pd.Series(values, index=index)
    with pytest.raises(DataIntegrityError):
        strategy_statistics(returns, daily_funding_share=returns * 0, funding_by_symbol={}, initial_equity=100, design_data_cutoff=_CUTOFF, seed=1, n_paths=8)


def test_tiny_downside_does_not_emit_infinite_ratio() -> None:
    returns = pd.Series([-1e-14, -1e-14], index=pd.date_range("2025-01-01", periods=2, tz="UTC"))
    stats = strategy_statistics(returns, daily_funding_share=returns * 0, funding_by_symbol={}, initial_equity=100, design_data_cutoff=_CUTOFF, seed=1, n_paths=8)
    assert np.isnan(stats.sortino)


def _path(n: int = 730, rate: float = 0.001) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(np.full(n, rate, dtype="float64"), index=idx, dtype="float64")


def _funding_zero(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(np.zeros(len(index), dtype="float64"), index=index, dtype="float64")


def test_known_path_statistics() -> None:
    """A constant +0.1% path compounds exactly, never draws down, and has no Sharpe."""
    returns = _path()
    stats = strategy_statistics(
        returns, daily_funding_share=_funding_zero(returns.index), funding_by_symbol={},
        initial_equity=100000.0, design_data_cutoff=_CUTOFF, seed=1, n_paths=64,
    )
    assert stats.cagr == pytest.approx(1.001**365 - 1.0)
    assert stats.max_drawdown == pytest.approx(0.0)
    assert np.isnan(stats.sharpe)


def test_bootstrap_determinism() -> None:
    """A fixed seed reproduces quantiles bit-identically; a new seed moves them."""
    rng = np.random.default_rng(7)
    returns = pd.Series(
        rng.normal(0.0005, 0.01, size=730).astype("float64"),
        index=pd.date_range("2024-01-01", periods=730, freq="D", tz="UTC"), dtype="float64",
    )
    first = strategy_statistics(
        returns, daily_funding_share=_funding_zero(returns.index), funding_by_symbol={},
        initial_equity=100000.0, design_data_cutoff=_CUTOFF, seed=11, n_paths=128,
    )
    second = strategy_statistics(
        returns, daily_funding_share=_funding_zero(returns.index), funding_by_symbol={},
        initial_equity=100000.0, design_data_cutoff=_CUTOFF, seed=11, n_paths=128,
    )
    assert first.bootstrap.cagr_q50 == second.bootstrap.cagr_q50
    third = strategy_statistics(
        returns, daily_funding_share=_funding_zero(returns.index), funding_by_symbol={},
        initial_equity=100000.0, design_data_cutoff=_CUTOFF, seed=12, n_paths=128,
    )
    assert third.bootstrap.cagr_q50 != first.bootstrap.cagr_q50


def test_bootstrap_quantile_ordering() -> None:
    """Bootstrap quantiles never invert."""
    returns = _path()
    stats = strategy_statistics(
        returns, daily_funding_share=_funding_zero(returns.index), funding_by_symbol={},
        initial_equity=100000.0, design_data_cutoff=_CUTOFF, seed=3, n_paths=64,
    )
    assert stats.bootstrap.cagr_q05 <= stats.bootstrap.cagr_q50 <= stats.bootstrap.cagr_q95


def test_funding_concentration() -> None:
    """One dominant funding symbol tops the income ranking with exact shares and peak day."""
    n = 365
    idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    returns = pd.Series(np.zeros(n, dtype="float64"), index=idx, dtype="float64")
    funding = pd.Series(np.zeros(n, dtype="float64"), index=idx, dtype="float64")
    funding.iloc[100] = -0.02
    by_symbol = {"AAA": -2000.0, "BBB": -500.0, "CCC": 100.0}
    stats = strategy_statistics(
        returns, daily_funding_share=funding, funding_by_symbol=by_symbol,
        initial_equity=100000.0, design_data_cutoff=_CUTOFF, seed=5, n_paths=32,
    )
    assert stats.funding.total_contribution == pytest.approx(-2400.0 / 100000.0)
    assert [sym for sym, _ in stats.funding.top5_symbols][:2] == ["AAA", "BBB"]
    assert stats.funding.top5_share == pytest.approx(2500.0 / 2500.0)
    assert stats.funding.max_day == idx[100]
    assert stats.funding.max_day_share == pytest.approx(0.02)


def test_in_out_of_sample_split() -> None:
    """Days at or before the cutoff count as in-sample."""
    idx = pd.date_range("2026-06-28", periods=6, freq="D", tz="UTC")
    returns = pd.Series(np.full(6, 0.001, dtype="float64"), index=idx, dtype="float64")
    stats = strategy_statistics(
        returns, daily_funding_share=_funding_zero(idx), funding_by_symbol={},
        initial_equity=100000.0, design_data_cutoff=_CUTOFF, seed=9, n_paths=16,
    )
    assert stats.in_sample_days == 4
    assert stats.out_of_sample_days == 2


def test_invalid_returns_fail_closed() -> None:
    """A NaN day rejects the whole path."""
    idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    values = np.full(10, 0.001, dtype="float64")
    values[4] = np.nan
    returns = pd.Series(values, index=idx, dtype="float64")
    with pytest.raises(DataIntegrityError):
        strategy_statistics(
            returns, daily_funding_share=_funding_zero(idx), funding_by_symbol={},
            initial_equity=100000.0, design_data_cutoff=_CUTOFF, seed=1, n_paths=16,
        )
    with pytest.raises(DataIntegrityError):
        strategy_statistics(
            pd.Series([], dtype="float64"),
            daily_funding_share=pd.Series([], dtype="float64"), funding_by_symbol={},
            initial_equity=100000.0, design_data_cutoff=_CUTOFF, seed=1, n_paths=16,
        )


def test_short_year_reports_nan_sharpe() -> None:
    """A one-day year reports NaN Sharpe instead of raising."""
    idx = pd.DatetimeIndex([pd.Timestamp("2024-03-05", tz="UTC"), pd.Timestamp("2025-06-07", tz="UTC")])
    returns = pd.Series([0.001, 0.002], index=idx, dtype="float64")
    stats = strategy_statistics(
        returns, daily_funding_share=_funding_zero(idx), funding_by_symbol={},
        initial_equity=100000.0, design_data_cutoff=_CUTOFF, seed=1, n_paths=16,
    )
    assert all(year.days == 1 for year in stats.years)
    assert all(np.isnan(year.sharpe) for year in stats.years)


def test_sortino_degenerate_paths_report_nan() -> None:
    """Sortino uses all-day downside RMS and allows constant negative observations."""
    single = pd.Series(
        [0.001], index=pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")]), dtype="float64",
    )
    stats = strategy_statistics(
        single, daily_funding_share=_funding_zero(single.index), funding_by_symbol={},
        initial_equity=100000.0, design_data_cutoff=_CUTOFF, seed=1, n_paths=8,
    )
    assert np.isnan(stats.sortino)
    two = pd.Series(
        [0.001, -0.0005],
        index=pd.DatetimeIndex(
            [pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC")]
        ),
        dtype="float64",
    )
    stats = strategy_statistics(
        two, daily_funding_share=_funding_zero(two.index), funding_by_symbol={},
        initial_equity=100000.0, design_data_cutoff=_CUTOFF, seed=1, n_paths=8,
    )
    assert stats.sortino == pytest.approx(two.mean() / (0.0005 / np.sqrt(2)) * np.sqrt(365))
    flat_downside = pd.Series(
        [0.05, -0.001, -0.001],
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC"),
                pd.Timestamp("2024-01-03", tz="UTC"),
            ]
        ),
        dtype="float64",
    )
    stats = strategy_statistics(
        flat_downside, daily_funding_share=_funding_zero(flat_downside.index), funding_by_symbol={},
        initial_equity=100000.0, design_data_cutoff=_CUTOFF, seed=1, n_paths=8,
    )
    expected = flat_downside.mean() / np.sqrt((0.001**2 + 0.001**2) / 3) * np.sqrt(365)
    assert stats.sortino == pytest.approx(expected)


def test_statistics_payload_is_json_safe() -> None:
    """Non-finite statistics serialize to null with ISO-8601 timestamps."""
    import json

    from src.evaluation.report import statistics_payload

    returns = _path()
    stats = strategy_statistics(
        returns, daily_funding_share=_funding_zero(returns.index), funding_by_symbol={},
        initial_equity=100000.0, design_data_cutoff=_CUTOFF, seed=1, n_paths=16,
    )
    payload = statistics_payload(stats)
    assert payload["sharpe"] is None
    assert payload["funding"]["max_day"] == stats.funding.max_day.isoformat()
    json.dumps(payload)


def test_statistics_input_validation_branches() -> None:
    """Every malformed statistics input fails closed with DataIntegrityError."""
    idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    good = pd.Series(np.full(10, 0.001, dtype="float64"), index=idx, dtype="float64")
    zero = _funding_zero(idx)
    with pytest.raises(DataIntegrityError):
        strategy_statistics(
            [0.001] * 10, daily_funding_share=zero, funding_by_symbol={},  # type: ignore[arg-type]
            initial_equity=100000.0, design_data_cutoff=_CUTOFF, seed=1, n_paths=8,
        )
    with pytest.raises(DataIntegrityError):
        strategy_statistics(
            pd.Series(np.full(10, 0.001), index=list(range(10))),  # type: ignore[arg-type]
            daily_funding_share=zero, funding_by_symbol={},
            initial_equity=100000.0, design_data_cutoff=_CUTOFF, seed=1, n_paths=8,
        )
    with pytest.raises(DataIntegrityError):
        strategy_statistics(
            good, daily_funding_share=zero.iloc[:9], funding_by_symbol={},
            initial_equity=100000.0, design_data_cutoff=_CUTOFF, seed=1, n_paths=8,
        )
    shifted = zero.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=12)
    with pytest.raises(DataIntegrityError):
        strategy_statistics(
            good, daily_funding_share=shifted, funding_by_symbol={},
            initial_equity=100000.0, design_data_cutoff=_CUTOFF, seed=1, n_paths=8,
        )
    bad_share = zero.copy()
    bad_share.iloc[0] = np.nan
    with pytest.raises(DataIntegrityError):
        strategy_statistics(
            good, daily_funding_share=bad_share, funding_by_symbol={},
            initial_equity=100000.0, design_data_cutoff=_CUTOFF, seed=1, n_paths=8,
        )
    with pytest.raises(DataIntegrityError):
        strategy_statistics(
            good, daily_funding_share=zero, funding_by_symbol=[("AAA", 1.0)],  # type: ignore[arg-type]
            initial_equity=100000.0, design_data_cutoff=_CUTOFF, seed=1, n_paths=8,
        )
    with pytest.raises(DataIntegrityError):
        strategy_statistics(
            good, daily_funding_share=zero, funding_by_symbol={"AAA": np.nan},
            initial_equity=100000.0, design_data_cutoff=_CUTOFF, seed=1, n_paths=8,
        )
    for bad_equity in (0.0, -5.0, float("nan"), True):
        with pytest.raises(DataIntegrityError):
            strategy_statistics(
                good, daily_funding_share=zero, funding_by_symbol={},
                initial_equity=bad_equity, design_data_cutoff=_CUTOFF, seed=1, n_paths=8,  # type: ignore[arg-type]
            )
    with pytest.raises(DataIntegrityError):
        strategy_statistics(
            good, daily_funding_share=zero, funding_by_symbol={},
            initial_equity=100000.0, design_data_cutoff=pd.NaT, seed=1, n_paths=8,  # type: ignore[arg-type]
        )
    with pytest.raises(DataIntegrityError):
        strategy_statistics(
            good, daily_funding_share=zero, funding_by_symbol={},
            initial_equity=100000.0, design_data_cutoff=pd.Timestamp("2026-07-01"), seed=1, n_paths=8,  # type: ignore[arg-type]
        )
    with pytest.raises(DataIntegrityError):
        strategy_statistics(
            good, daily_funding_share=zero, funding_by_symbol={},
            initial_equity=100000.0, design_data_cutoff=_CUTOFF, seed=True, n_paths=8,  # type: ignore[arg-type]
        )
    with pytest.raises(DataIntegrityError):
        strategy_statistics(
            good, daily_funding_share=zero, funding_by_symbol={},
            initial_equity=100000.0, design_data_cutoff=_CUTOFF, seed=1, n_paths=0,
        )
