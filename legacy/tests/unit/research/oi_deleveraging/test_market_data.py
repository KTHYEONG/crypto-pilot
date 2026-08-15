from __future__ import annotations

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.research.oi_deleveraging.market_data import (
    load_metrics_asof,
    load_oi_deleveraging_market_data,
    validate_oi_deleveraging_market_data,
)


def _bars() -> pd.DataFrame:
    grid = pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")
    return pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0},
        index=grid,
    )


def _canonical_metrics() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": [1704067200000, 1704153600000],
        "datetime": [
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-02", tz="UTC"),
        ],
        "available_at": [
            pd.Timestamp("2024-01-01 00:05", tz="UTC"),
            pd.Timestamp("2024-01-02 00:05", tz="UTC"),
        ],
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "sum_open_interest": [100.0, 200.0],
        "sum_open_interest_value": [500.0, 1000.0],
        "long_short_ratio": [1.0, 1.0],
        "top_trader_long_short_ratio": [1.0, 1.0],
        "sum_taker_long_short_vol_ratio": [1.0, 1.0],
    })


def _overnight_bars() -> pd.DataFrame:
    grid = pd.date_range("2024-01-01 20:00", periods=4, freq="4h", tz="UTC")
    return pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0},
        index=grid,
    )


class TestLoadMetricsAsof:
    def test_load_metrics_asof_rejects_future_feature(
        self, make_oi_metrics_lake, tmp_path, monkeypatch,
    ) -> None:
        # FD-03: a metric released after a 4h decision is never visible; the
        # decision selects only the prior available daily metric. The Jan-02
        # 00:00 decision (released 00:05) must resolve to the Jan-01 metric.
        make_oi_metrics_lake(tmp_path, monkeypatch, n_bars=4, metrics_frame=_canonical_metrics())
        joined = load_metrics_asof("BTCUSDT", _overnight_bars(), "2024-01-01", "2024-01-02")

        assert "decision_time" in joined.columns
        assert "feature_datetime" in joined.columns
        assert "feature_available_at" in joined.columns
        assert "mark_return_24h" in joined.columns

        mask = joined["feature_available_at"].notna()
        assert (
            joined.loc[mask, "feature_available_at"] <= joined.loc[mask, "decision_time"]
        ).all()

        midnight = joined[joined["decision_time"] == pd.Timestamp("2024-01-02 00:00", tz="UTC")]
        assert set(midnight["feature_datetime"]) == {pd.Timestamp("2024-01-01", tz="UTC")}
        later = joined[joined["decision_time"] >= pd.Timestamp("2024-01-02 04:00", tz="UTC")]
        assert set(later["feature_datetime"]) == {pd.Timestamp("2024-01-02", tz="UTC")}
        assert (later["feature_oi_value_change"] == 500.0).all()
        assert midnight["feature_oi_value_change"].isna().all()

    def test_load_metrics_asof_missing_metric_is_no_signal(
        self, make_oi_metrics_lake, tmp_path, monkeypatch,
    ) -> None:
        # FD-03: a missing metric leaves a no-signal interval, not an imputed feature.
        metrics = _canonical_metrics().iloc[[1]].reset_index(drop=True)
        make_oi_metrics_lake(tmp_path, monkeypatch, n_bars=4, metrics_frame=metrics)
        joined = load_metrics_asof("BTCUSDT", _overnight_bars(), "2024-01-01", "2024-01-02")

        midnight = joined[joined["decision_time"] == pd.Timestamp("2024-01-02 00:00", tz="UTC")]
        assert midnight["feature_available_at"].isna().all()
        assert midnight["feature_oi_value_change"].isna().all()

    def test_load_metrics_asof_fails_closed_on_non_monotonic(
        self, make_oi_metrics_lake, tmp_path, monkeypatch,
    ) -> None:
        # FD-05: non-monotonic metrics input fails closed before any backtest.
        unsorted = _canonical_metrics().iloc[::-1].reset_index(drop=True)
        make_oi_metrics_lake(tmp_path, monkeypatch, n_bars=4, metrics_frame=unsorted)
        with pytest.raises(DataIntegrityError, match="monotonic"):
            load_metrics_asof("BTCUSDT", _bars(), "2024-01-01", "2024-01-01 23:59:59")

    def test_load_metrics_asof_fails_closed_on_missing_column(
        self, make_oi_metrics_lake, tmp_path, monkeypatch,
    ) -> None:
        # FD-05: a metrics frame missing a required canonical column is rejected.
        broken = _canonical_metrics().drop(columns=["sum_open_interest_value"])
        make_oi_metrics_lake(tmp_path, monkeypatch, n_bars=4, metrics_frame=broken)
        with pytest.raises(DataIntegrityError, match="missing canonical columns"):
            load_metrics_asof("BTCUSDT", _bars(), "2024-01-01", "2024-01-01 23:59:59")


class TestLoadOIDeleveragingMarketData:
    def test_loads_and_validates_causal_market_data(
        self, make_oi_metrics_lake, tmp_path, monkeypatch,
    ) -> None:
        # FD-03: end-to-end load joins metrics as-of, retains audit timestamps,
        # and the assembled market data passes the fail-closed validator.
        make_oi_metrics_lake(tmp_path, monkeypatch, n_bars=4, metrics_frame=_canonical_metrics())
        data = load_oi_deleveraging_market_data("BTCUSDT", "2024-01-01", "2024-01-01 23:59:59")
        validate_oi_deleveraging_market_data(data)
        assert data.symbol == "BTCUSDT"
        assert len(data.joined) == len(data.bars)
        assert data.bars.index.is_monotonic_increasing

    def test_missing_metrics_file_raises(
        self, make_oi_metrics_lake, tmp_path, monkeypatch,
    ) -> None:
        metrics = _canonical_metrics()
        lake = make_oi_metrics_lake(tmp_path, monkeypatch, n_bars=4, metrics_frame=metrics)
        lake["metrics"].unlink()
        with pytest.raises(DataIntegrityError, match="metrics data missing"):
            load_oi_deleveraging_market_data("BTCUSDT", "2024-01-01", "2024-01-01 23:59:59")

    def test_missing_bars_file_raises(
        self, make_oi_metrics_lake, tmp_path, monkeypatch,
    ) -> None:
        lake = make_oi_metrics_lake(tmp_path, monkeypatch, n_bars=4, metrics_frame=_canonical_metrics())
        lake["ohlcv"].unlink()
        with pytest.raises(DataIntegrityError, match="perp_ohlcv data missing"):
            load_oi_deleveraging_market_data("BTCUSDT", "2024-01-01", "2024-01-01 23:59:59")
