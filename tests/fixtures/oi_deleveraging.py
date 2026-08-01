from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def make_oi_market_data():
    """Factory building a validator-ready OIDeleveragingMarketData fixture.

    ``mark_return_24h`` defaults to all-NaN (a completed-24h series that is
    always a no-signal interval) and ``oi_change`` defaults to zeros, so callers
    opt into the short state explicitly. ``funding`` maps timestamps to rates;
    when None a zero-rate event is emitted on every bar.
    """
    from src.research.oi_deleveraging.contracts import OIDeleveragingMarketData

    def _build(
        *,
        symbol: str = "BTCUSDT",
        n_bars: int = 8,
        freq: str = "4h",
        start: str = "2024-01-01",
        opens: list[float] | None = None,
        closes: list[float] | None = None,
        mark_return_24h: list[float] | None = None,
        oi_change: list[float] | None = None,
        funding: dict[str, float] | None = None,
    ) -> OIDeleveragingMarketData:
        grid = pd.date_range(start, periods=n_bars, freq=freq, tz="UTC")
        period = grid[1] - grid[0]
        oa = np.full(n_bars, 100.0) if opens is None else np.asarray(opens, dtype=float)
        ca = np.full(n_bars, 100.0) if closes is None else np.asarray(closes, dtype=float)
        if len(oa) != n_bars or len(ca) != n_bars:
            raise ValueError("price arrays must match n_bars")
        high = np.maximum(oa, ca) + 1.0
        low = np.minimum(oa, ca) - 1.0

        mr = (
            np.full(n_bars, np.nan)
            if mark_return_24h is None
            else np.asarray(mark_return_24h, dtype=float)
        )
        oc = np.zeros(n_bars) if oi_change is None else np.asarray(oi_change, dtype=float)
        if len(mr) != n_bars or len(oc) != n_bars:
            raise ValueError("feature arrays must match n_bars")

        bars = pd.DataFrame(
            {"open": oa, "high": high, "low": low, "close": ca, "volume": 1000.0},
            index=grid,
        )
        decision_times = grid + period
        # A decision at 00:00 can only see the metric released at 00:05 of the
        # previous day, so the feature day is the latest day whose metric is
        # released before the decision time.
        feature_dt = (decision_times - pd.Timedelta(minutes=5)).normalize()
        joined = pd.DataFrame({
            "open": oa,
            "high": high,
            "low": low,
            "close": ca,
            "volume": 1000.0,
            "decision_time": decision_times,
            "mark_return_24h": mr,
            "feature_datetime": feature_dt,
            "feature_available_at": feature_dt + pd.Timedelta(minutes=5),
            "feature_sum_open_interest": 100.0,
            "feature_sum_open_interest_value": 1000.0,
            "feature_long_short_ratio": 1.0,
            "feature_top_trader_long_short_ratio": 1.0,
            "feature_sum_taker_long_short_vol_ratio": 1.0,
            "feature_oi_value_change": oc,
        })
        if funding is None:
            funding_series = pd.Series(0.0, index=grid, dtype=float)
        else:
            idx = pd.DatetimeIndex([pd.Timestamp(key, tz="UTC") for key in funding])
            funding_series = pd.Series(
                list(funding.values()), index=idx, dtype=float,
            ).sort_index()
        return OIDeleveragingMarketData(
            symbol=symbol, bars=bars, joined=joined, funding=funding_series,
        )

    return _build


@pytest.fixture
def make_oi_metrics_lake():
    """Factory writing a canonical metrics/OHLCV/funding lake under ``tmp_path``.

    Returns a dict of ``Path`` objects keyed by ``"ohlcv"``/``"funding"``/
    ``"metrics"`` and monkeypatches the OI market-data path helpers so the
    loader reads exactly these files.
    """

    def _build(
        tmp_path,
        monkeypatch,
        *,
        symbol: str = "BTCUSDT",
        n_bars: int = 4,
        start: str = "2024-01-01",
        metrics_frame: pd.DataFrame,
    ) -> dict[str, object]:
        import src.research.oi_deleveraging.market_data as md

        def _ohlcv(sym: str, timeframe: str):
            return tmp_path / "futures" / "ohlcv" / timeframe / f"{sym.replace('/', '_')}.parquet"

        def _funding(sym: str):
            return tmp_path / "futures" / "funding" / f"{sym.replace('/', '_')}.parquet"

        def _metrics(sym: str):
            return tmp_path / "futures" / "metrics" / "1d" / f"{sym.replace('/', '_')}.parquet"

        monkeypatch.setattr(md, "ohlcv_path", _ohlcv)
        monkeypatch.setattr(md, "funding_path", _funding)
        monkeypatch.setattr(md, "metrics_path", _metrics)

        hourly = pd.date_range(start, periods=n_bars * 4, freq="1h", tz="UTC")
        price = 100.0 + np.arange(len(hourly), dtype=np.float64)
        ohlcv_frame = pd.DataFrame({
            "timestamp": (hourly - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms"),
            "open": price, "high": price + 1.0, "low": price - 1.0,
            "close": price, "volume": 10.0,
        })
        ohlcv_path = _ohlcv(symbol, "1h")
        ohlcv_path.parent.mkdir(parents=True, exist_ok=True)
        ohlcv_frame.to_parquet(ohlcv_path, index=False)

        fund_path = _funding(symbol)
        fund_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "datetime": pd.date_range(start, periods=n_bars, freq="4h", tz="UTC"),
            "funding_rate": 0.0,
        }).to_parquet(fund_path, index=False)

        metrics_path = _metrics(symbol)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_frame.to_parquet(metrics_path, index=False)

        return {"ohlcv": ohlcv_path, "funding": fund_path, "metrics": metrics_path}

    return _build
