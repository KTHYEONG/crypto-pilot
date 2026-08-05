from __future__ import annotations

from src.common.config import (
    bookdepth_path,
    borrow_path,
    funding_path,
    indicator_kline_path,
    metrics_path,
    ohlcv_path,
    spot_ohlcv_path,
)


class TestCoreConfigPaths:
    def test_core_config_paths(self) -> None:
        assert ohlcv_path("BTC/USDT", "4h").name == "BTCUSDT.parquet"
        assert ohlcv_path("BTC/USDT", "4h").parts[-4:] == (
            "futures", "ohlcv", "4h", "BTCUSDT.parquet",
        )
        assert funding_path("BTC/USDT").name == "BTCUSDT.parquet"
        assert funding_path("BTC/USDT").parts[-3:] == (
            "futures", "funding", "BTCUSDT.parquet",
        )

    def test_ohlcv_path_contract_assertion(self) -> None:
        p = ohlcv_path("BTC/USDT", "4h")
        assert p.name == "BTCUSDT.parquet"

    def test_carry_data_paths(self) -> None:
        assert spot_ohlcv_path("BTC/USDT", "1h").parts[-4:] == (
            "spot", "ohlcv", "1h", "BTCUSDT.parquet",
        )
        assert borrow_path("BTC/USDT").parts[-3:] == ("spot", "borrow", "BTCUSDT.parquet")


class TestIndicatorKlinePath:
    def test_indicator_kline_path_scoped_by_dataset_symbol_timeframe(self) -> None:
        # XS-EXP-01 / SCENARIO_INDICATOR_KLINE_PATH_01: mark/index/premium klines
        # land under futures/<dataset>/<timeframe>/ with safe-symbol normalization.
        p = indicator_kline_path("premiumIndexKlines", "BTC/USDT", "4h")
        assert p.parts[-4:] == ("futures", "premiumIndexKlines", "4h", "BTCUSDT.parquet")
        assert indicator_kline_path("indexPriceKlines", "ETHUSDT", "1h").parts[-3] == "indexPriceKlines"

    def test_indicator_kline_path_contract_assertion(self) -> None:
        p = indicator_kline_path("premiumIndexKlines", "BTCUSDT", "4h")
        assert p.parts[-4:] == ("futures", "premiumIndexKlines", "4h", "BTCUSDT.parquet")


class TestBookdepthPath:
    def test_bookdepth_path_under_futures_bookdepth(self) -> None:
        p = bookdepth_path("BTC/USDT")
        assert p.parts[-3:] == ("futures", "bookdepth", "BTCUSDT.parquet")

    def test_bookdepth_path_contract_assertion(self) -> None:
        p = bookdepth_path("BTCUSDT")
        assert p.parts[-3:] == ("futures", "bookdepth", "BTCUSDT.parquet")


class TestMetricsPath:
    def test_metrics_path_uses_canonical_symbol(self) -> None:
        # FD-01: canonical futures metrics path, distinct from any OHLCV/funding path.
        assert str(metrics_path("BTC/USDT")).endswith("data/futures/metrics/1d/BTCUSDT.parquet")
        assert metrics_path("BTC/USDT").parts[-4:] == (
            "futures", "metrics", "1d", "BTCUSDT.parquet",
        )

    def test_every_fixed_universe_symbol_maps_distinctly(self) -> None:
        # FD-01: every fixed-universe symbol maps to a distinct canonical parquet path.
        symbols = ["BTCUSDT", "ETHUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT"]
        paths = [metrics_path(s) for s in symbols]
        assert len(set(paths)) == len(symbols)
        for p in paths:
            assert str(p).endswith(".parquet")
