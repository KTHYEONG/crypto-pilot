from __future__ import annotations

from src.common.config import borrow_path, funding_path, ohlcv_path, spot_ohlcv_path


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
