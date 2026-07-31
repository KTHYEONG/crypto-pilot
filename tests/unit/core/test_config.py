from __future__ import annotations

from src.core.config import funding_path, ohlcv_path


class TestCoreConfigPaths:
    def test_core_config_paths(self) -> None:
        assert ohlcv_path("BTC/USDT", "4h").name == "BTC_USDT.parquet"
        assert ohlcv_path("BTC/USDT", "4h").parts[-4:] == ("futures", "ohlcv", "4h", "BTC_USDT.parquet")
        assert funding_path("BTC/USDT").name == "BTC_USDT.parquet"
        assert funding_path("BTC/USDT").parts[-3:] == ("futures", "funding", "BTC_USDT.parquet")

    def test_ohlcv_path_contract_assertion(self) -> None:
        p = ohlcv_path("BTC/USDT", "4h")
        assert p.name == "BTC_USDT.parquet"
