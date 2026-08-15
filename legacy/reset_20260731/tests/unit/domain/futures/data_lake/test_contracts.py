from __future__ import annotations

from pathlib import Path

import pytest

from src.domain.futures.data_lake.contracts import DataLakeConfig, DatasetKind


class TestDataLakeConfig:
    def test_valid_config(self) -> None:
        config = DataLakeConfig(root=Path("/tmp/lake"))
        assert config.soft_cap_gib == 48
        assert config.hard_cap_gib == 64
        assert config.max_workers == 4

    def test_rejects_soft_ge_hard_cap(self) -> None:
        with pytest.raises(ValueError, match="soft.*<.*hard"):
            DataLakeConfig(root=Path("/tmp/lake"), soft_cap_gib=50, hard_cap_gib=40)

    def test_rejects_hard_cap_over_64(self) -> None:
        with pytest.raises(ValueError, match="hard"):
            DataLakeConfig(root=Path("/tmp/lake"), hard_cap_gib=65)

    def test_rejects_wrong_market(self) -> None:
        with pytest.raises(ValueError, match="market"):
            DataLakeConfig(root=Path("/tmp/lake"), market="cm")

    def test_rejects_wrong_quote_asset(self) -> None:
        with pytest.raises(ValueError, match="quote_asset"):
            DataLakeConfig(root=Path("/tmp/lake"), quote_asset="BUSD")


class TestDatasetKind:
    def test_values(self) -> None:
        assert DatasetKind.KLINES_1H.value == "klines_1h"
        assert DatasetKind.FUNDING_EVENT.value == "funding_event"
        assert DatasetKind.METRICS_5M.value == "metrics_5m"
