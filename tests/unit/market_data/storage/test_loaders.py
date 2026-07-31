from __future__ import annotations

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.market_data.storage.loaders import load_funding_rates


def test_load_funding_rates_rejects_missing_rate_column(tmp_path) -> None:
    path = tmp_path / "funding.parquet"
    pd.DataFrame({"datetime": ["2024-01-01T00:00:00Z"]}).to_parquet(path)

    with pytest.raises(DataIntegrityError, match="funding_rate"):
        load_funding_rates(path)


def test_load_funding_rates_accepts_timestamp_column(tmp_path) -> None:
    path = tmp_path / "funding.parquet"
    pd.DataFrame({"timestamp": [1704067200000], "funding_rate": [0.0001]}).to_parquet(path)

    result = load_funding_rates(path)

    assert result.index[0] == pd.Timestamp("2024-01-01", tz="UTC")


def test_load_funding_rates_rejects_missing_timestamp_columns(tmp_path) -> None:
    path = tmp_path / "funding.parquet"
    pd.DataFrame({"funding_rate": [0.0001]}).to_parquet(path)

    with pytest.raises(DataIntegrityError, match="timestamp"):
        load_funding_rates(path)


def test_load_funding_rates_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(DataIntegrityError, match="does not exist"):
        load_funding_rates(tmp_path / "missing.parquet")
