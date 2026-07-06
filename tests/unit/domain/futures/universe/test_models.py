from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.domain.futures.universe.models import load_ledger_slice


def _write_parquet_ledger(path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "symbol": "BTC/USDT",
                "tf": "4h",
                "date": "2025-01-01",
                "knowledge_date": "2025-01-01",
                "is_listed": True,
                "is_trading": True,
                "status": "TRADING",
                "adv_usdt_median": 120_000_000.0,
            },
            {
                "symbol": "ETH/USDT",
                "tf": "4h",
                "date": "2025-01-01",
                "knowledge_date": "2025-01-01",
                "is_listed": True,
                "is_trading": True,
                "status": "TRADING",
                "adv_usdt_median": 80_000_000.0,
            },
            {
                "symbol": "XRP/USDT",
                "tf": "4h",
                "date": "2025-02-01",
                "knowledge_date": "2025-02-01",
                "is_listed": True,
                "is_trading": True,
                "status": "TRADING",
                "adv_usdt_median": 30_000_000.0,
            },
        ]
    )
    frame.to_parquet(path, index=False)


def test_load_ledger_slice_reads_parquet_and_applies_pit_filter(tmp_path: Path) -> None:
    ledger_path = tmp_path / "universe_ledger.parquet"
    _write_parquet_ledger(ledger_path)

    frame = load_ledger_slice(
        as_of="2025-01-01",
        tf="4h",
        columns=("status", "adv_usdt_median"),
        symbols=("BTC/USDT", "ETH/USDT"),
        ledger_path=ledger_path,
    )

    assert tuple(frame["symbol"].astype(str).tolist()) == ("BTC/USDT", "ETH/USDT")
    assert set(frame["status"].astype(str).tolist()) == {"TRADING"}
    assert float(frame.loc[frame["symbol"] == "BTC/USDT", "adv_usdt_median"].iloc[0]) == pytest.approx(120_000_000.0)


def test_load_ledger_slice_rejects_missing_required_parquet_columns(tmp_path: Path) -> None:
    ledger_path = tmp_path / "universe_ledger.parquet"
    pd.DataFrame(
        [
            {
                "symbol": "BTC/USDT",
                "tf": "4h",
                "date": "2025-01-01",
                "knowledge_date": "2025-01-01",
                "is_listed": True,
                "is_trading": True,
            }
        ]
    ).to_parquet(ledger_path, index=False)

    with pytest.raises(ValueError, match="Parquet ledger missing required columns"):
        load_ledger_slice(
            as_of="2025-01-01",
            tf="4h",
            columns=("status",),
            ledger_path=ledger_path,
        )


def test_load_ledger_slice_rejects_unsupported_backend_suffix(tmp_path: Path) -> None:
    ledger_path = tmp_path / "universe_ledger.csv"
    ledger_path.write_text("symbol,tf,date,knowledge_date\nBTC/USDT,4h,2025-01-01,2025-01-01\n")

    with pytest.raises(ValueError, match="unsupported ledger backend"):
        load_ledger_slice(
            as_of="2025-01-01",
            tf="4h",
            columns=("status",),
            ledger_path=ledger_path,
        )
