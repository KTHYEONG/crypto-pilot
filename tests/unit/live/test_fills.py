import pytest
import pandas as pd
from decimal import Decimal

from src.live.fills import FillEvent, append_fills, load_fills


def test_SCENARIO_PARITY_03_fill_schema_parity(tmp_path):
    """SCENARIO_PARITY_03-fill-schema-parity"""
    fills_dir = tmp_path / "fills"
    fills_dir.mkdir()
    ts = pd.Timestamp("2026-01-15 00:00:00", tz="UTC")
    ev = FillEvent(
        decision_time=ts,
        timestamp=ts,
        symbol="BTCUSDT",
        quantity_delta=Decimal("1.0"),
        fill_price=Decimal("100.0"),
        fee_bps=2.0,
        reason="maker_fill",
        pre_trade_equity=Decimal("2000"),
        liquidity="maker",
        mode="paper",
        run_id="20260115",
        leg_index=0,
        client_order_id="20260115-BTCUSDT-0-0-0",
        decision_mark=Decimal("100"),
        sizing_anchor="book_mid",
    )
    append_fills([ev], fills_dir)
    df = load_fills(fills_dir)
    for col in ["timestamp", "symbol", "quantity_delta", "fill_price", "fee_bps", "reason", "pre_trade_equity"]:
        assert col in df.columns
    assert str(df["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert df["quantity_delta"].dtype == "float64"
    assert df["fill_price"].dtype == "float64"
    assert df["fee_bps"].dtype == "float64"
    assert df["pre_trade_equity"].dtype == "float64"
    # invalid reason should raise
    bad = FillEvent(
        decision_time=ts,
        timestamp=ts,
        symbol="BTCUSDT",
        quantity_delta=Decimal("1.0"),
        fill_price=Decimal("100.0"),
        fee_bps=2.0,
        reason="invalid_reason",
        pre_trade_equity=Decimal("2000"),
        liquidity="maker",
        mode="paper",
        run_id="20260115",
        leg_index=0,
        client_order_id="bad",
    )
    try:
        append_fills([bad], fills_dir)
        pytest.fail("should have raised ValueError")
    except ValueError:
        pass


def test_SCENARIO_PARITY_04_monthly_partition_no_prune(tmp_path):
    """SCENARIO_PARITY_04-monthly-partition-no-prune"""
    fills_dir = tmp_path / "fills2"
    fills_dir.mkdir()
    total = 0
    for i in range(18):
        year = 2026 + (i // 12)
        month = 1 + (i % 12)
        ts = pd.Timestamp(f"{year}-{month:02d}-15 00:00:00", tz="UTC")
        ev = FillEvent(
            decision_time=ts,
            timestamp=ts,
            symbol="BTCUSDT",
            quantity_delta=Decimal("1.0"),
            fill_price=Decimal("100.0"),
            fee_bps=2.0,
            reason="maker_fill",
            pre_trade_equity=Decimal("2000"),
            liquidity="maker",
            mode="paper",
            run_id=ts.strftime("%Y%m%d"),
            leg_index=0,
            client_order_id=f"id-{i}",
        )
        append_fills([ev], fills_dir)
        total += 1
    files = list(fills_dir.glob("fills_*.parquet"))
    assert len(files) == 18
    df = load_fills(fills_dir)
    assert len(df) == total
