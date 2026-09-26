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


def _fill(ts: pd.Timestamp, run_id: str, client_order_id: str) -> FillEvent:
    return FillEvent(
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
        run_id=run_id,
        leg_index=0,
        client_order_id=client_order_id,
    )


def test_append_fills_preserves_every_earlier_row(tmp_path):
    """Appending to an existing month extends it instead of replacing it."""
    fills_dir = tmp_path / "fills3"
    fills_dir.mkdir()
    ts = pd.Timestamp("2026-03-15 00:00:00", tz="UTC")
    append_fills([_fill(ts, "20260315", "id-1")], fills_dir)
    path = fills_dir / "fills_202603.parquet"
    before = path.read_bytes()
    ts2 = pd.Timestamp("2026-03-20 00:00:00", tz="UTC")
    append_fills([_fill(ts2, "20260320", "id-2")], fills_dir)
    assert path.read_bytes() != before
    df = load_fills(fills_dir)
    assert len(df) == 2
    assert set(df["client_order_id"]) == {"id-1", "id-2"}


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


def _identified_fill(ts: pd.Timestamp, run_id: str, client_order_id: str, fill_id: str | None) -> FillEvent:
    return FillEvent(
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
        run_id=run_id,
        leg_index=0,
        client_order_id=client_order_id,
        fill_id=fill_id,
    )


def test_reemitted_fill_ids_are_idempotent(tmp_path):
    """Re-emitting journal:5 alongside journal:6 stores each exactly once."""
    fills_dir = tmp_path / "fills_ids"
    fills_dir.mkdir()
    ts = pd.Timestamp("2026-04-15 00:00:00", tz="UTC")
    append_fills([_identified_fill(ts, "20260415", "id-5", "journal:5")], fills_dir)
    append_fills(
        [
            _identified_fill(ts, "20260415", "id-5-retry", "journal:5"),
            _identified_fill(ts, "20260415", "id-6", "journal:6"),
        ],
        fills_dir,
    )
    df = load_fills(fills_dir)
    assert sorted(df["fill_id"].dropna().tolist()) == ["journal:5", "journal:6"]
    assert len(df) == 2
    # duplicates inside one batch collapse, first wins
    append_fills(
        [
            _identified_fill(ts, "20260415", "id-7", "journal:7"),
            _identified_fill(ts, "20260415", "id-7-dup", "journal:7"),
        ],
        fills_dir,
    )
    df = load_fills(fills_dir)
    assert len(df) == 3
    assert df[df["fill_id"] == "journal:7"]["client_order_id"].tolist() == ["id-7"]


def test_legacy_rows_without_fill_id_survive_append(tmp_path):
    """Legacy partitions without the column load with nulls and keep every row."""
    fills_dir = tmp_path / "fills_legacy"
    fills_dir.mkdir()
    ts = pd.Timestamp("2026-05-15 00:00:00", tz="UTC")
    append_fills(
        [
            _identified_fill(ts, "20260515", "legacy-1", None),
            _identified_fill(ts, "20260515", "legacy-2", None),
        ],
        fills_dir,
    )
    # simulate a pre-migration partition by dropping the column on disk
    path = fills_dir / "fills_202605.parquet"
    legacy_df = pd.read_parquet(path).drop(columns=["fill_id"])
    assert "fill_id" not in legacy_df.columns
    legacy_df.to_parquet(path)
    append_fills([_identified_fill(ts, "20260515", "new-1", "journal:9")], fills_dir)
    df = load_fills(fills_dir)
    assert len(df) == 3
    assert set(df["client_order_id"]) == {"legacy-1", "legacy-2", "new-1"}
    assert "fill_id" in df.columns
    legacy_rows = df[df["client_order_id"].isin(["legacy-1", "legacy-2"])]
    assert legacy_rows["fill_id"].isna().all()
    assert df[df["client_order_id"] == "new-1"]["fill_id"].tolist() == ["journal:9"]


def test_pure_legacy_partition_loads_with_null_fill_ids(tmp_path):
    """A partition written before fill_id existed loads with an all-null fill_id column."""
    fills_dir = tmp_path / "fills_pure_legacy"
    fills_dir.mkdir()
    ts = pd.Timestamp("2026-06-15 00:00:00", tz="UTC")
    append_fills([_identified_fill(ts, "20260615", "legacy-1", None)], fills_dir)
    path = fills_dir / "fills_202606.parquet"
    pd.read_parquet(path).drop(columns=["fill_id"]).to_parquet(path)

    df = load_fills(fills_dir)

    assert df["fill_id"].tolist() == [None]


def test_legacy_null_ids_are_never_deduplicated(tmp_path):
    """Rows without fill_id are appended even when the partition already holds identified and null rows."""
    fills_dir = tmp_path / "fills_mixed"
    fills_dir.mkdir()
    ts = pd.Timestamp("2026-07-15 00:00:00", tz="UTC")
    append_fills(
        [_identified_fill(ts, "20260715", "id-1", "journal:1"), _identified_fill(ts, "20260715", "legacy-1", None)],
        fills_dir,
    )
    append_fills(
        [_identified_fill(ts, "20260715", "id-1-retry", "journal:1"), _identified_fill(ts, "20260715", "legacy-2", None)],
        fills_dir,
    )

    df = load_fills(fills_dir)

    assert sorted(df["client_order_id"]) == ["id-1", "legacy-1", "legacy-2"]


def test_fully_duplicate_reemission_does_not_rewrite_partition(tmp_path):
    """Re-emitting only already-stored fill_ids leaves the partition file untouched."""
    fills_dir = tmp_path / "fills_noop"
    fills_dir.mkdir()
    ts = pd.Timestamp("2026-08-15 00:00:00", tz="UTC")
    append_fills([_identified_fill(ts, "20260815", "id-3", "journal:3")], fills_dir)
    path = fills_dir / "fills_202608.parquet"
    before = path.stat().st_mtime_ns

    append_fills([_identified_fill(ts, "20260815", "id-3-retry", "journal:3")], fills_dir)

    assert path.stat().st_mtime_ns == before
    assert load_fills(fills_dir)["client_order_id"].tolist() == ["id-3"]


def test_load_fills_empty_schema_when_nothing_stored(tmp_path):
    """Missing directory, no shards and only-empty shards all yield the same typed empty frame."""
    missing = load_fills(tmp_path / "absent")
    (tmp_path / "none").mkdir()
    no_shards = load_fills(tmp_path / "none")
    assert list(missing.columns) == list(no_shards.columns)
    assert "fill_id" in missing.columns
    empty_dir = tmp_path / "empty_shard"
    empty_dir.mkdir()
    load_fills(tmp_path / "absent").to_parquet(empty_dir / "fills_202609.parquet")
    only_empty = load_fills(empty_dir)
    assert list(only_empty.columns) == list(missing.columns)
    assert missing.empty
    assert no_shards.empty
    assert only_empty.empty
    assert str(missing["timestamp"].dtype) == "datetime64[ns, UTC]"
