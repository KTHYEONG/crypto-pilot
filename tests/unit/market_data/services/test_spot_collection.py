from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.market_data.services.borrow_collection as borrow
import src.market_data.services.spot_collection as spot
import src.market_data.storage.manifest as manifest
from src.common.errors import DataIntegrityError

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _ms(index: pd.DatetimeIndex) -> pd.Series:
    return (index - _EPOCH) // pd.Timedelta("1ms")


@pytest.fixture
def spot_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "spot"

    def _spot(symbol: str, timeframe: str) -> Path:
        return root / "ohlcv" / timeframe / f"{symbol.replace('/', '_')}.parquet"

    def _borrow(symbol: str) -> Path:
        return root / "borrow" / f"{symbol.replace('/', '_')}.parquet"

    monkeypatch.setattr(spot, "spot_ohlcv_path", _spot)
    monkeypatch.setattr(borrow, "borrow_path", _borrow)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", root / "manifest.json")
    return root


class TestEnsureSpotOhlcv:
    def test_missing_bar_detector_reports_internal_gap(self) -> None:
        idx = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 02:00"], tz="UTC")
        frame = pd.DataFrame({"timestamp": _ms(idx)})

        missing = spot._missing_bar_timestamps(
            frame,
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-01 03:00", tz="UTC"),
            pd.Timedelta(hours=1),
        )

        assert list(missing) == [pd.Timestamp("2024-01-01 01:00", tz="UTC")]

    def test_writes_canonical_file_and_manifest(self, spot_paths: Path, monkeypatch) -> None:
        idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
        fetch_result = pd.DataFrame({
            "timestamp": _ms(idx),
            "open": [100.0] * 4, "high": [101.0] * 4, "low": [99.0] * 4,
            "close": [100.5] * 4, "volume": [10.0] * 4,
            "quote_vol": [1000.0] * 4,
            "taker_buy_base_volume": [5.0] * 4,
            "taker_buy_quote_volume": [500.0] * 4,
        })

        class FakeClient:
            def fetch_spot_ohlcv(self, symbol, timeframe, start, end):
                return fetch_result

        spot.ensure_spot_ohlcv("BTCUSDT", "1h", "2024-01-01", "2024-01-01 04:00", client=FakeClient())  # type: ignore[arg-type]

        path = spot.spot_ohlcv_path("BTCUSDT", "1h")
        assert path.exists()
        out = pd.read_parquet(path)
        assert len(out) == 4
        assert str(out["open"].dtype) == "float32"
        manifest_data = json.loads(manifest.MANIFEST_PATH.read_text(encoding="utf-8"))
        record = manifest_data["datasets"]["ohlcv/1h"]["BTCUSDT"]
        assert record["venue"] == "binance"
        assert record["row_count"] == 4
        assert record["sha256"] == manifest._file_sha256(path)

    def test_incremental_fetch_only_uncovered_tail(self, spot_paths: Path, monkeypatch) -> None:
        # SC-SPOT-01: existing cached rows are not re-fetched; only the
        # uncovered range after the cached maximum is requested.
        cached_idx = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
        cached = pd.DataFrame({
            "timestamp": _ms(cached_idx),
            "open": [100.0, 100.0], "high": [101.0, 101.0], "low": [99.0, 99.0],
            "close": [100.5, 100.5], "volume": [10.0, 10.0],
            "quote_vol": [1000.0, 1000.0],
        })
        path = spot.spot_ohlcv_path("BTCUSDT", "1h")
        path.parent.mkdir(parents=True, exist_ok=True)
        cached.to_parquet(path)

        new_idx = pd.date_range("2024-01-01 02:00", periods=2, freq="1h", tz="UTC")
        new_part = pd.DataFrame({
            "timestamp": _ms(new_idx),
            "open": [101.0, 101.0], "high": [102.0, 102.0], "low": [100.0, 100.0],
            "close": [101.5, 101.5], "volume": [10.0, 10.0],
            "quote_vol": [1000.0, 1000.0],
        })

        class FakeClient:
            def __init__(self):
                self.requested: list[str] = []

            def fetch_spot_ohlcv(self, symbol, timeframe, start, end):
                self.requested.append(start)
                return new_part

        client = FakeClient()
        spot.ensure_spot_ohlcv("BTCUSDT", "1h", "2024-01-01", "2024-01-01 04:00", client=client)  # type: ignore[arg-type]
        assert client.requested, "fetch must be called for the uncovered tail"
        assert "2024-01-01 01:00" in client.requested[0]
        out = pd.read_parquet(path)
        assert len(out) == 4

    def test_internal_gap_is_refetched_without_interpolation(self, spot_paths: Path) -> None:
        cached_idx = pd.DatetimeIndex(
            ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 03:00"], tz="UTC",
        )
        cached = pd.DataFrame({
            "timestamp": _ms(cached_idx), "open": [100.0] * 3, "high": [101.0] * 3,
            "low": [99.0] * 3, "close": [100.5] * 3, "volume": [10.0] * 3,
        })
        path = spot.spot_ohlcv_path("BTCUSDT", "1h")
        path.parent.mkdir(parents=True, exist_ok=True)
        cached.to_parquet(path)

        missing = pd.DataFrame({
            "timestamp": _ms(pd.DatetimeIndex(["2024-01-01 02:00"], tz="UTC")),
            "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [10.0],
        })

        class FakeClient:
            def __init__(self) -> None:
                self.requested: list[tuple[str, str]] = []

            def fetch_spot_ohlcv(self, symbol, timeframe, start, end):
                self.requested.append((start, end))
                return missing

        client = FakeClient()
        spot.ensure_spot_ohlcv("BTCUSDT", "1h", "2024-01-01", "2024-01-01 04:00", client=client)  # type: ignore[arg-type]

        out = pd.read_parquet(path)
        timestamps = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
        assert len(out) == 4
        assert timestamps.diff().dropna().eq(pd.Timedelta(hours=1)).all()
        assert any("02:00" in start for start, _ in client.requested)

    def test_repairs_one_gap_with_boundary_bridge_and_manifest(self, spot_paths: Path) -> None:
        idx = pd.DatetimeIndex(
            ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 03:00"], tz="UTC",
        )
        frame = pd.DataFrame({
            "timestamp": _ms(idx), "open": [100.0, 101.0, 99.0],
            "high": [101.0, 102.0, 100.0], "low": [99.0, 100.0, 98.0],
            "close": [100.5, 101.5, 99.5], "volume": [10.0, 11.0, 12.0],
        })
        path = spot.spot_ohlcv_path("BTCUSDT", "1h")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)

        spot.repair_spot_ohlcv_gap("BTCUSDT", "1h", "2024-01-01 02:00")

        out = pd.read_parquet(path)
        repaired = out[pd.to_datetime(out["timestamp"], unit="ms", utc=True) == pd.Timestamp("2024-01-01 02:00", tz="UTC")].iloc[0]
        assert repaired["open"] == 101.5
        assert repaired["close"] == 99.0
        assert repaired["high"] == 101.5
        assert repaired["low"] == 99.0
        expected_volume = np.expm1((np.log1p(11.0) + np.log1p(12.0)) / 2.0)
        assert repaired["volume"] == pytest.approx(expected_volume)
        assert repaired["quote_vol"] > 0.0
        assert 0.0 <= repaired["taker_buy_base_volume"] <= repaired["volume"]
        assert 0.0 <= repaired["taker_buy_quote_volume"] <= repaired["quote_vol"]
        manifest_data = json.loads(manifest.MANIFEST_PATH.read_text(encoding="utf-8"))
        record = manifest_data["datasets"]["ohlcv/1h"]["BTCUSDT"]
        assert record["imputations"][0]["method"] == "bridge_unknown_volume"
        assert record["data_quality"]["volume_quality"] == "unknown"

    def test_ensure_auto_repairs_isolated_gap_when_fetch_cannot_fill(
        self, spot_paths: Path,
    ) -> None:
        idx = pd.DatetimeIndex(
            ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 03:00"], tz="UTC",
        )
        frame = pd.DataFrame({
            "timestamp": _ms(idx), "open": [100.0, 101.0, 99.0],
            "high": [101.0, 102.0, 100.0], "low": [99.0, 100.0, 98.0],
            "close": [100.5, 101.5, 99.5], "volume": [10.0, 11.0, 12.0],
        })
        path = spot.spot_ohlcv_path("ETHUSDT", "1h")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)

        class EmptyClient:
            def fetch_spot_ohlcv(self, symbol, timeframe, start, end):
                return pd.DataFrame()

        spot.ensure_spot_ohlcv(
            "ETHUSDT", "1h", "2024-01-01", "2024-01-01 04:00",
            client=EmptyClient(),  # type: ignore[arg-type]
        )

        out = pd.read_parquet(path)
        timestamps = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
        assert len(out) == 4
        assert timestamps.diff().dropna().eq(pd.Timedelta(hours=1)).all()
        record = json.loads(manifest.MANIFEST_PATH.read_text(encoding="utf-8"))[
            "datasets"
        ]["ohlcv/1h"]["ETHUSDT"]
        assert record["data_quality"]["volume_quality"] == "unknown"
        assert record["imputations"][0]["timestamp"] == "2024-01-01T02:00:00+00:00"


class TestImportQuoteBorrowHistory:
    def test_imports_hourly_export_to_canonical_columns(self, spot_paths: Path) -> None:
        idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
        src = spot_paths / "src" / "borrow_export.parquet"
        src.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "datetime": idx,
            "borrow_rate": [0.001, 0.001, 0.001, 0.001],
        }).to_parquet(src)

        borrow.import_quote_borrow_history("BTCUSDT", src, "operator:v1", "hourly")

        out = pd.read_parquet(borrow.borrow_path("BTCUSDT"))
        assert list(out.columns) == ["timestamp", "borrow_rate", "accrual_seconds"]
        assert (out["accrual_seconds"] == 3600.0).all()
        manifest_data = json.loads(manifest.MANIFEST_PATH.read_text(encoding="utf-8"))
        record = manifest_data["datasets"]["borrow"]["BTCUSDT"]
        assert record["source_locator"] == "operator:v1"
        assert record["conversion"]["source_units"] == "hourly"


class TestBinanceMarginBorrowCollection:
    def test_converts_daily_interest_rate_to_exact_interval_cost(self) -> None:
        rates = pd.DataFrame({
            "timestamp": [0, 3_600_000, 7_200_000],
            "dailyInterestRate": [0.024, 0.024, 0.024],
        })

        events = borrow._daily_rates_to_borrow_events(
            rates,
            pd.Timestamp(0, unit="ms", tz="UTC"),
            pd.Timestamp(3_600_000, unit="ms", tz="UTC"),
        )

        assert len(events) == 1
        assert events.iloc[0]["borrow_rate"] == pytest.approx(0.001)
        assert events.iloc[0]["accrual_seconds"] == 3600.0

    def test_rejects_unbracketed_rate_history(self) -> None:
        rates = pd.DataFrame({
            "timestamp": [0, 3_600_000],
            "dailyInterestRate": [0.024, 0.024],
        })

        with pytest.raises(DataIntegrityError, match="boundary event"):
            borrow._daily_rates_to_borrow_events(
                rates,
                pd.Timestamp(0, unit="ms", tz="UTC"),
                pd.Timestamp(3_600_000, unit="ms", tz="UTC"),
            )

    def test_collects_binance_history_to_canonical_borrow_file(self, spot_paths: Path) -> None:
        raw = pd.DataFrame({
            "timestamp": [-3_600_000, 0, 3_600_000, 7_200_000],
            "dailyInterestRate": [0.024, 0.024, 0.024, 0.024],
            "asset": ["USDT"] * 4,
            "vipLevel": [0] * 4,
        })

        class FakeMarginClient:
            def fetch_margin_interest_rate_history(self, asset, start, end):
                assert asset == "USDT"
                return raw

        borrow.collect_binance_quote_borrow_history(
            "BTCUSDT", "USDT", "1970-01-01", "1970-01-01 01:00",
            client=FakeMarginClient(),  # type: ignore[arg-type]
        )

        out = pd.read_parquet(borrow.borrow_path("BTCUSDT"))
        assert list(out.columns) == ["timestamp", "borrow_rate", "accrual_seconds"]
        assert len(out) == 1
        manifest_data = json.loads(manifest.MANIFEST_PATH.read_text(encoding="utf-8"))
        record = manifest_data["datasets"]["borrow"]["BTCUSDT"]
        assert record["source_locator"] == "sapi/v1/margin/interestRateHistory"
        assert record["conversion"]["source_units"] == "dailyInterestRate"

    def test_uses_explicit_accrual_seconds_when_present(self, spot_paths: Path) -> None:
        idx = pd.date_range("2024-01-01", periods=2, freq="4h", tz="UTC")
        src = spot_paths / "src" / "borrow.parquet"
        src.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "timestamp": _ms(idx),
            "borrow_rate": [0.002, 0.002],
            "accrual_seconds": [14400, 14400],
        }).to_parquet(src)

        borrow.import_quote_borrow_history("BTCUSDT", src, "export:v2", "unknown-source")
        out = pd.read_parquet(borrow.borrow_path("BTCUSDT"))
        assert (out["accrual_seconds"] == 14400.0).all()

    def test_rejects_ambiguous_rate_period(self, spot_paths: Path) -> None:
        idx = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
        src = spot_paths / "src" / "borrow.parquet"
        src.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "datetime": idx,
            "borrow_rate": [0.001, 0.001],
        }).to_parquet(src)
        with pytest.raises(DataIntegrityError, match="ambiguous"):
            borrow.import_quote_borrow_history("BTCUSDT", src, "op:v1", "per-second-of-magic")

    def test_rejects_overlapping_events(self, spot_paths: Path) -> None:
        idx = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
        src = spot_paths / "src" / "borrow.parquet"
        src.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "datetime": idx,
            "borrow_rate": [0.001, 0.001],
            "accrual_seconds": [7200, 7200],
        }).to_parquet(src)
        with pytest.raises(DataIntegrityError, match="overlap"):
            borrow.import_quote_borrow_history("BTCUSDT", src, "op:v1", "hourly")

    def test_rejects_gap_between_events(self, spot_paths: Path) -> None:
        idx = pd.date_range("2024-01-01", periods=2, freq="4h", tz="UTC")
        src = spot_paths / "src" / "borrow.parquet"
        src.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "datetime": idx,
            "borrow_rate": [0.001, 0.001],
            "accrual_seconds": [3600, 3600],
        }).to_parquet(src)
        with pytest.raises(DataIntegrityError, match="gap"):
            borrow.import_quote_borrow_history("BTCUSDT", src, "op:v1", "hourly")

    def test_rejects_duplicate_timestamps(self, spot_paths: Path) -> None:
        idx = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 00:00"], tz="UTC")
        src = spot_paths / "src" / "borrow.parquet"
        src.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "datetime": idx,
            "borrow_rate": [0.001, 0.002],
            "accrual_seconds": [3600, 3600],
        }).to_parquet(src)
        with pytest.raises(DataIntegrityError, match="duplicate"):
            borrow.import_quote_borrow_history("BTCUSDT", src, "op:v1", "hourly")

    def test_rejects_non_positive_accrual(self, spot_paths: Path) -> None:
        idx = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
        src = spot_paths / "src" / "borrow.parquet"
        src.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "datetime": idx,
            "borrow_rate": [0.001, 0.001],
            "accrual_seconds": [0, 3600],
        }).to_parquet(src)
        with pytest.raises(DataIntegrityError, match="> 0"):
            borrow.import_quote_borrow_history("BTCUSDT", src, "op:v1", "hourly")

    def test_rejects_non_finite_rate(self, spot_paths: Path) -> None:
        idx = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
        src = spot_paths / "src" / "borrow.parquet"
        src.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "datetime": idx,
            "borrow_rate": [0.001, float("nan")],
            "accrual_seconds": [3600, 3600],
        }).to_parquet(src)
        with pytest.raises(DataIntegrityError, match="non-finite"):
            borrow.import_quote_borrow_history("BTCUSDT", src, "op:v1", "hourly")
