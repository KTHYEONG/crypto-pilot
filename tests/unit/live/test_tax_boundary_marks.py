"""OHLCV boundary-mark derivation invariants."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from src.live.tax_boundary_marks import derive_ohlcv_boundary_marks

B = pd.Timestamp("2026-12-31T15:00Z")


def _lake(tmp_path: Path, symbol: str, rows: list[tuple[int, float]]) -> Path:
    path = tmp_path / f"{symbol}.parquet"
    pd.DataFrame({"timestamp": [r[0] for r in rows], "close": [r[1] for r in rows]}).to_parquet(path)
    return path


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def test_close_of_bar_ending_at_boundary(tmp_path: Path):
    _lake(tmp_path, "BTCUSDT", [
        (_ms(pd.Timestamp("2026-12-31T13:00Z")), 101.5),
        (_ms(pd.Timestamp("2026-12-31T14:00Z")), 102.25),
        (_ms(pd.Timestamp("2026-12-31T15:00Z")), 103.0),
    ])
    out = derive_ohlcv_boundary_marks({"BTCUSDT"}, B, lake_path=lambda s: tmp_path / f"{s}.parquet")
    mark = out["BTCUSDT"]
    assert mark.price == Decimal("102.25")
    assert mark.source == "ohlcv_1h_close"
    assert mark.unavailable_reason is None


def test_missing_data_is_null_with_reason(tmp_path: Path):
    _lake(tmp_path, "NO_BAR", [(_ms(pd.Timestamp("2026-12-31T13:00Z")), 1.0)])
    dup = tmp_path / "DUP.parquet"
    pd.DataFrame({"timestamp": [_ms(pd.Timestamp("2026-12-31T14:00Z"))] * 2, "close": [1.0, 2.0]}).to_parquet(dup)
    nan_path = tmp_path / "NAN.parquet"
    pd.DataFrame({"timestamp": [_ms(pd.Timestamp("2026-12-31T14:00Z"))], "close": [float("nan")]}).to_parquet(nan_path)
    nocol = tmp_path / "NOCOL.parquet"
    pd.DataFrame({"timestamp": [_ms(pd.Timestamp("2026-12-31T14:00Z"))], "open": [1.0]}).to_parquet(nocol)
    paths = {"ABSENT": tmp_path / "ABSENT.parquet", "NO_BAR": tmp_path / "NO_BAR.parquet",
             "DUP": dup, "NAN": nan_path, "NOCOL": nocol}

    def _resolve(symbol: str) -> Path:
        return paths[symbol]

    out = derive_ohlcv_boundary_marks(set(paths), B, lake_path=_resolve)
    assert out["ABSENT"].unavailable_reason == "ohlcv_file_missing"
    assert out["NO_BAR"].unavailable_reason == "ohlcv_bar_missing"
    assert out["DUP"].unavailable_reason == "ohlcv_bar_duplicated"
    assert out["NAN"].unavailable_reason == "ohlcv_close_invalid"
    assert out["NOCOL"].unavailable_reason.startswith("ohlcv_unreadable")
    assert all(m.price is None for m in out.values())


def test_only_boundary_row_is_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import pandas as _pd
    path = _lake(tmp_path, "BTCUSDT", [(_ms(pd.Timestamp("2026-12-31T14:00Z")), 5.0)])
    seen: dict = {}
    real = _pd.read_parquet

    def _spy(target, **kwargs):
        seen.update(kwargs)
        return real(target, **kwargs)

    monkeypatch.setattr(_pd, "read_parquet", _spy)

    def _one(symbol: str) -> Path:
        return path

    derive_ohlcv_boundary_marks({"BTCUSDT"}, B, lake_path=_one)
    assert seen["columns"] == ["timestamp", "close"]
    assert seen["filters"] == [("timestamp", "==", _ms(pd.Timestamp("2026-12-31T14:00Z")))]


def test_boundary_must_be_exact_utc_hour():
    with pytest.raises(ValueError, match="exact UTC hour"):
        derive_ohlcv_boundary_marks({"BTCUSDT"}, B + pd.Timedelta(minutes=1))
    with pytest.raises(ValueError, match="tz-aware"):
        derive_ohlcv_boundary_marks({"BTCUSDT"}, pd.Timestamp("2026-12-31 15:00"))


def test_non_numeric_close_is_invalid(tmp_path: Path):
    path = tmp_path / "STR.parquet"
    pd.DataFrame({"timestamp": [_ms(pd.Timestamp("2026-12-31T14:00Z"))], "close": ["twelve"]}).to_parquet(path)
    out = derive_ohlcv_boundary_marks({"STR"}, B, lake_path=lambda s: path)
    assert out["STR"].price is None
    assert out["STR"].unavailable_reason == "ohlcv_close_invalid"


def test_reader_programming_errors_are_not_hidden(tmp_path: Path, monkeypatch):
    path = _lake(tmp_path, "BTCUSDT", [(_ms(B - pd.Timedelta(hours=1)), 100.0)])

    def fail(*args, **kwargs):
        raise RuntimeError("reader defect")

    monkeypatch.setattr(pd, "read_parquet", fail)
    with pytest.raises(RuntimeError, match="reader defect"):
        derive_ohlcv_boundary_marks({"BTCUSDT"}, B, lake_path=lambda symbol: path)


def test_reader_missing_columns_are_unreadable(tmp_path: Path, monkeypatch):
    path = _lake(tmp_path, "BTCUSDT", [(_ms(B - pd.Timedelta(hours=1)), 100.0)])
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: pd.DataFrame({"timestamp": [1]}))
    result = derive_ohlcv_boundary_marks({"BTCUSDT"}, B, lake_path=lambda symbol: path)
    assert result["BTCUSDT"].price is None
    assert result["BTCUSDT"].unavailable_reason == "ohlcv_unreadable:KeyError"
