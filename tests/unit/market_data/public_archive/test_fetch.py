from __future__ import annotations

import hashlib
import io
import zipfile

import pytest

from src.common.errors import DataIntegrityError
from src.market_data.binance.vision import BinanceVisionDownloader
from src.market_data.public_archive.catalog import is_perpetual_usdt
from src.market_data.public_archive.fetch import fetch_verified_zip, read_single_csv


def _zip_bytes(members: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, body in members.items():
            zf.writestr(name, body)
    return buffer.getvalue()


def _downloader_with(mapping: dict[str, bytes | None]) -> BinanceVisionDownloader:
    downloader = BinanceVisionDownloader()

    def _fake(*parts: str) -> bytes | None:
        key = "/".join(parts)
        if key not in mapping:
            raise AssertionError(f"unexpected key: {key}")
        return mapping[key]

    downloader.fetch_archive_bytes = _fake  # type: ignore[method-assign]
    return downloader


def test_checksum_mismatch_raises_and_writes_nothing(tmp_path) -> None:
    payload = _zip_bytes({"a.csv": "x,y\n1,2\n"})
    mapping = {
        "daily/metrics/BTCUSDT/f": payload,
        "daily/metrics/BTCUSDT/f.CHECKSUM": b"0" * 64 + b"  f\n",
    }
    downloader = _downloader_with(mapping)
    with pytest.raises(DataIntegrityError):
        fetch_verified_zip(downloader, "data/futures/um/daily/metrics/BTCUSDT/f")
    assert list(tmp_path.iterdir()) == []


def test_missing_zip_returns_none() -> None:
    downloader = _downloader_with({"daily/metrics/BTCUSDT/f": None})
    assert fetch_verified_zip(downloader, "data/futures/um/daily/metrics/BTCUSDT/f") is None


def test_fetch_verified_zip_ok() -> None:
    payload = _zip_bytes({"a.csv": "x,y\n1,2\n"})
    digest = hashlib.sha256(payload).hexdigest()
    mapping = {
        "daily/metrics/BTCUSDT/f": payload,
        "daily/metrics/BTCUSDT/f.CHECKSUM": f"{digest}  f\n".encode(),
    }
    downloader = _downloader_with(mapping)
    body, sha = fetch_verified_zip(downloader, "data/futures/um/daily/metrics/BTCUSDT/f") or (None, None)
    assert body == payload
    assert sha == digest


def test_single_csv_required() -> None:
    payload = _zip_bytes({"a.csv": "x\n1\n", "b.csv": "x\n2\n"})
    with pytest.raises(DataIntegrityError):
        read_single_csv(payload)
    empty = io.BytesIO()
    with zipfile.ZipFile(empty, "w") as zf:
        zf.writestr("note.txt", "hi")
    with pytest.raises(DataIntegrityError):
        read_single_csv(empty.getvalue())


def test_perpetual_filter() -> None:
    assert is_perpetual_usdt("BTCUSDT") is True
    assert is_perpetual_usdt("BTCUSDT_250627") is False
    assert is_perpetual_usdt("币安人生USDT") is True
    assert is_perpetual_usdt("ETHBUSD") is False


_AGG_HEADER = (
    "agg_trade_id,price,quantity,first_trade_id,last_trade_id,"
    "transact_time,is_buyer_maker"
)


def test_header_and_headerless_equal_dtypes() -> None:
    import pandas as pd

    row = "7,100.5,2.25,1,2,1705270200000,true\n"
    headered = _zip_bytes({"a.csv": f"{_AGG_HEADER}\n{row}"})
    headerless = _zip_bytes({"a.csv": row})
    first = read_single_csv(headered)
    second = read_single_csv(headerless)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 1
    for dtype in first.dtypes:
        assert str(dtype) != "object"
    assert str(first["agg_trade_id"].dtype) == "int64"
    assert str(first["transact_time"].dtype) == "int64"
    assert str(first["price"].dtype) == "float64"
    assert str(first["quantity"].dtype) == "float64"
    assert str(first["is_buyer_maker"].dtype) == "bool"


def test_no_dtype_warning() -> None:
    import warnings

    row = "7,100.5,2.25,1,2,1705270200000,true\n"
    payload = _zip_bytes({"a.csv": f"{_AGG_HEADER}\n{row}"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frame = read_single_csv(payload)
    assert len(frame) == 1


def test_maker_int_forms_coerce_to_bool() -> None:
    body = (
        f"{_AGG_HEADER}\n"
        "7,100.5,2.25,1,2,1705270200000,1\n"
        "8,101.5,3.25,3,4,1705270260000,0\n"
    )
    frame = read_single_csv(_zip_bytes({"a.csv": body}))
    assert str(frame["is_buyer_maker"].dtype) == "bool"
    assert frame["is_buyer_maker"].tolist() == [True, False]


def test_maker_invalid_form_raises() -> None:
    import pytest

    from src.common.errors import DataIntegrityError

    body = f"{_AGG_HEADER}\n7,100.5,2.25,1,2,1705270200000,maybe\n"
    with pytest.raises(DataIntegrityError):
        read_single_csv(_zip_bytes({"a.csv": body}))
    numeric = f"{_AGG_HEADER}\n7,100.5,2.25,1,2,1705270200000,2\n"
    with pytest.raises(DataIntegrityError):
        read_single_csv(_zip_bytes({"a.csv": numeric}))


def test_maker_padded_forms_coerce_to_bool() -> None:
    body = (
        f"{_AGG_HEADER}\n"
        "7,100.5,2.25,1,2,1705270200000, true \n"
        "8,101.5,3.25,3,4,1705270260000, false \n"
    )
    frame = read_single_csv(_zip_bytes({"a.csv": body}))
    assert str(frame["is_buyer_maker"].dtype) == "bool"
    assert frame["is_buyer_maker"].tolist() == [True, False]


@pytest.mark.parametrize("value", ["7.5", "-1", "9223372036854775808", "NaN", "inf"])
@pytest.mark.parametrize("header", [True, False])
def test_invalid_integer_fields_are_not_silently_coerced(value: str, header: bool) -> None:
    body = f"{value},100.5,2.25,1,2,1705270200000,true\n"
    if header:
        body = f"{_AGG_HEADER}\n{body}"
    with pytest.raises(DataIntegrityError, match="integer archive field"):
        read_single_csv(_zip_bytes({"a.csv": body}))


def test_header_only_aggtrades_has_canonical_dtypes() -> None:
    frame = read_single_csv(_zip_bytes({"a.csv": f"{_AGG_HEADER}\n"}))
    assert frame.empty
    assert [str(dtype) for dtype in frame.dtypes] == [
        "int64", "float64", "float64", "int64", "int64", "int64", "bool",
    ]


def test_csv_member_is_streamed_with_bom_header(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _zip_bytes({"a.csv": f"\ufeff{_AGG_HEADER}\n7,100,2,1,2,1705270200000,true\n"})

    def forbid_full_read(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("CSV member must not be materialized before parsing")

    monkeypatch.setattr(zipfile.ZipFile, "read", forbid_full_read)
    frame = read_single_csv(payload)
    assert frame["agg_trade_id"].tolist() == [7]
    assert frame["price"].tolist() == [100.0]


def test_fetch_verified_zip_accepts_non_ascii_filename_in_checksum() -> None:
    payload = _zip_bytes({"a.csv": "x,y\n1,2\n"})
    digest = hashlib.sha256(payload).hexdigest()
    mapping = {
        "daily/metrics/币安人生USDT/f": payload,
        "daily/metrics/币安人生USDT/f.CHECKSUM": f"{digest}  币安人生USDT-f.zip\n".encode(),
    }
    downloader = _downloader_with(mapping)
    result = fetch_verified_zip(downloader, "data/futures/um/daily/metrics/币安人生USDT/f")
    assert result is not None and result[1] == digest


def test_fetch_verified_zip_rejects_non_hex_checksum() -> None:
    payload = _zip_bytes({"a.csv": "x,y\n1,2\n"})
    mapping = {
        "daily/metrics/BTCUSDT/f": payload,
        "daily/metrics/BTCUSDT/f.CHECKSUM": b"<html>error</html>\n",
    }
    downloader = _downloader_with(mapping)
    with pytest.raises(DataIntegrityError, match="malformed checksum"):
        fetch_verified_zip(downloader, "data/futures/um/daily/metrics/BTCUSDT/f")
