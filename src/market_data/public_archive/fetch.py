from __future__ import annotations

import hashlib
import io
import re
import zipfile

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.market_data.binance.vision import BinanceVisionDownloader

_BUCKET_PREFIX = "data/futures/um/"

_AGG_INT_COLUMNS = frozenset({
    "agg_trade_id",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
})

_AGG_FLOAT_COLUMNS = frozenset({
    "price",
    "quantity",
})

_AGG_INT_POSITIONS = frozenset({0, 3, 4, 5})
_AGG_FLOAT_POSITIONS = frozenset({1, 2})

_AGG_HEADERLESS_COLUMNS = (
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
)

_KNOWN_FIRST_COLUMNS = frozenset({
    "create_time",
    "calc_time",
    "timestamp",
    "open_time",
    "agg_trade_id",
    "symbol",
    "transact_time",
    "price",
    "percentage",
})


def _coerce_agg_frame(frame: pd.DataFrame, *, named: bool) -> pd.DataFrame:
    if named:
        for column in _AGG_INT_COLUMNS:
            if column in frame.columns:
                frame[column] = _integer_field(frame[column])
        for column in _AGG_FLOAT_COLUMNS:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
        if "is_buyer_maker" in frame.columns:
            frame["is_buyer_maker"] = _normalize_maker(frame["is_buyer_maker"])
    elif frame.shape[1] == 7:
        for position in sorted(_AGG_INT_POSITIONS):
            frame[position] = _integer_field(frame[position])
        for position in sorted(_AGG_FLOAT_POSITIONS):
            frame[position] = pd.to_numeric(frame[position], errors="raise").astype("float64")
        frame[6] = _normalize_maker(frame[6])
    return frame


def _integer_field(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="raise")
    if not np.isfinite(values).all() or (values % 1 != 0).any() or (values < 0).any() or (values >= 2**63).any():
        raise DataIntegrityError("invalid integer archive field")
    return values.astype("int64")


def _normalize_maker(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype("bool")

    def _one(value: object) -> bool:
        if isinstance(value, (int, np.integer)):
            if int(value) in (0, 1):
                return bool(int(value))
            raise DataIntegrityError(f"invalid is_buyer_maker: {value!r}")
        text = str(value).strip().lower()
        if text == "true":
            return True
        if text == "false":
            return False
        raise DataIntegrityError(f"invalid is_buyer_maker: {value!r}")

    return series.map(_one).astype("bool")


_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def _strip_prefix(key: str) -> tuple[str, ...]:
    if key.startswith(_BUCKET_PREFIX):
        key = key[len(_BUCKET_PREFIX):]
    return tuple(part for part in key.split("/") if part)


def fetch_verified_zip(
    downloader: BinanceVisionDownloader, key: str
) -> tuple[bytes, str] | None:
    """Download a bucket key with checksum verification."""
    zip_bytes = downloader.fetch_archive_bytes(*_strip_prefix(key))
    if zip_bytes is None:
        return None
    checksum_bytes = downloader.fetch_archive_bytes(*_strip_prefix(key + ".CHECKSUM"))
    if checksum_bytes is None:
        raise DataIntegrityError(f"missing checksum for {key!r}")
    try:
        expected = checksum_bytes.decode("utf-8", errors="strict").split()[0]
    except (UnicodeDecodeError, IndexError) as exc:
        raise DataIntegrityError(f"malformed checksum for {key!r}") from exc
    if _SHA256_HEX.fullmatch(expected.strip().lower()) is None:
        raise DataIntegrityError(f"malformed checksum for {key!r}")
    actual = hashlib.sha256(zip_bytes).hexdigest()
    if actual != expected.strip().lower():
        raise DataIntegrityError(f"checksum mismatch for {key!r}")
    return zip_bytes, actual


def read_single_csv(zip_bytes: bytes) -> pd.DataFrame:
    """Read the single CSV member of a Vision zip."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        members = [name for name in zf.namelist() if name.endswith(".csv")]
        if len(members) != 1:
            raise DataIntegrityError(f"expected one csv member, found {len(members)}")
        with zf.open(members[0]) as handle:
            first_line = handle.readline().decode("utf-8-sig", errors="strict")
            first_cell = first_line.split(",", 1)[0].strip().strip('"').strip("'").lower()
            named = first_cell in _KNOWN_FIRST_COLUMNS
            handle.seek(0)
            frame = pd.read_csv(handle, header=0 if named else None, low_memory=False)
    frame = _coerce_agg_frame(frame, named=named)
    if not named and list(frame.columns) == list(range(7)):
        frame.columns = list(_AGG_HEADERLESS_COLUMNS)
    return frame
