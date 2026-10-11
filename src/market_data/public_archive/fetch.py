from __future__ import annotations

import hashlib
import io
import zipfile

import pandas as pd

from src.common.errors import DataIntegrityError
from src.market_data.binance.vision import BinanceVisionDownloader

_BUCKET_PREFIX = "data/futures/um/"

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
        expected = checksum_bytes.decode("ascii", errors="strict").split()[0]
    except (UnicodeDecodeError, IndexError) as exc:
        raise DataIntegrityError(f"malformed checksum for {key!r}") from exc
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
            frame = pd.read_csv(handle, header=None)
    if not frame.empty and str(frame.iloc[0, 0]).strip().lower() in _KNOWN_FIRST_COLUMNS:
        frame.columns = frame.iloc[0].tolist()
        return frame.iloc[1:].reset_index(drop=True)
    return frame
