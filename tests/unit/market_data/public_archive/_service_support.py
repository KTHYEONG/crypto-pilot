from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import date

import pandas as pd

from src.market_data.public_archive.catalog import daily_key

_METRICS_HEADER = (
    "create_time,symbol,sum_open_interest,sum_open_interest_value,"
    "count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,"
    "count_long_short_ratio,sum_taker_long_short_vol_ratio"
)

_AGG_HEADER = (
    "agg_trade_id,price,quantity,first_trade_id,last_trade_id,"
    "transact_time,is_buyer_maker"
)


def _zip_csv(body: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("data.csv", body)
    return buffer.getvalue()


def _metrics_row(symbol: str, day: date, hour: int = 0) -> str:
    stamp = (pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=hour)).strftime("%Y-%m-%d %H:%M:%S")
    return f"{stamp},{symbol},1,2,3,4,5,6"


def _metrics_zip(symbol: str, day: date) -> bytes:
    return _zip_csv(f"{_METRICS_HEADER}\n{_metrics_row(symbol, day)}\n")


def _agg_zip(symbol: str, day: date) -> bytes:
    ms = int(pd.Timestamp(day, tz="UTC").value // 10**6) + 23 * 3_600_000 + 30 * 60_000
    return _zip_csv(f"{_AGG_HEADER}\n7,100.0,2.0,1,2,{ms},true\n")


class FakeDownloader:
    def __init__(self) -> None:
        self.keys: list[str] = []
        self.blobs: dict[str, bytes] = {}
        self.fetch_calls: list[str] = []
        self.fail_listing = False

    def add_zip(self, full_key: str, payload: bytes) -> None:
        digest = hashlib.sha256(payload).hexdigest()
        stripped = full_key.removeprefix("data/futures/um/")
        self.keys.extend([full_key, full_key + ".CHECKSUM"])
        self.blobs[stripped] = payload
        self.blobs[stripped + ".CHECKSUM"] = f"{digest}  {full_key}\n".encode()

    def add_corrupt(self, full_key: str, payload: bytes) -> None:
        stripped = full_key.removeprefix("data/futures/um/")
        self.keys.extend([full_key, full_key + ".CHECKSUM"])
        self.blobs[stripped] = payload
        self.blobs[stripped + ".CHECKSUM"] = ("0" * 64 + f"  {full_key}\n").encode()

    def list_object_keys(self, *, prefix: str, timeout: int | None = None) -> list[str] | None:
        del timeout
        if self.fail_listing:
            return None
        return [k for k in self.keys if k.startswith(prefix)]

    def fetch_archive_bytes(self, *parts: str) -> bytes | None:
        key = "/".join(parts)
        self.fetch_calls.append(key)
        return self.blobs.get(key)

    def list_all_symbols(self) -> list[str]:
        return ["BTCUSDT"]


def _metrics_fake(symbols: list[str], days: list[date], *, corrupt: set[date] | None = None) -> FakeDownloader:
    fake = FakeDownloader()
    for symbol in symbols:
        for day in days:
            key = daily_key("metrics", symbol, day)
            if corrupt and day in corrupt:
                fake.add_corrupt(key, _metrics_zip(symbol, day))
            else:
                fake.add_zip(key, _metrics_zip(symbol, day))
    return fake


_NOW = pd.Timestamp("2024-02-01", tz="UTC")


def _agg_rows_zip(day: date, rows: list[tuple[int, float, float, str]], *, header: bool = True) -> bytes:
    start = int(pd.Timestamp(day, tz="UTC").value // 10**6)
    lines = [f"{i},{price},{qty},{i},{i},{start + offset},{maker}" for i, (offset, price, qty, maker) in enumerate(rows)]
    return _zip_csv(("\n".join([_AGG_HEADER, *lines]) if header else "\n".join(lines)) + "\n")
