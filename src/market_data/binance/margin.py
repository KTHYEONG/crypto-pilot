from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, cast

import pandas as pd


class BinanceMarginClient:
    """Signed Binance Margin interest-rate history adapter.

    The Margin endpoint returns a daily rate at rate-change timestamps.  This
    adapter only retrieves and validates those source observations; conversion
    to a canonical accrued borrow event belongs in the spot borrow collector.
    """

    BASE_URL = "https://api.binance.com"
    INTEREST_HISTORY_PATH = "/sapi/v1/margin/interestRateHistory"
    MAX_QUERY_WINDOW = pd.Timedelta(days=30)
    TIMEOUT_SECONDS = 30

    def __init__(self, api_key: str | None = None, secret: str | None = None) -> None:
        self.api_key = api_key or os.getenv("BINANCE_API_KEY")
        self.secret = secret or os.getenv("BINANCE_SECRET")

    @staticmethod
    def _utc_timestamp(value: str | datetime) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")

    def _signed_interest_request(self, asset: str, start: pd.Timestamp, end: pd.Timestamp) -> list[dict[str, Any]]:
        if not self.api_key or not self.secret:
            raise RuntimeError("Binance Margin credentials are required for interest-rate history")
        params = {
            "asset": asset,
            "startTime": int(start.timestamp() * 1000),
            "endTime": int(end.timestamp() * 1000),
            "timestamp": int(time.time() * 1000),
            "recvWindow": 5000,
        }
        query = urllib.parse.urlencode(params)
        signature = hmac.new(self.secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        url = f"{self.BASE_URL}{self.INTEREST_HISTORY_PATH}?{query}&signature={signature}"
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL scheme: {url}")
        request = urllib.request.Request(  # noqa: S310
            url, method="GET", headers={"X-MBX-APIKEY": self.api_key},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.TIMEOUT_SECONDS) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"Binance Margin interest history request failed: HTTP {exc.code} {detail}") from exc
        if not isinstance(payload, list):
            raise RuntimeError("Binance Margin interest history payload must be a list")
        if not all(isinstance(row, dict) for row in payload):
            raise RuntimeError("Binance Margin interest history payload contains a non-object row")
        return cast(list[dict[str, Any]], payload)

    def fetch_margin_interest_rate_history(
        self,
        asset: str,
        start: str | datetime,
        end: str | datetime,
    ) -> pd.DataFrame:
        """Fetch all source rate-change events in a bounded historical range."""
        normalized_asset = asset.strip().upper()
        if not normalized_asset:
            raise ValueError("asset must not be empty")
        start_ts = self._utc_timestamp(start)
        end_ts = self._utc_timestamp(end)
        if start_ts >= end_ts:
            raise ValueError(f"invalid interest-rate range start={start} end={end}")

        rows: list[dict[str, Any]] = []
        cursor = start_ts
        while cursor < end_ts:
            window_end = min(cursor + self.MAX_QUERY_WINDOW, end_ts)
            rows.extend(self._signed_interest_request(normalized_asset, cursor, window_end))
            cursor = window_end

        columns = ["timestamp", "dailyInterestRate", "asset", "vipLevel"]
        if not rows:
            return pd.DataFrame(columns=columns)
        frame = pd.DataFrame(rows)
        if not set(columns).issubset(frame.columns):
            raise RuntimeError("Binance Margin interest history payload missing required fields")
        frame = frame.loc[:, columns].copy()
        frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
        frame["dailyInterestRate"] = pd.to_numeric(frame["dailyInterestRate"], errors="coerce")
        if frame[["timestamp", "dailyInterestRate"]].isna().any().any():
            raise RuntimeError("Binance Margin interest history contains invalid numeric values")
        if (frame["asset"].astype(str).str.upper() != normalized_asset).any():
            raise RuntimeError("Binance Margin interest history returned an unexpected asset")
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        duplicates = frame[frame["timestamp"].duplicated(keep=False)]
        if not duplicates.empty:
            conflicting = duplicates.groupby("timestamp")["dailyInterestRate"].nunique() > 1
            if conflicting.any():
                raise RuntimeError("Binance Margin interest history has conflicting duplicate timestamps")
            frame = frame.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
        return frame
