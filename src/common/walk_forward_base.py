from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseWalkForwardAnalyzer(ABC):
    def __init__(
        self,
        hourly_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        params: dict[str, Any],
        *,
        eval_start_time: Any = None,
        hourly_buffer: int = 300,
        daily_buffer_days: int = 200,
        min_segment_bars: int = 100,
    ) -> None:
        self.hourly_df = hourly_df
        self.daily_df = daily_df
        self.params = params
        self.eval_start_time = pd.Timestamp(eval_start_time) if eval_start_time is not None else None
        self.hourly_buffer = int(hourly_buffer)
        self.daily_buffer_days = int(daily_buffer_days)
        self.min_segment_bars = int(min_segment_bars)

    @abstractmethod
    def run_backtest_segment(
        self,
        segment_hourly: pd.DataFrame,
        segment_daily: pd.DataFrame,
        actual_start_time: pd.Timestamp,
        actual_end_time: pd.Timestamp,
        warmup_bars: int,
    ) -> tuple[float, float]:
        raise NotImplementedError

    def run(self, n_splits: int = 5) -> pd.DataFrame:
        n = len(self.hourly_df)
        segment_size = n // n_splits
        results: list[dict[str, Any]] = []

        for i in range(n_splits):
            start_idx = i * segment_size
            end_idx = n if i == (n_splits - 1) else (start_idx + segment_size)

            buf_start_idx = max(0, start_idx - self.hourly_buffer)
            segment_hourly = self.hourly_df.iloc[buf_start_idx:end_idx].copy()
            if len(segment_hourly) < self.min_segment_bars:
                continue

            warmup_bars = int(start_idx - buf_start_idx)
            actual_start_time = pd.Timestamp(self.hourly_df.iloc[start_idx]["datetime"])
            actual_end_time = (
                pd.Timestamp(self.hourly_df.iloc[end_idx - 1]["datetime"])
                if end_idx < n
                else pd.Timestamp(self.hourly_df.iloc[-1]["datetime"])
            )

            if self.eval_start_time is not None and actual_end_time < self.eval_start_time:
                continue

            start_time_buffered = pd.Timestamp(segment_hourly["datetime"].iloc[0])
            end_time = pd.Timestamp(segment_hourly["datetime"].iloc[-1])
            daily_buffer_start = start_time_buffered - pd.Timedelta(days=self.daily_buffer_days)
            segment_daily = self.daily_df[
                (self.daily_df["datetime"] >= daily_buffer_start)
                & (self.daily_df["datetime"] <= end_time)
            ].copy()

            ret, mdd = self.run_backtest_segment(
                segment_hourly,
                segment_daily,
                actual_start_time,
                actual_end_time,
                warmup_bars,
            )
            results.append(
                {
                    "Split": i + 1,
                    "Period": f"{actual_start_time} ~ {actual_end_time}",
                    "Return": float(ret),
                    "MDD": float(mdd),
                }
            )

        if not results:
            return pd.DataFrame(columns=["Split", "Period", "Return", "MDD"])
        return pd.DataFrame(results)
